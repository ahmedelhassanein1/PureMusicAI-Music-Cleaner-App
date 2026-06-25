"""
Phase 2 — sound effect (SFX) detection and attenuation using PANNs / AudioSet.

Scans audio in short time frames, flags non-musical sounds (explosions, whooshes,
punches, etc.), and lowers their volume while trying to keep musical content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

PANNS_SAMPLE_RATE = 32_000
# PANNs CNN14 outputs one score every 320 samples at 32 kHz ≈ 10 ms per frame.
PANNS_HOP_SECONDS = 320 / PANNS_SAMPLE_RATE

_sed_instance = None
_label_to_index: dict[str, int] | None = None

# AudioSet class names we want to reduce (denylist).
# Includes generic SFX plus anime-friendly mappings (ki blast → Explosion/Whoosh, etc.).
DEFAULT_SFX_DENYLIST: list[str] = [
    "Explosion",
    "Burst, pop",
    "Whoosh, swoosh, swish",
    "Punch",
    "Slap, smack",
    "Thump, thud",
    "Boom",
    "Thunder",
    "Thunderstorm",
    "Gunshot, gunfire",
    "Fireworks",
    "Crash",
    "Shatter",
    "Smash, crash",
    "Hum",
    "Static",
    "Roar",
    "Screech",
]

# AudioSet classes that indicate music — we avoid attenuating when these dominate a frame.
DEFAULT_MUSICAL_KEEP: list[str] = [
    "Music",
    "Musical instrument",
    "Singing",
    "Song",
    "Guitar",
    "Piano",
    "Drum",
    "Bass guitar",
    "Violin, fiddle",
    "Synthesizer",
    "Orchestra",
    "Choir",
    "Humming",
]


@dataclass(frozen=True)
class SfxSegment:
    """A time range where unwanted SFX was detected (seconds)."""

    start: float
    end: float
    label: str
    score: float


def detect_sfx_segments(
    audio_path: Path,
    *,
    denylist: list[str] | None = None,
    musical_keep: list[str] | None = None,
    threshold: float = 0.25,
) -> tuple[list[SfxSegment], list[str]]:
    """
    Run PANNs frame-wise detection and return SFX time ranges + labels that fired.

    For each ~10 ms frame we compare denylist scores (explosion, whoosh, …)
    against musical-keep scores. A frame is flagged as SFX when a denylist
    class exceeds `threshold` and beats the strongest musical class.

    Returns:
        segments: merged (start, end) ranges to attenuate
        fired_labels: unique AudioSet labels that triggered at least once
    """
    denylist = denylist or DEFAULT_SFX_DENYLIST
    musical_keep = musical_keep or DEFAULT_MUSICAL_KEEP

    audio_mono = _load_mono_for_panns(audio_path)
    framewise = _run_framewise_detection(audio_mono)

    deny_indices = _resolve_label_indices(denylist)
    keep_indices = _resolve_label_indices(musical_keep)

    fired_labels: set[str] = set()
    raw_segments: list[SfxSegment] = []

    num_frames = framewise.shape[0]
    all_labels = _get_audioset_labels()

    for frame_idx in range(num_frames):
        scores = framewise[frame_idx]

        best_deny_idx, best_deny_score = _best_score(scores, deny_indices)
        best_keep_score = _best_score(scores, keep_indices)[1]

        if best_deny_idx is None or best_deny_score < threshold:
            continue
        if best_deny_score <= best_keep_score:
            continue

        label = all_labels[best_deny_idx]
        fired_labels.add(label)
        start = frame_idx * PANNS_HOP_SECONDS
        end = start + PANNS_HOP_SECONDS
        raw_segments.append(
            SfxSegment(start=start, end=end, label=label, score=float(best_deny_score))
        )

    merged = _merge_sfx_segments(raw_segments)
    logger.info(
        "SFX detection: %d segment(s), labels fired: %s",
        len(merged),
        sorted(fired_labels),
    )
    return merged, sorted(fired_labels)


def attenuate_sfx(
    input_path: Path,
    output_path: Path,
    *,
    strength: float = 1.0,
    segments: list[SfxSegment] | None = None,
    padding_sec: float = 0.03,
    fade_ms: float = 15.0,
    denylist: list[str] | None = None,
    threshold: float = 0.25,
) -> Path:
    """
    Lower volume in SFX regions and write the result to output_path.

    strength: 0.0 = no change, 1.0 = full mute in flagged regions.
    segments: optional pre-detected list; runs detection if omitted.
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 0.0:
        audio, sample_rate = _load_audio(input_path)
        sf.write(output_path, audio, sample_rate)
        return output_path

    audio, sample_rate = _load_audio(input_path)

    if segments is None:
        segments, _ = detect_sfx_segments(
            input_path, denylist=denylist, threshold=threshold
        )

    if not segments:
        logger.info("No SFX segments — copying input unchanged")
        sf.write(output_path, audio, sample_rate)
        return output_path

    processed = _apply_sfx_gain(
        audio=audio,
        sample_rate=sample_rate,
        segments=segments,
        strength=strength,
        padding_sec=padding_sec,
        fade_ms=fade_ms,
    )
    sf.write(output_path, processed, sample_rate)
    logger.info("Wrote SFX-attenuated audio to %s", output_path.name)
    return output_path


def _get_sed():
    """Return a shared SoundEventDetection model (loaded once per worker)."""
    global _sed_instance
    if _sed_instance is None:
        from panns_inference import SoundEventDetection

        _sed_instance = SoundEventDetection(checkpoint_path=None, device="cpu")
    return _sed_instance


def _get_audioset_labels() -> list[str]:
    """AudioSet label strings in the same order as PANNs output columns."""
    global _label_to_index
    from panns_inference import labels

    if _label_to_index is None:
        _label_to_index = {name.lower(): i for i, name in enumerate(labels)}
    return labels


def _resolve_label_indices(target_names: list[str]) -> list[int]:
    """
    Map human-readable AudioSet names to PANNs column indices.
    Tries exact match first, then substring match (case-insensitive).
    """
    all_labels = _get_audioset_labels()
    lookup = {name.lower(): i for i, name in enumerate(all_labels)}
    indices: list[int] = []

    for target in target_names:
        key = target.lower()
        if key in lookup:
            indices.append(lookup[key])
            continue
        for i, name in enumerate(all_labels):
            if key in name.lower() or name.lower() in key:
                indices.append(i)
                break

    return list(dict.fromkeys(indices))


def _best_score(
    frame_scores: np.ndarray, indices: list[int]
) -> tuple[int | None, float]:
    """Return (index, score) of the highest-scoring label in indices, or (None, 0)."""
    if not indices:
        return None, 0.0
    best_idx = max(indices, key=lambda i: frame_scores[i])
    return best_idx, float(frame_scores[best_idx])


def _load_mono_for_panns(path: Path) -> np.ndarray:
    """
    Load audio as mono float32 at 32 kHz — PANNs' expected input format.
    Shape: (samples,)
    """
    audio, _ = librosa.load(path, sr=PANNS_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load full-quality audio for writing output (may be stereo)."""
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.astype(np.float32), int(sample_rate)


def _run_framewise_detection(audio_mono: np.ndarray) -> np.ndarray:
    """
    Run PANNs SoundEventDetection and return per-frame class scores.

    Output shape: (num_frames, 527) — one row per ~10 ms of audio.
    """
    sed = _get_sed()
    batch = audio_mono[None, :]  # (1, samples)
    framewise_output = sed.inference(batch)
    if isinstance(framewise_output, tuple):
        framewise_output = framewise_output[0]
    return np.asarray(framewise_output[0])


def _merge_sfx_segments(segments: list[SfxSegment]) -> list[SfxSegment]:
    """
    Merge consecutive or overlapping SFX frames into longer spans.
    Keeps the label/score from the loudest frame in each merged span.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged: list[SfxSegment] = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        prev = merged[-1]
        if seg.start <= prev.end + PANNS_HOP_SECONDS:
            merged[-1] = SfxSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                label=seg.label if seg.score > prev.score else prev.label,
                score=max(prev.score, seg.score),
            )
        else:
            merged.append(seg)

    return merged


def _segments_to_sample_ranges(
    segments: list[SfxSegment],
    sample_rate: int,
    padding_sec: float,
    num_samples: int,
) -> list[tuple[int, int]]:
    """Convert second-based SFX segments to sample index ranges with padding."""
    ranges: list[tuple[int, int]] = []
    pad = int(padding_sec * sample_rate)
    for seg in segments:
        start = max(0, int(seg.start * sample_rate) - pad)
        end = min(num_samples, int(seg.end * sample_rate) + pad)
        if end > start:
            ranges.append((start, end))
    return _merge_ranges(ranges)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping sample ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        ps, pe = merged[-1]
        if start <= pe:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged


def _apply_sfx_gain(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SfxSegment],
    strength: float,
    padding_sec: float,
    fade_ms: float,
) -> np.ndarray:
    """
    Attenuate SFX regions by (1 - strength) with short linear fades.
    Same envelope idea as speech.py — avoids clicks at boundaries.
    """
    out = audio.copy()
    num_samples = out.shape[0]
    target_gain = 1.0 - strength
    fade_samples = max(1, int(sample_rate * fade_ms / 1000.0))

    for start, end in _segments_to_sample_ranges(
        segments, sample_rate, padding_sec, num_samples
    ):
        fade_in_end = min(end, start + fade_samples)
        for i, idx in enumerate(range(start, fade_in_end)):
            t = (i + 1) / fade_samples
            gain = 1.0 - (1.0 - target_gain) * t
            out[idx] *= gain

        core_start = fade_in_end
        core_end = max(core_start, end - fade_samples)
        if core_end > core_start:
            out[core_start:core_end] *= target_gain

        for i, idx in enumerate(range(core_end, end)):
            t = 1.0 - (i + 1) / fade_samples
            gain = 1.0 - (1.0 - target_gain) * t
            out[idx] *= gain

    return out
