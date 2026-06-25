"""
Phase 2 — speech detection and removal.

Detects spoken dialogue with OmniVAD, then attenuates those time ranges in the
audio. Music outside speech segments is left unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_vad_instance = None

# OmniVAD only accepts 16 kHz audio.
DEFAULT_MAX_SPEECH_COVERAGE = 0.35
VAD_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SpeechSegment:
    """Speech time range in seconds (start, end)."""

    start: float
    end: float


def detect_speech_segments(
    audio_path: Path,
    *,
    chunk_seconds: int | None = None,
    overlap_seconds: int = 2,
) -> list[SpeechSegment]:
    """
    Detect speech regions in an audio file using OmniVAD.

    Audio is resampled to 16 kHz before detection (required by OmniVAD).
    Returns a list of (start, end) segments in seconds.
    Empty list means no speech was found.
    Use chunk_seconds for long files to limit memory use.
    """
    vad = _get_vad()
    audio_16k, duration_sec = _load_audio_16k(audio_path)

    try:
        if chunk_seconds is not None:
            result = vad.detect(
                audio_16k,
                sample_rate=VAD_SAMPLE_RATE,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
        else:
            result = vad.detect(audio_16k, sample_rate=VAD_SAMPLE_RATE)
    except Exception:
        logger.exception("OmniVAD detect failed for %s", audio_path.name)
        return []

    timestamps = result.get("timestamps", [])
    segments = [
        SpeechSegment(start=float(start), end=float(end))
        for start, end in timestamps
        if float(end) > float(start)
    ]
    coverage = speech_coverage_fraction(segments, duration_sec)
    logger.info(
        "Detected %d speech segment(s) in %s (%.1f%% coverage)",
        len(segments),
        audio_path.name,
        coverage * 100,
    )
    return segments


def speech_coverage_fraction(
    segments: list[SpeechSegment],
    duration_sec: float,
) -> float:
    """Return the fraction of the timeline covered by merged speech segments."""
    if not segments or duration_sec <= 0:
        return 0.0

    merged = _merge_time_segments(segments)
    covered_sec = sum(end - start for start, end in merged)
    return min(1.0, covered_sec / duration_sec)


def should_apply_speech_removal(
    segments: list[SpeechSegment],
    duration_sec: float,
    *,
    max_coverage: float = DEFAULT_MAX_SPEECH_COVERAGE,
) -> bool:
    """Return whether speech segments are within a typical dialogue-over-music range."""
    if not segments:
        return False
    return speech_coverage_fraction(segments, duration_sec) <= max_coverage


def attenuate_speech(
    input_path: Path,
    output_path: Path,
    *,
    strength: float = 1.0,
    segments: list[SpeechSegment] | None = None,
    padding_sec: float = 0.12,
    fade_ms: float = 20.0,
) -> Path:
    """
    Lower volume in speech regions and write the result to output_path.

    strength: 0.0 = no change, 1.0 = full mute in speech regions.
    segments: optional pre-detected list; runs detection if omitted.
    padding_sec / fade_ms: soften segment edges to avoid clicks.
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 0.0:
        audio, sample_rate = _load_audio(input_path)
        sf.write(output_path, audio, sample_rate)
        return output_path

    audio, sample_rate = _load_audio(input_path)

    if segments is None:
        segments = detect_speech_segments(input_path)

    if not segments:
        logger.info("No speech segments — copying input unchanged")
        sf.write(output_path, audio, sample_rate)
        return output_path

    processed = _apply_segment_gain(
        audio=audio,
        sample_rate=sample_rate,
        segments=segments,
        strength=strength,
        padding_sec=padding_sec,
        fade_ms=fade_ms,
    )
    sf.write(output_path, processed, sample_rate)
    logger.info("Wrote speech-attenuated audio to %s", output_path.name)
    return output_path


def _get_vad():
    """Return a shared OmniVAD instance (loaded once per worker process)."""
    global _vad_instance
    if _vad_instance is None:
        from omnivad import OmniVAD

        _vad_instance = OmniVAD()
    return _vad_instance


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio as float32 array (samples, channels) and sample rate.
    always_2d=True keeps mono and stereo shapes consistent.
    """
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.astype(np.float32), int(sample_rate)


def _load_audio_16k(path: Path) -> tuple[np.ndarray, float]:
    """Load mono audio resampled to 16 kHz; return (samples, duration_sec)."""
    audio, sample_rate = _load_audio(path)
    mono = _to_mono(audio)
    if sample_rate != VAD_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=VAD_SAMPLE_RATE)
    duration_sec = len(mono) / VAD_SAMPLE_RATE
    return np.ascontiguousarray(mono, dtype=np.float32), duration_sec


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels to mono (used for VAD input only)."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _merge_time_segments(segments: list[SpeechSegment]) -> list[tuple[float, float]]:
    """Merge overlapping speech segments in the time domain."""
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda s: s.start)
    merged: list[tuple[float, float]] = [(sorted_segments[0].start, sorted_segments[0].end)]
    for seg in sorted_segments[1:]:
        prev_start, prev_end = merged[-1]
        if seg.start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, seg.end))
        else:
            merged.append((seg.start, seg.end))
    return merged


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) sample ranges into non-overlapping spans."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _segments_to_sample_ranges(
    segments: list[SpeechSegment],
    sample_rate: int,
    padding_sec: float,
    num_samples: int,
) -> list[tuple[int, int]]:
    """
    Convert second-based segments to sample index ranges.
    Applies padding and clamps to the audio length.
    """
    ranges: list[tuple[int, int]] = []
    pad = int(padding_sec * sample_rate)
    for seg in segments:
        start = max(0, int(seg.start * sample_rate) - pad)
        end = min(num_samples, int(seg.end * sample_rate) + pad)
        if end > start:
            ranges.append((start, end))
    return _merge_ranges(ranges)


def _apply_segment_gain(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SpeechSegment],
    strength: float,
    padding_sec: float,
    fade_ms: float,
) -> np.ndarray:
    """
    Attenuate speech regions by (1 - strength) with short linear fades.
    Each segment: fade in → hold at target gain → fade out.
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
