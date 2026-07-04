"""
Phase 3 — remix module: combine pipeline stems into the final downloadable WAV.

Takes the UVR instrumental bed plus optional choir overlay and optional SFX
attenuation, then writes one output file. main.py calls this as the last
processing step before marking a job complete.

Phase 4 polish (tasks 4.1–4.2):
  - Cosine crossfades at SFX segment boundaries (no clicks between bed and ducked regions)
  - Peak normalization to -1 dBFS before write
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.pipeline.sfx import SfxSegment

logger = logging.getLogger(__name__)

# Phase 4 output polish defaults.
TARGET_PEAK_DBFS = -1.0
CROSSFADE_MS = 20.0
SFX_PADDING_SEC = 0.03


@dataclass(frozen=True)
class RemixPlan:
    """Inputs for the final mix-down step of a job."""

    instrumental_path: Path
    output_path: Path
    choir_path: Path | None = None
    choir_gain: float = 0.0
    sfx_segments: tuple[SfxSegment, ...] = ()
    sfx_strength: float = 0.0


def finalize_instrumental(plan: RemixPlan) -> Path:
    """
    Build the final instrumental WAV from a remix plan.

    Order of operations:
      1. Start from the instrumental bed
      2. Optionally add a choir/backing stem at ``choir_gain``
      3. Optionally attenuate detected SFX regions (with boundary crossfades)
      4. Normalize peak to ``TARGET_PEAK_DBFS`` and write the output file
    """
    choir_gain = float(np.clip(plan.choir_gain, 0.0, 2.0))
    sfx_strength = float(np.clip(plan.sfx_strength, 0.0, 1.0))
    has_choir = (
        choir_gain > 0.0
        and plan.choir_path is not None
        and plan.choir_path.exists()
    )
    has_sfx = bool(plan.sfx_segments) and sfx_strength > 0.0

    audio, sample_rate = _load_audio(plan.instrumental_path)

    if has_choir:
        audio = _mix_overlays_on_audio(
            audio,
            sample_rate,
            overlays=[(plan.choir_path, choir_gain)],
        )
        logger.info("Mixed choir overlay onto instrumental bed")

    if has_sfx:
        audio = _apply_segment_crossfade(
            audio,
            sample_rate,
            list(plan.sfx_segments),
            strength=sfx_strength,
            fade_ms=CROSSFADE_MS,
            padding_sec=SFX_PADDING_SEC,
        )
        logger.info(
            "Applied SFX crossfade attenuation (%d segment(s))",
            len(plan.sfx_segments),
        )

    audio = _normalize_peak_dbfs(audio, target_db=TARGET_PEAK_DBFS)
    sf.write(plan.output_path, audio, sample_rate)
    logger.info(
        "Remixed stems → %s (peak normalized to %.1f dBFS)",
        plan.output_path.name,
        TARGET_PEAK_DBFS,
    )
    return plan.output_path


def mix_stems(
    base_path: Path,
    output_path: Path,
    *,
    overlays: list[tuple[Path | None, float]],
) -> Path:
    """
    Sum the base instrumental with one or more weighted overlay stems.

    Each overlay is (path, gain). Overlays with gain <= 0 or missing files are
    skipped. Output is clipped to [-1, 1].
    """
    base, sample_rate = _load_audio(base_path)
    mixed = _mix_overlays_on_audio(base, sample_rate, overlays)
    sf.write(output_path, mixed, sample_rate)
    logger.info(
        "Mixed %s + %d overlay(s) → %s",
        base_path.name,
        sum(1 for p, g in overlays if g > 0 and p and p.exists()),
        output_path.name,
    )
    return output_path


def remix_with_choir(
    instrumental_path: Path,
    choir_path: Path,
    output_path: Path,
    *,
    choir_gain: float = 1.0,
) -> Path:
    """Add a choir/backing candidate stem onto the instrumental bed."""
    return finalize_instrumental(
        RemixPlan(
            instrumental_path=instrumental_path,
            output_path=output_path,
            choir_path=choir_path,
            choir_gain=choir_gain,
        )
    )


def _mix_overlays_on_audio(
    base: np.ndarray,
    sample_rate: int,
    overlays: list[tuple[Path | None, float]],
) -> np.ndarray:
    """Sum weighted overlay stems onto a base waveform (in memory)."""
    mixed = base.copy()

    for overlay_path, gain in overlays:
        gain = float(np.clip(gain, 0.0, 2.0))
        if gain <= 0.0 or overlay_path is None or not overlay_path.exists():
            continue
        overlay, overlay_sr = _load_audio(overlay_path)
        if overlay_sr != sample_rate:
            raise ValueError(
                f"Overlay sample rate {overlay_sr} != base {sample_rate} "
                f"({overlay_path.name})"
            )
        length = min(len(mixed), len(overlay))
        mixed[:length] += gain * overlay[:length]

    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _apply_segment_crossfade(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SfxSegment],
    *,
    strength: float,
    fade_ms: float,
    padding_sec: float,
) -> np.ndarray:
    """
    4.1 — Attenuate flagged SFX spans with cosine crossfades at each boundary.

    Instead of hard-switching gain at segment edges (which can click), each
    boundary ramps smoothly between full level and the ducked level using an
    equal-power sine curve.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or not segments:
        return audio

    out = audio.copy()
    num_samples = out.shape[0]
    target_gain = 1.0 - strength
    fade_samples = max(1, int(sample_rate * fade_ms / 1000.0))

    for start, end in _segment_sample_ranges(
        segments, sample_rate, padding_sec, num_samples
    ):
        fade_in_end = min(end, start + fade_samples)
        in_len = fade_in_end - start
        if in_len > 0:
            curve = _crossfade_curve(in_len)
            for ch in range(out.shape[1]):
                out[start:fade_in_end, ch] *= 1.0 - (1.0 - target_gain) * curve

        core_start = fade_in_end
        core_end = max(core_start, end - fade_samples)
        if core_end > core_start:
            out[core_start:core_end] *= target_gain

        fade_out_start = core_end
        out_len = end - fade_out_start
        if out_len > 0:
            curve = _crossfade_curve(out_len)
            for ch in range(out.shape[1]):
                out[fade_out_start:end, ch] *= 1.0 - (1.0 - target_gain) * curve[::-1]

    return out.astype(np.float32)


def _normalize_peak_dbfs(audio: np.ndarray, target_db: float = TARGET_PEAK_DBFS) -> np.ndarray:
    """
    4.2 — Scale audio so the absolute peak equals ``target_db`` dBFS.

    Float WAV peaks live in [-1.0, 1.0] where 0 dBFS = 1.0.
    -1 dBFS ≈ 0.891 linear — leaves a little headroom below clipping.
    """
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-8:
        return audio

    target_linear = 10.0 ** (target_db / 20.0)
    scaled = audio * (target_linear / peak)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def _crossfade_curve(length: int) -> np.ndarray:
    """Equal-power fade from 0 → 1 over ``length`` samples."""
    if length <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.sin(t * np.pi / 2.0) ** 2


def _segment_sample_ranges(
    segments: list[SfxSegment],
    sample_rate: int,
    padding_sec: float,
    num_samples: int,
) -> list[tuple[int, int]]:
    """Convert second-based SFX segments to merged sample ranges with padding."""
    pad = int(padding_sec * sample_rate)
    ranges: list[tuple[int, int]] = []
    for seg in segments:
        start = max(0, int(seg.start * sample_rate) - pad)
        end = min(num_samples, int(seg.end * sample_rate) + pad)
        if end > start:
            ranges.append((start, end))
    return _merge_sample_ranges(ranges)


def _merge_sample_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping sample ranges."""
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


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.astype(np.float32), int(sample_rate)
