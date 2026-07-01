"""
Phase 3 — remix module: combine pipeline stems into the final downloadable WAV.

Takes the UVR instrumental bed plus optional choir overlay and optional SFX
attenuation, then writes one output file. main.py calls this as the last
processing step before marking a job complete.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.pipeline.sfx import SfxSegment, attenuate_sfx

logger = logging.getLogger(__name__)


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
      3. Optionally attenuate detected SFX regions
    """
    choir_gain = float(np.clip(plan.choir_gain, 0.0, 2.0))
    sfx_strength = float(np.clip(plan.sfx_strength, 0.0, 1.0))
    has_choir = (
        choir_gain > 0.0
        and plan.choir_path is not None
        and plan.choir_path.exists()
    )
    has_sfx = bool(plan.sfx_segments) and sfx_strength > 0.0

    if not has_choir and not has_sfx:
        return _passthrough(plan.instrumental_path, plan.output_path)

    mix_dest = plan.output_path
    if has_sfx:
        mix_dest = plan.output_path.parent / "_remix_pre_sfx.wav"
    elif has_choir and mix_dest.resolve() == plan.instrumental_path.resolve():
        mix_dest = plan.output_path.parent / "_remix_mix.wav"

    if has_choir:
        mix_stems(
            plan.instrumental_path,
            mix_dest,
            overlays=[(plan.choir_path, choir_gain)],
        )
    else:
        _passthrough(plan.instrumental_path, mix_dest)

    if has_sfx:
        attenuate_sfx(
            mix_dest,
            plan.output_path,
            strength=sfx_strength,
            segments=list(plan.sfx_segments),
        )
        if mix_dest.exists() and mix_dest.resolve() != plan.output_path.resolve():
            mix_dest.unlink()
        logger.info(
            "Remixed with SFX attenuation (%d segments) → %s",
            len(plan.sfx_segments),
            plan.output_path.name,
        )
    elif mix_dest.resolve() != plan.output_path.resolve():
        shutil.move(mix_dest, plan.output_path)
        logger.info("Remixed stems → %s", plan.output_path.name)
    else:
        logger.info("Remixed stems → %s", plan.output_path.name)

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

    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
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


def _passthrough(source: Path, dest: Path) -> Path:
    """Copy audio unchanged when no remix processing is required."""
    if source.resolve() == dest.resolve():
        return dest
    shutil.copy2(source, dest)
    return dest


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.astype(np.float32), int(sample_rate)
