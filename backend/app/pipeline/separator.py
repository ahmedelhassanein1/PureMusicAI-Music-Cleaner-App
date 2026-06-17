"""Thin wrapper around audio-separator for instrumental stem output."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from app.pipeline.model_registry import ModelPreset
from app.settings import settings

logger = logging.getLogger(__name__)


def separate_instrumental(
    input_path: Path,
    output_dir: Path,
    preset: ModelPreset,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Path:
    """
    Run UVR separation and return the path to the instrumental WAV.

    audio-separator writes multiple stems; we pick the instrumental file.
    """
    if progress_callback:
        progress_callback(10, "loading_model")

    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=str(output_dir),
        output_format="WAV",
        model_file_dir=str(settings.models_dir),
    )

    separator.load_model(model_filename=preset.model_filename)

    if progress_callback:
        progress_callback(25, "separating")

    output_files = separator.separate(str(input_path))

    if progress_callback:
        progress_callback(90, "finalizing")

    instrumental = _pick_instrumental(output_files, output_dir)
    logger.info("Separation complete: %s", instrumental)
    return instrumental


def _pick_instrumental(output_files: list[str], output_dir: Path) -> Path:
    """Choose the instrumental stem from separator output filenames."""
    candidates = [Path(f) for f in output_files]

    for path in candidates:
        name = path.name.lower()
        if "instrumental" in name or "(inst)" in name or "_inst" in name:
            return path

    # Fallback: instrumental is often the second stem or any non-vocal file.
    for path in candidates:
        name = path.name.lower()
        if "vocal" not in name:
            return path

    # Last resort: first file in output dir matching .wav
    wavs = sorted(output_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError("No output WAV files from audio-separator")
    return wavs[0]
