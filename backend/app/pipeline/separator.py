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


def _resolve_output_path(path_str: str, output_dir: Path) -> Path:
    """audio-separator often returns bare filenames, not full paths."""
    path = Path(path_str)
    
    # Case 1: separator already gave us a full path that exists on disk.
    if path.is_absolute() and path.exists():
        return path
    in_output_dir = output_dir / path.name
    
    # Case 2: bare filename — file lives in the job output folder.
    if in_output_dir.exists():
        return in_output_dir

    # Case 3: relative path from the current working directory.
    if path.exists():
        return path.resolve()
    
    # Case 4: best guess — expected location even if not found yet.
    return in_output_dir


def _pick_instrumental(output_files: list[str], output_dir: Path) -> Path:
    """Choose the instrumental stem from separator output filenames."""
    # Turn each returned name/path into a full path under the job output folder.
    candidates = [_resolve_output_path(f, output_dir) for f in output_files]

    # Prefer files whose name clearly marks them as the instrumental stem.
    for path in candidates:
        if path.exists():
            name = path.name.lower()
            if "instrumental" in name or ("inst") in name or "_inst" in name:
                return path
    
    # Otherwise take any existing stem that is not the vocal track.
    for path in candidates:
        if path.exists():
            name = path.name.lower()
            if "vocal" not in name:
                return path

    wavs = sorted(output_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No output WAV files in {output_dir}. Returned: {output_files}"
        )
    return wavs[0]