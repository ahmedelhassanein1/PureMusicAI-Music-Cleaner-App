"""Thin wrapper around audio-separator for instrumental and vocal stems."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from app.pipeline.model_registry import ModelPreset, is_ensemble_preset
from app.settings import settings

logger = logging.getLogger(__name__)

# Pipeline-managed names — never treat these as fresh UVR separator output.
_MANAGED_STEMS = frozenset(
    {
        "instrumental_raw.wav",
        "instrumental.wav",
        "karaoke_instrumental_raw.wav",
        "choir_candidate.wav",
        "choir_preserved.wav",
        "_remix_pre_sfx.wav",
        "_remix_mix.wav",
        "vocals_stem.wav",
        "after_vocals.wav",
    }
)


def separate_instrumental(
    input_path: Path,
    output_dir: Path,
    preset: ModelPreset,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Path:
    """
    Run UVR separation and return the path to the instrumental WAV.

    Supports single models and audio-separator ensemble presets / multi-model
    ensembling configured on ``ModelPreset``.
    """
    output_files = _separate(
        input_path=input_path,
        output_dir=output_dir,
        preset=preset,
        progress_callback=progress_callback,
    )
    instrumental = _pick_instrumental(output_files, output_dir)
    logger.info("Separation complete: %s", instrumental)
    return instrumental


def canonicalize_instrumental(
    uvr_path: Path,
    output_dir: Path,
    *,
    dest_name: str = "instrumental_raw.wav",
) -> Path:
    """
    Copy the UVR instrumental stem to a stable path for download / post-processing.

    Uses copy (not move) so the original UVR-named file remains in the job folder.
    """
    canonical = output_dir / dest_name
    if uvr_path.resolve() != canonical.resolve():
        shutil.copy2(uvr_path, canonical)
    logger.info("Canonical instrumental: %s (from %s)", canonical.name, uvr_path.name)
    return canonical


def separate_vocals(
    input_path: Path,
    output_dir: Path,
    model_filename: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Path:
    """Run a vocal UVR model and return the vocals stem WAV."""
    output_files = _separate(
        input_path=input_path,
        output_dir=output_dir,
        preset=_single_model_preset(model_filename),
        progress_callback=progress_callback,
    )
    vocals = _pick_vocals(output_files, output_dir)
    logger.info("Vocal separation complete: %s", vocals)
    return vocals


def _single_model_preset(model_filename: str) -> ModelPreset:
    """Minimal preset wrapper for call sites that only have a filename."""
    return ModelPreset(
        id="adhoc",
        name="adhoc",
        description="",
        model_filename=model_filename,
        arch="mdx",
    )


def _separate(
    input_path: Path,
    output_dir: Path,
    preset: ModelPreset,
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[str]:
    if progress_callback:
        progress_callback(10, "loading_model")

    from audio_separator.separator import Separator

    separator_kwargs: dict[str, str] = {
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "model_file_dir": str(settings.models_dir),
    }

    if preset.ensemble_preset:
        separator_kwargs["ensemble_preset"] = preset.ensemble_preset
        logger.info(
            "Loading ensemble preset %s (algorithm=%s)",
            preset.ensemble_preset,
            preset.ensemble_algorithm,
        )
    elif is_ensemble_preset(preset):
        separator_kwargs["ensemble_algorithm"] = preset.ensemble_algorithm
        logger.info(
            "Loading custom ensemble: %s + %s (%s)",
            preset.model_filename,
            ", ".join(preset.extra_model_filenames),
            preset.ensemble_algorithm,
        )

    separator = Separator(**separator_kwargs)

    if preset.ensemble_preset:
        separator.load_model()
    elif preset.extra_model_filenames:
        model_filenames = [preset.model_filename, *preset.extra_model_filenames]
        separator.load_model(model_filename=model_filenames)
    else:
        separator.load_model(model_filename=preset.model_filename)

    if progress_callback:
        progress_callback(25, "separating")

    output_files = separator.separate(str(input_path))

    if progress_callback:
        progress_callback(90, "finalizing")

    return output_files


def _resolve_output_path(path_str: str, output_dir: Path) -> Path:
    """audio-separator often returns bare filenames, not full paths."""
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path
    in_output_dir = output_dir / path.name

    if in_output_dir.exists():
        return in_output_dir

    if path.exists():
        return path.resolve()

    return in_output_dir


def _pick_instrumental(output_files: list[str], output_dir: Path) -> Path:
    """Choose the instrumental stem from separator output filenames."""
    candidates = [
        _resolve_output_path(f, output_dir)
        for f in output_files
        if _resolve_output_path(f, output_dir).name not in _MANAGED_STEMS
    ]

    for path in candidates:
        if not path.exists():
            continue
        name = path.name.lower()
        if name in _MANAGED_STEMS:
            continue
        if "instrumental" in name or "_inst" in name:
            return path

    for path in candidates:
        if path.exists() and path.name not in _MANAGED_STEMS:
            name = path.name.lower()
            if "vocal" not in name:
                return path

    for path in sorted(output_dir.glob("*.wav")):
        if path.name in _MANAGED_STEMS:
            continue
        name = path.name.lower()
        if "instrumental" in name or "_inst" in name:
            if "vocal" not in name:
                return path

    wavs = [
        p
        for p in sorted(output_dir.glob("*.wav"))
        if p.name not in _MANAGED_STEMS and "vocal" not in p.name.lower()
    ]
    if not wavs:
        raise FileNotFoundError(
            f"No UVR instrumental WAV in {output_dir}. Returned: {output_files}"
        )
    return wavs[0]


def _pick_vocals(output_files: list[str], output_dir: Path) -> Path:
    """Choose the vocals stem from separator output filenames."""
    candidates = [_resolve_output_path(f, output_dir) for f in output_files]

    for path in candidates:
        if path.exists():
            name = path.name.lower()
            if "vocal" in name and "instrumental" not in name:
                return path

    for path in candidates:
        if path.exists():
            name = path.name.lower()
            if "vocal" in name:
                return path

    wavs = sorted(output_dir.glob("*Vocal*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No vocal WAV files in {output_dir}. Returned: {output_files}"
        )
    return wavs[0]
