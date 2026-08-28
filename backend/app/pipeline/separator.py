"""Thin wrapper around audio-separator for instrumental, vocal, and cleanup stems."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from app.pipeline.model_registry import (
    DENOISE_LITE_MODEL_ID,
    ModelPreset,
    get_preset,
    is_denoise_preset,
    is_ensemble_preset,
)
from app.settings import settings

logger = logging.getLogger(__name__)

# Pipeline-managed names — never treat these as fresh UVR separator output.
_MANAGED_STEMS = frozenset(
    {
        "instrumental_raw.wav",
        "instrumental.wav",
        "instrumental_denoised.wav",
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


def denoise_instrumental(
    input_path: Path,
    output_dir: Path,
    progress_callback: Callable[[int, str], None] | None = None,
    *,
    model_id: str = DENOISE_LITE_MODEL_ID,
    dest_name: str = "instrumental_denoised.wav",
) -> Path:
    """
    Run a UVR denoise cleanup model on an instrumental bed.

    ``model_id`` should be a cleanup preset (``denoise_lite`` or ``denoise``).
    First call for a given model may download its ``.pth`` into the models dir.
    Returns the stable ``dest_name`` path (default ``instrumental_denoised.wav``).
    """
    if not is_denoise_preset(model_id):
        raise ValueError(f"Not a denoise cleanup preset: {model_id}")
    preset = get_preset(model_id)
    if preset is None:
        raise ValueError(f"Missing cleanup preset: {model_id}")

    output_files = _separate(
        input_path=input_path,
        output_dir=output_dir,
        preset=preset,
        progress_callback=progress_callback,
    )
    clean = _pick_denoise_clean(output_files, output_dir)
    canonical = canonicalize_instrumental(clean, output_dir, dest_name=dest_name)
    logger.info(
        "Denoise complete (%s): %s (from %s)",
        preset.id,
        canonical.name,
        clean.name,
    )
    return canonical


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


def _looks_like_denoise_clean(name: str) -> bool:
    """True for UVR 'No Noise' / clean bed stems."""
    n = name.lower()
    return (
        "no noise" in n
        or "nonoise" in n
        or "no_noise" in n
        or "(clean)" in n
        or "_clean" in n
    )


def _looks_like_denoise_residual(name: str) -> bool:
    """True for the residual Noise stem (not the bed we want to keep)."""
    n = name.lower()
    if _looks_like_denoise_clean(n):
        return False
    # Strip model-family tokens so 'DeNoise' does not count as residual 'noise'.
    stripped = n.replace("denoise", "").replace("de-noise", "")
    return "noise" in stripped


def _pick_denoise_clean(output_files: list[str], output_dir: Path) -> Path:
    """
    Choose the cleaned bed from a DeNoise model run.

    Prefer explicit No Noise / clean names; never prefer the residual Noise stem.
    Falls back to the normal instrumental picker if naming is unfamiliar.
    """
    candidates = [
        _resolve_output_path(f, output_dir)
        for f in output_files
        if _resolve_output_path(f, output_dir).name not in _MANAGED_STEMS
    ]
    existing = [p for p in candidates if p.exists() and p.name not in _MANAGED_STEMS]

    for path in existing:
        if _looks_like_denoise_clean(path.name):
            logger.info("Picked denoise clean stem: %s", path.name)
            return path

    non_residual = [p for p in existing if not _looks_like_denoise_residual(p.name)]
    if len(non_residual) == 1:
        logger.info("Picked denoise non-residual stem: %s", non_residual[0].name)
        return non_residual[0]
    if non_residual:
        # Prefer instrumental-like names among non-residual outputs.
        for path in non_residual:
            name = path.name.lower()
            if "instrumental" in name or "_inst" in name:
                logger.info("Picked denoise instrumental-like stem: %s", path.name)
                return path
        logger.info("Picked first denoise non-residual stem: %s", non_residual[0].name)
        return non_residual[0]

    logger.warning(
        "No explicit denoise clean stem found; falling back to instrumental picker. "
        "Returned: %s",
        output_files,
    )
    return _pick_instrumental(output_files, output_dir)


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
