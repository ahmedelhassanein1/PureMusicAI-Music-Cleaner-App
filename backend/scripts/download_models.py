#!/usr/bin/env python3
"""
Download UVR model weights for all curated presets that are not on disk yet.

Run inside the backend container:
  docker compose exec backend python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/download_models.py` from /app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.model_registry import PRESETS, is_separable_preset
from app.settings import settings


def main() -> None:
    from audio_separator.separator import Separator

    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    filenames: list[str] = []
    ensemble_presets: list[str] = []
    for preset in PRESETS:
        if preset.ensemble_preset:
            ensemble_presets.append(preset.ensemble_preset)
            continue
        if is_separable_preset(preset):
            filenames.append(preset.model_filename)
        filenames.extend(preset.extra_model_filenames)

    unique = sorted(set(f for f in filenames if f))
    separator = Separator(model_file_dir=str(models_dir))

    for name in unique:
        path = models_dir / name
        if path.exists():
            print(f"skip  {name} (already present)")
            continue
        print(f"fetch {name} ...")
        separator.load_model(model_filename=name)
        if path.exists():
            print(f"  ok  {name}")
        else:
            print(f"  ??  {name} — load finished but file not found at {path}")

    for preset_name in sorted(set(ensemble_presets)):
        print(f"fetch ensemble preset {preset_name} ...")
        ens = Separator(
            model_file_dir=str(models_dir),
            ensemble_preset=preset_name,
        )
        ens.load_model()
        print(f"  ok  {preset_name}")

    print("done.")


if __name__ == "__main__":
    main()
