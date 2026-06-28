"""UVR model presets exposed to the frontend and full catalog via audio-separator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPreset:
    id: str
    name: str
    description: str
    model_filename: str
    arch: str  # "vr", "mdx", "mdxc", "ensemble"
    category: str = "starter"  # starter | karaoke | classic | ensemble
    is_karaoke: bool = False
    extra_model_filenames: tuple[str, ...] = field(default_factory=tuple)
    ensemble_preset: str | None = None
    ensemble_algorithm: str = "avg_wave"


# --- Phase 1 starter presets (unchanged defaults) ---
_STARTER_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="fast",
        name="Fast",
        description="Quick VR Arch model. Best for CPU or short clips.",
        model_filename="1_HP-UVR.pth",
        arch="vr",
        category="starter",
    ),
    ModelPreset(
        id="balanced",
        name="Balanced",
        description="MDX-Net instrumental model. Good speed/quality tradeoff.",
        model_filename="UVR-MDX-NET-Inst_HQ_3.onnx",
        arch="mdx",
        category="starter",
    ),
    ModelPreset(
        id="high_quality",
        name="High Quality",
        description="Roformer model. Slower but cleaner instrumental output.",
        model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        arch="mdxc",
        category="starter",
    ),
]

# --- 3.1 Karaoke UVR presets (lead vocal removal, choir-friendly) ---
_KARAOKE_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="karaoke_mdx_kara2",
        name="Karaoke MDX (KARA 2)",
        description="MDX karaoke model. Removes lead vocals; keeps backing/choir in the instrumental.",
        model_filename="UVR_MDXNET_KARA_2.onnx",
        arch="mdx",
        category="karaoke",
        is_karaoke=True,
    ),
    ModelPreset(
        id="karaoke_mdx_kara",
        name="Karaoke MDX (KARA)",
        description="Original MDX karaoke model. Older but lighter than KARA 2.",
        model_filename="UVR_MDXNET_KARA.onnx",
        arch="mdx",
        category="karaoke",
        is_karaoke=True,
    ),
    ModelPreset(
        id="karaoke_vr_hp_karaoke",
        name="Karaoke VR (HP-Karaoke)",
        description="VR karaoke architecture. Good for backing-vocal preservation on older tracks.",
        model_filename="5_HP-Karaoke-UVR.pth",
        arch="vr",
        category="karaoke",
        is_karaoke=True,
    ),
]

# --- 3.2 UVR Classic + ensemble presets ---
_CLASSIC_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="classic_hp2",
        name="Classic HP2",
        description="UVR HP2 — reliable classic vocal remover from the original UVR lineup.",
        model_filename="2_HP-UVR.pth",
        arch="vr",
        category="classic",
    ),
    ModelPreset(
        id="classic_hp3",
        name="Classic HP3",
        description="UVR HP3 — popular classic model; slightly heavier than HP2.",
        model_filename="3_HP-UVR.pth",
        arch="vr",
        category="classic",
    ),
    ModelPreset(
        id="classic_hp5",
        name="Classic HP5",
        description="UVR HP5 — strong classic model for difficult vocal separation.",
        model_filename="5_HP-UVR.pth",
        arch="vr",
        category="classic",
    ),
    ModelPreset(
        id="classic_vr_deecho",
        name="Classic De-Echo",
        description="VR de-echo / de-reverb model. Useful on live or reverberant recordings.",
        model_filename="UVR-DeEcho-DeReverb.pth",
        arch="vr",
        category="classic",
    ),
]

_ENSEMBLE_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="ensemble_vocal_balanced",
        name="Ensemble: Vocal Balanced",
        description="Built-in audio-separator preset blending multiple vocal models for balanced separation.",
        model_filename="",
        arch="ensemble",
        category="ensemble",
        ensemble_preset="vocal_balanced",
    ),
    ModelPreset(
        id="ensemble_karaoke",
        name="Ensemble: Karaoke",
        description="Built-in karaoke ensemble preset — optimized for lead-vocal removal.",
        model_filename="",
        arch="ensemble",
        category="ensemble",
        is_karaoke=True,
        ensemble_preset="karaoke",
    ),
    ModelPreset(
        id="ensemble_inst_karaoke",
        name="Ensemble: Inst + Karaoke",
        description="MDX instrumental HQ + KARA 2 ensembled for clean beds with choir retained.",
        model_filename="UVR-MDX-NET-Inst_HQ_3.onnx",
        arch="ensemble",
        category="ensemble",
        is_karaoke=True,
        extra_model_filenames=("UVR_MDXNET_KARA_2.onnx",),
        ensemble_algorithm="max_fft",
    ),
]

PRESETS: list[ModelPreset] = (
    _STARTER_PRESETS + _KARAOKE_PRESETS + _CLASSIC_PRESETS + _ENSEMBLE_PRESETS
)


def get_preset(model_id: str) -> ModelPreset | None:
    for preset in PRESETS:
        if preset.id == model_id:
            return preset
    return None


def list_presets() -> list[dict[str, str | bool]]:
    """Curated presets for the default model picker (tasks 3.1 + 3.2)."""
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "arch": p.arch,
            "category": p.category,
            "is_karaoke": p.is_karaoke,
        }
        for p in PRESETS
    ]


def list_all_models() -> list[dict[str, Any]]:
    """
    Full UVR catalog from audio-separator (proxies CLI --list_models).

    Returns JSON-serializable dicts with model_filename, arch, friendly_name, stems, etc.
    """
    from audio_separator.separator import Separator

    separator = Separator(
        model_file_dir=str(settings.models_dir),
        info_only=True,
    )
    raw = separator.list_models()

    normalized: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            normalized.append(_normalize_catalog_entry(entry))
        else:
            normalized.append({"model_filename": str(entry)})

    logger.info("Listed %d models from audio-separator catalog", len(normalized))
    return normalized


def _normalize_catalog_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Map audio-separator list_models keys to a stable API shape."""
    filename = (
        entry.get("model_filename")
        or entry.get("filename")
        or entry.get("model")
        or ""
    )
    return {
        "model_filename": filename,
        "arch": entry.get("arch") or entry.get("architecture") or "",
        "friendly_name": entry.get("friendly_name") or entry.get("name") or filename,
        "stems": entry.get("stems") or entry.get("output_stems") or entry.get("Output Stems (SDR)"),
        "sdr": entry.get("sdr"),
    }
