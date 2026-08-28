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
    category: str = "starter"  # starter | karaoke | classic | ensemble | cleanup
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
        name="Classic HP Vocal",
        description="VR HP Vocal model (3_HP-Vocal-UVR). Vocal-optimized; still outputs a strong instrumental stem.",
        model_filename="3_HP-Vocal-UVR.pth",
        arch="vr",
        category="classic",
    ),
    ModelPreset(
        id="classic_hp2_9",
        name="Classic HP2-9",
        description="VR HP2 series (9_HP2-UVR). Strongest classic instrumental SDR in the HP2 lineup.",
        model_filename="9_HP2-UVR.pth",
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

# --- Cleanup models (post-instrumental polish; not used as main vocal removers) ---
_CLEANUP_PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="denoise_lite",
        name="Denoise Lite",
        description=(
            "UVR DeNoise-Lite — gentle hiss/hum cleanup on an instrumental bed. "
            "Optional pipeline pass; not a vocal remover and not aimed at loud anime SFX."
        ),
        model_filename="UVR-DeNoise-Lite.pth",
        arch="vr",
        category="cleanup",
    ),
    ModelPreset(
        id="denoise",
        name="Denoise",
        description=(
            "UVR DeNoise — standard denoise on an instrumental bed. Stronger than Lite; "
            "may thin air/cymbals more. Still not aimed at loud anime SFX."
        ),
        model_filename="UVR-DeNoise.pth",
        arch="vr",
        category="cleanup",
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
    _STARTER_PRESETS
    + _KARAOKE_PRESETS
    + _CLASSIC_PRESETS
    + _CLEANUP_PRESETS
    + _ENSEMBLE_PRESETS
)

DEFAULT_KARAOKE_MODEL_ID = "karaoke_mdx_kara2"
REFERENCE_INSTRUMENTAL_MODEL_ID = "balanced"
DENOISE_LITE_MODEL_ID = "denoise_lite"
DENOISE_MODEL_ID = "denoise"
DENOISE_PRESET_IDS = frozenset({DENOISE_LITE_MODEL_ID, DENOISE_MODEL_ID})


def get_preset(model_id: str) -> ModelPreset | None:
    for preset in PRESETS:
        if preset.id == model_id:
            return preset
    return None


def is_denoise_preset(model_id: str) -> bool:
    """True when ``model_id`` is a registered cleanup denoise preset."""
    return model_id in DENOISE_PRESET_IDS


def list_cleanup_presets() -> list[dict[str, str | bool]]:
    """Cleanup-only presets (denoise) for optional post-UVR polish UI."""
    return [_preset_to_dict(p) for p in PRESETS if p.category == "cleanup"]


def list_presets() -> list[dict[str, str | bool]]:
    """Curated presets for the default model picker (tasks 3.1 + 3.2).

    Cleanup models (e.g. denoise) are omitted — they are optional post-passes,
    not main vocal removers.
    """
    return [
        _preset_to_dict(p) for p in PRESETS if p.category != "cleanup"
    ]


def list_karaoke_presets() -> list[dict[str, str | bool]]:
    """Karaoke-only presets for the choir-preservation sub-model dropdown."""
    return [_preset_to_dict(p) for p in PRESETS if p.category == "karaoke"]


def is_ensemble_preset(preset: ModelPreset) -> bool:
    """True when this preset runs audio-separator's multi-model ensemble path."""
    if preset.arch == "ensemble":
        return True
    if preset.ensemble_preset:
        return True
    return bool(preset.model_filename and preset.extra_model_filenames)


def is_separable_preset(preset: ModelPreset) -> bool:
    """True when audio-separator can run this preset."""
    if preset.ensemble_preset:
        return True
    if preset.model_filename and preset.extra_model_filenames:
        return True
    return bool(preset.model_filename)


def _preset_to_dict(preset: ModelPreset) -> dict[str, str | bool]:
    return {
        "id": preset.id,
        "name": preset.name,
        "description": preset.description,
        "arch": preset.arch,
        "category": preset.category,
        "is_karaoke": preset.is_karaoke,
    }


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
