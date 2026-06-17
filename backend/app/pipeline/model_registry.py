"""UVR model presets exposed to the frontend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    id: str
    name: str
    description: str
    model_filename: str
    arch: str  # "vr", "mdx", "mdxc"


# Three starter presets — audio-separator downloads weights on first use.
PRESETS: list[ModelPreset] = [
    ModelPreset(
        id="fast",
        name="Fast",
        description="Quick VR Arch model. Best for CPU or short clips.",
        model_filename="1_HP-UVR.pth",
        arch="vr",
    ),
    ModelPreset(
        id="balanced",
        name="Balanced",
        description="MDX-Net instrumental model. Good speed/quality tradeoff.",
        model_filename="UVR-MDX-NET-Inst_HQ_3.onnx",
        arch="mdx",
    ),
    ModelPreset(
        id="high_quality",
        name="High Quality",
        description="Roformer model. Slower but cleaner instrumental output.",
        model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        arch="mdxc",
    ),
]


def get_preset(model_id: str) -> ModelPreset | None:
    for preset in PRESETS:
        if preset.id == model_id:
            return preset
    return None


def list_presets() -> list[dict[str, str]]:
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "arch": p.arch,
        }
        for p in PRESETS
    ]
