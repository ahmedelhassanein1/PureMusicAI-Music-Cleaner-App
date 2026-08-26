"""
Phase 2b — embed reference SFX clips with PANNs AudioTagging.

Anime-specific sounds (ki blast, aura, …) miss generic AudioSet labels.
This turns short reference WAVs into 2048-d vectors for later similarity
matching (task 2b.3): librosa @ 32 kHz → AudioTagging → L2-normalized embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

PANNS_SAMPLE_RATE = 32_000
EMBEDDING_DIM = 2048
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

_tagger_instance = None


@dataclass(frozen=True)
class ReferenceEmbedding:
    """One reference clip + its L2-normalized PANNs embedding."""

    name: str
    path: Path
    embedding: np.ndarray
    duration_sec: float


def embed_reference_clip(
    path: Path,
    *,
    name: str | None = None,
) -> ReferenceEmbedding:
    """Load one clip → PANNs embedding. `name` defaults to the file stem."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference clip not found: {path}")

    audio = _load_mono_for_panns(path)
    if audio.size == 0:
        raise ValueError(f"Reference clip is empty: {path}")

    embedding = _embed_waveform(audio)
    clip_name = name if name is not None else path.stem
    duration_sec = float(audio.shape[0]) / PANNS_SAMPLE_RATE

    logger.info(
        "Embedded reference '%s' (%.2fs) → %d-d vector",
        clip_name,
        duration_sec,
        embedding.shape[0],
    )
    return ReferenceEmbedding(
        name=clip_name,
        path=path.resolve(),
        embedding=embedding,
        duration_sec=duration_sec,
    )


def embed_reference_clips(
    paths: list[Path] | list[str],
) -> list[ReferenceEmbedding]:
    """Embed several clips; skip bad paths with a warning."""
    results: list[ReferenceEmbedding] = []
    for raw in paths:
        try:
            results.append(embed_reference_clip(Path(raw)))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping reference clip: %s", exc)
    return results


def load_reference_bank(
    directory: Path | str,
    *,
    recursive: bool = False,
) -> list[ReferenceEmbedding]:
    """Embed all audio files in a folder (e.g. test/fixtures/references/)."""
    root = Path(directory)
    if not root.is_dir():
        logger.warning("Reference bank folder not found: %s", root)
        return []

    pattern = "**/*" if recursive else "*"
    paths = sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
    )
    if not paths:
        logger.info("No reference audio found in %s", root)
        return []

    logger.info("Loading reference bank from %s (%d clip(s))", root, len(paths))
    return embed_reference_clips(paths)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; for L2-normalized vectors this is just a · b."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _get_audio_tagger():
    """Shared AudioTagging model (loaded once; checkpoint_path=None auto-downloads)."""
    global _tagger_instance
    if _tagger_instance is None:
        from panns_inference import AudioTagging

        _tagger_instance = AudioTagging(checkpoint_path=None, device="cpu")
        logger.info("Loaded PANNs AudioTagging model for custom SFX embeddings")
    return _tagger_instance


def _load_mono_for_panns(path: Path) -> np.ndarray:
    """Load mono float32 @ 32 kHz — same input space as sfx.py."""
    audio, _ = librosa.load(str(path), sr=PANNS_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _embed_waveform(audio_mono: np.ndarray) -> np.ndarray:
    """PANNs inference → L2-normalized embedding (class scores discarded)."""
    tagger = _get_audio_tagger()
    batch = np.asarray(audio_mono, dtype=np.float32)[None, :]
    _clipwise, embedding = tagger.inference(batch)

    vector = np.asarray(embedding[0], dtype=np.float32).reshape(-1)
    if vector.shape[0] != EMBEDDING_DIM:
        logger.warning(
            "Unexpected embedding size %d (expected %d)",
            vector.shape[0],
            EMBEDDING_DIM,
        )
    return _l2_normalize(vector)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize so cosine similarity == dot product."""
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32)
