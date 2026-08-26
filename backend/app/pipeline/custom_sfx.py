"""
Phase 2b — custom SFX via PANNs embeddings.

2b.2: reference WAV → 2048-d L2-normalized embedding
2b.3: sliding-window cosine match against a mix
2b.4: matched regions are attenuated in remix.py (same crossfade as generic SFX)
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

# Window length clamped so very short/long refs still get a sane search size.
_MIN_WINDOW_SEC = 0.40
_MAX_WINDOW_SEC = 2.00
_DEFAULT_THRESHOLD = 0.70
_EMBED_BATCH_SIZE = 8

_tagger_instance = None


@dataclass(frozen=True)
class ReferenceEmbedding:
    """One reference clip + its L2-normalized PANNs embedding."""

    name: str
    path: Path
    embedding: np.ndarray
    duration_sec: float


@dataclass(frozen=True)
class MatchSegment:
    """Time range in a mix that matched a reference clip (seconds)."""

    start: float
    end: float
    label: str
    score: float


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


def match_references_in_mix(
    mix_path: Path | str,
    references: list[ReferenceEmbedding],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    hop_sec: float | None = None,
    window_sec: float | None = None,
) -> list[MatchSegment]:
    """Slide a window over the mix; return regions similar to any reference."""
    if not references:
        return []

    mix_path = Path(mix_path)
    mix = _load_mono_for_panns(mix_path)
    if mix.size == 0:
        raise ValueError(f"Mix is empty: {mix_path}")

    threshold = float(np.clip(threshold, 0.0, 1.0))
    all_hits: list[MatchSegment] = []

    for ref in references:
        hits = _match_one_reference(
            mix,
            ref,
            threshold=threshold,
            hop_sec=hop_sec,
            window_sec=window_sec,
        )
        all_hits.extend(hits)

    merged = _merge_match_segments(all_hits)
    logger.info(
        "Custom SFX match: %d segment(s) from %d reference(s) in %s",
        len(merged),
        len(references),
        mix_path.name,
    )
    return merged


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


def _match_one_reference(
    mix: np.ndarray,
    ref: ReferenceEmbedding,
    *,
    threshold: float,
    hop_sec: float | None,
    window_sec: float | None,
) -> list[MatchSegment]:
    """Sliding-window match for a single reference embedding."""
    win_sec = window_sec if window_sec is not None else ref.duration_sec
    win_sec = float(np.clip(win_sec, _MIN_WINDOW_SEC, _MAX_WINDOW_SEC))
    hop = hop_sec if hop_sec is not None else win_sec / 2.0
    hop = max(0.05, float(hop))

    win_samples = max(1, int(round(win_sec * PANNS_SAMPLE_RATE)))
    hop_samples = max(1, int(round(hop * PANNS_SAMPLE_RATE)))

    if mix.shape[0] < win_samples:
        # Mix shorter than window — score the whole clip once.
        score = cosine_similarity(_embed_waveform(mix), ref.embedding)
        if score >= threshold:
            return [
                MatchSegment(
                    start=0.0,
                    end=float(mix.shape[0]) / PANNS_SAMPLE_RATE,
                    label=ref.name,
                    score=score,
                )
            ]
        return []

    starts = list(range(0, mix.shape[0] - win_samples + 1, hop_samples))
    windows = [mix[s : s + win_samples] for s in starts]
    embeddings = _embed_waveforms_batched(windows)

    # Batched cosine: refs are normalized → score = embedding · ref
    scores = embeddings @ ref.embedding.astype(np.float32)

    raw: list[MatchSegment] = []
    for start_samp, score in zip(starts, scores):
        if float(score) < threshold:
            continue
        start = start_samp / PANNS_SAMPLE_RATE
        raw.append(
            MatchSegment(
                start=start,
                end=start + win_sec,
                label=ref.name,
                score=float(score),
            )
        )
    return raw


def _merge_match_segments(segments: list[MatchSegment]) -> list[MatchSegment]:
    """Merge overlapping/adjacent hits; keep best score + its label."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: (s.start, -s.score))
    merged: list[MatchSegment] = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        prev = merged[-1]
        # Allow a small gap (~one hop) so near-hits become one span.
        if seg.start <= prev.end + 0.05:
            merged[-1] = MatchSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                label=seg.label if seg.score > prev.score else prev.label,
                score=max(prev.score, seg.score),
            )
        else:
            merged.append(seg)
    return merged


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
    return _embed_waveforms_batched([audio_mono])[0]


def _embed_waveforms_batched(
    waveforms: list[np.ndarray],
    *,
    batch_size: int = _EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Embed many mono clips (zero-pad each batch to a common length)."""
    if not waveforms:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    tagger = _get_audio_tagger()
    out_rows: list[np.ndarray] = []

    for i in range(0, len(waveforms), batch_size):
        chunk = waveforms[i : i + batch_size]
        max_len = max(w.shape[0] for w in chunk)
        batch = np.zeros((len(chunk), max_len), dtype=np.float32)
        for row, wave in enumerate(chunk):
            batch[row, : wave.shape[0]] = wave

        _clipwise, embedding = tagger.inference(batch)
        vectors = np.asarray(embedding, dtype=np.float32)
        for row in vectors:
            out_rows.append(_l2_normalize(row.reshape(-1)))

    return np.stack(out_rows, axis=0)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize so cosine similarity == dot product."""
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32)
