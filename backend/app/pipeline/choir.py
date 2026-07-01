"""
Phase 3 — choir extraction via lead-vocal subtraction and heuristics.

Karaoke UVR models retain backing/choir in the instrumental stem. This module:

  1. Extracts a choir candidate (karaoke instrumental − standard instrumental).
  2. Weights regions using stereo-width + polyphony heuristics.
  3. Remixes the weighted choir back onto the standard instrumental bed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.pipeline.remix import remix_with_choir

logger = logging.getLogger(__name__)

# Frames ≈ 10 ms at 44.1 kHz with default librosa hop (512).
_HEURISTIC_HOP = 512


def preserve_choir(
    standard_instrumental_path: Path,
    karaoke_instrumental_path: Path,
    output_path: Path,
    *,
    aggressiveness: float = 1.0,
) -> Path:
    """
    Full choir-preservation pass: extract candidate → heuristic mask → remix.

    aggressiveness: 0.0–1.0 scales how much choir energy is re-added.
    """
    aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
    if aggressiveness <= 0.0:
        import shutil

        shutil.copy2(standard_instrumental_path, output_path)
        return output_path

    directory = output_path.parent
    choir_candidate = directory / "choir_candidate.wav"
    extract_choir_candidate(
        karaoke_instrumental_path,
        standard_instrumental_path,
        choir_candidate,
        gain=1.0,
    )
    remix_with_choir(
        standard_instrumental_path,
        choir_candidate,
        output_path,
        choir_gain=aggressiveness,
    )
    return output_path


def subtract_lead_vocal(
    audio_path: Path,
    lead_vocal_path: Path,
    output_path: Path,
    *,
    strength: float = 1.0,
) -> Path:
    """Subtract a lead vocal stem from audio (instrumental bed or full mix)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        audio, sample_rate = _load_audio(audio_path)
        sf.write(output_path, audio, sample_rate)
        return output_path

    base, sample_rate = _load_audio(audio_path)
    lead, lead_sr = _load_audio(lead_vocal_path)
    if lead_sr != sample_rate:
        raise ValueError(
            f"Lead vocal sample rate {lead_sr} != audio {sample_rate}"
        )

    length = min(len(base), len(lead))
    if length == 0:
        raise ValueError("Empty audio in lead vocal subtraction")

    processed = base[:length] - (strength * lead[:length])
    processed = np.clip(processed, -1.0, 1.0)
    sf.write(output_path, processed, sample_rate)
    logger.info(
        "Subtracted lead vocal (strength=%.2f) → %s",
        strength,
        output_path.name,
    )
    return output_path


def extract_lead_vocal(
    full_vocals_path: Path,
    backing_vocals_path: Path,
    output_path: Path,
    *,
    strength: float = 1.0,
) -> Path:
    """Estimate the lead vocal stem as (full vocals − backing vocals)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    full, sample_rate = _load_audio(full_vocals_path)
    backing, backing_sr = _load_audio(backing_vocals_path)
    if backing_sr != sample_rate:
        raise ValueError(
            f"Backing vocal sample rate {backing_sr} != full vocals {sample_rate}"
        )

    length = min(len(full), len(backing))
    lead = (full[:length] - strength * backing[:length]).astype(np.float32)
    lead = np.clip(lead, -1.0, 1.0)
    sf.write(output_path, lead, sample_rate)
    logger.info("Extracted lead vocal stem → %s", output_path.name)
    return output_path


def extract_choir_candidate(
    karaoke_instrumental_path: Path,
    standard_instrumental_path: Path,
    output_path: Path,
    *,
    gain: float = 1.0,
) -> Path:
    """
    Estimate choir/backing content from stem difference, weighted by heuristics.

    Stereo-width and polyphony scores are computed on the karaoke instrumental
    to favour wide, harmonically dense regions (typical of backing/choir).
    """
    gain = float(np.clip(gain, 0.0, 2.0))
    karaoke, sample_rate = _load_audio(karaoke_instrumental_path)
    standard, standard_sr = _load_audio(standard_instrumental_path)
    if standard_sr != sample_rate:
        raise ValueError(
            f"Standard instrumental sr {standard_sr} != karaoke {sample_rate}"
        )

    length = min(len(karaoke), len(standard))
    raw = (karaoke[:length] - standard[:length]).astype(np.float32)

    mask = choir_weight_mask(
        karaoke[:length],
        sample_rate,
        num_samples=length,
    )
    choir = raw * mask
    choir = np.clip(choir * gain, -1.0, 1.0).astype(np.float32)
    sf.write(output_path, choir, sample_rate)
    logger.info("Extracted choir candidate → %s", output_path.name)
    return output_path


def choir_weight_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    num_samples: int,
) -> np.ndarray:
    """
    Per-sample weight in [0, 1] from stereo-width + polyphony heuristics.

  - Wide stereo (high side/mid energy) → likely backing/choir.
  - High spectral flatness / peak count → likely polyphonic choir vs single lead.
    """
    width = _stereo_width_envelope(audio, sample_rate, num_samples)
    polyphony = _polyphony_envelope(audio, sample_rate, num_samples)
    combined = np.clip(0.55 * width + 0.45 * polyphony, 0.0, 1.0)
    # Keep a small floor so quiet backing isn't fully discarded.
    return (0.15 + 0.85 * combined).astype(np.float32)[:, np.newaxis]


def _stereo_width_envelope(
    audio: np.ndarray,
    sample_rate: int,
    num_samples: int,
) -> np.ndarray:
    """Map side/mid energy ratio to [0, 1] per sample."""
    if audio.shape[1] < 2:
        return np.ones(num_samples, dtype=np.float32) * 0.5

    left = audio[:, 0]
    right = audio[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    frame_mid = librosa.feature.rms(y=mid, hop_length=_HEURISTIC_HOP)[0]
    frame_side = librosa.feature.rms(y=side, hop_length=_HEURISTIC_HOP)[0]
    ratio = frame_side / (frame_mid + 1e-8)
    # Typical lead vocal is centred (low ratio); choir sections spread wider.
    frame_score = np.clip((ratio - 0.08) / 0.55, 0.0, 1.0)
    return _frames_to_samples(frame_score, num_samples, _HEURISTIC_HOP)


def _polyphony_envelope(
    audio: np.ndarray,
    sample_rate: int,
    num_samples: int,
) -> np.ndarray:
    """
    Higher spectral flatness + active peak count → more polyphonic (choir-like).
    """
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    stft = np.abs(librosa.stft(mono, hop_length=_HEURISTIC_HOP))
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    # Normalise flatness into 0–1 (typical music 0.01–0.35).
    flat_score = np.clip((flatness - 0.02) / 0.28, 0.0, 1.0)

    peak_counts = np.sum(stft > (0.12 * np.max(stft, axis=0, keepdims=True)), axis=0)
    peak_score = np.clip(peak_counts / 40.0, 0.0, 1.0)

    frame_score = np.clip(0.6 * flat_score + 0.4 * peak_score, 0.0, 1.0)
    return _frames_to_samples(frame_score, num_samples, _HEURISTIC_HOP)


def _frames_to_samples(
    frame_values: np.ndarray,
    num_samples: int,
    hop_length: int,
) -> np.ndarray:
    """Repeat each frame value across hop_length samples and trim/pad."""
    samples = np.repeat(frame_values, hop_length)
    if len(samples) < num_samples:
        samples = np.pad(samples, (0, num_samples - len(samples)))
    return samples[:num_samples].astype(np.float32)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True)
    return audio.astype(np.float32), int(sample_rate)
