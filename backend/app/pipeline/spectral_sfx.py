"""
Phase 2b — ref-guided soft spectral mask (Approach B).

STFT(mix window) + ref frequency template → Wiener-like mask → iSTFT,
then cosine edge crossfade back into the untouched bed (2b-S.3).
Cuts SFX-like frequency bins while leaving other bins closer to full level.
Pure DSP (numpy/librosa); called from remix.finalize_instrumental (2b-S.4).
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

N_FFT = 2048
HOP_LENGTH = 512
# Long refs: keep loudest ~2 s so silence/tails don't dilute the template.
MAX_TEMPLATE_SEC = 2.0
MIN_TEMPLATE_SEC = 0.25
EPS = 1e-8
# Blend edited ↔ original at segment edges so hard STFT splices don't click.
EDGE_CROSSFADE_MS = 20.0


def _stft(audio_mono: np.ndarray) -> np.ndarray:
    """Complex STFT → (freq_bins, time_frames)."""
    return librosa.stft(
        np.asarray(audio_mono, dtype=np.float32),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=True,
    )


def _istft(stft_matrix: np.ndarray, *, length: int | None = None) -> np.ndarray:
    """Inverse STFT → mono float32 waveform."""
    audio = librosa.istft(
        stft_matrix,
        hop_length=HOP_LENGTH,
        center=True,
        length=length,
    )
    return np.asarray(audio, dtype=np.float32)


def _crop_peak_energy(audio_mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Keep ≤ MAX_TEMPLATE_SEC around the loudest RMS peak.

    Short clips are unchanged; long ones (e.g. 17 s) drop quiet tails.
    """
    audio_mono = np.asarray(audio_mono, dtype=np.float32).reshape(-1)
    max_samples = int(MAX_TEMPLATE_SEC * sample_rate)
    if audio_mono.size <= max_samples:
        return audio_mono

    win = max(1, int(MIN_TEMPLATE_SEC * sample_rate))
    # Sliding RMS via squared moving average → find densest SFX region.
    sq = audio_mono.astype(np.float64) ** 2
    kernel = np.ones(win, dtype=np.float64) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + EPS)
    peak = int(np.argmax(rms))
    half = max_samples // 2
    start = max(0, peak - half)
    end = min(audio_mono.size, start + max_samples)
    start = max(0, end - max_samples)
    return audio_mono[start:end]


def ref_template_magnitude(
    ref_audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Time-averaged |STFT| of the ref → (freq_bins,) fingerprint.

    Describes which frequencies the SFX occupies, not its full timeline.
    """
    mono = np.asarray(ref_audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=-1)
    mono = _crop_peak_energy(mono, sample_rate)
    if mono.size == 0:
        raise ValueError("Reference audio is empty")

    mag = np.abs(_stft(mono))
    # Average over time so one vector describes "what this SFX sounds like".
    return np.mean(mag, axis=1).astype(np.float32)


def soft_mask(
    mix_mag: np.ndarray,
    ref_mag: np.ndarray,
    *,
    floor: float = 0.0,
) -> np.ndarray:
    """
    Wiener-like mask in [floor, 1]; high where ref explains mix energy.

    mix_mag: (freq, time); ref_mag: (freq,) or (freq, time).
    """
    mix_mag = np.asarray(mix_mag, dtype=np.float32)
    ref_mag = np.asarray(ref_mag, dtype=np.float32)

    if ref_mag.ndim == 1:
        ref_b = ref_mag[:, None]
    else:
        ref_b = ref_mag

    if ref_b.shape[0] != mix_mag.shape[0]:
        raise ValueError(
            f"Frequency bin mismatch: ref {ref_b.shape[0]} vs mix {mix_mag.shape[0]}"
        )

    # Broadcast / tile ref across mix time frames.
    if ref_b.shape[1] == 1:
        ref_b = np.broadcast_to(ref_b, mix_mag.shape)
    elif ref_b.shape[1] != mix_mag.shape[1]:
        reps = int(np.ceil(mix_mag.shape[1] / ref_b.shape[1]))
        ref_b = np.tile(ref_b, (1, reps))[:, : mix_mag.shape[1]]

    mask = ref_b / (ref_b + mix_mag + EPS)
    floor = float(np.clip(floor, 0.0, 1.0))
    return np.clip(mask, floor, 1.0).astype(np.float32)


def _crossfade_curve(length: int) -> np.ndarray:
    """Equal-power fade 0 → 1 over ``length`` samples (same curve as remix 4.1)."""
    if length <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.sin(t * np.pi / 2.0) ** 2


def _blend_edited_edges(
    original: np.ndarray,
    edited: np.ndarray,
    sample_rate: int,
    fade_ms: float = EDGE_CROSSFADE_MS,
) -> np.ndarray:
    """
    Crossfade edited samples into the original bed at both edges (2b-S.3).

    original / edited: (N,) or (N, C). Weight w: 0 = keep bed, 1 = full edit.
    Short windows shrink the fade so in/out ramps never overlap past the midpoint.
    """
    original = np.asarray(original, dtype=np.float32)
    edited = np.asarray(edited, dtype=np.float32)
    if original.shape != edited.shape:
        raise ValueError(
            f"Shape mismatch for edge blend: {original.shape} vs {edited.shape}"
        )

    squeezed = False
    if original.ndim == 1:
        original = original[:, None]
        edited = edited[:, None]
        squeezed = True
    elif original.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio, got shape {original.shape}")

    n = original.shape[0]
    fade = min(max(1, int(sample_rate * fade_ms / 1000.0)), max(1, n // 2))
    weight = np.ones(n, dtype=np.float32)

    fade_in = _crossfade_curve(fade)
    weight[:fade] = fade_in
    weight[n - fade :] = fade_in[::-1]

    # Broadcast weight over channels: out = (1-w)*original + w*edited
    w = weight[:, None]
    blended = (1.0 - w) * original + w * edited
    blended = blended.astype(np.float32)
    return blended.squeeze(axis=1) if squeezed else blended


def apply_spectral_mask_to_segment(
    mix: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    ref_audio: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Spectrally attenuate [start, end) using ref as template.

    mix: mono (N,) or stereo (N, C). strength 0 = no-op, 1 = full mask.
    Same mono-derived mask applied to every channel; edges crossfaded into bed.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    out = np.array(mix, dtype=np.float32, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    elif out.ndim != 2:
        raise ValueError(f"Expected 1D or 2D mix, got shape {out.shape}")

    n_samples, n_ch = out.shape
    start_i = max(0, int(start * sample_rate))
    end_i = min(n_samples, int(end * sample_rate))
    if end_i <= start_i or strength <= 0.0:
        return out.squeeze(axis=1) if mix.ndim == 1 else out

    segment = out[start_i:end_i, :]
    seg_len = segment.shape[0]
    # One mask from mono downmix so L/R stay phase-aligned.
    mono_seg = np.mean(segment, axis=1)
    if mono_seg.size < N_FFT // 4:
        logger.warning(
            "Segment too short for stable STFT (%d samples) — leaving unchanged",
            mono_seg.size,
        )
        return out.squeeze(axis=1) if mix.ndim == 1 else out

    mix_mag = np.abs(_stft(mono_seg))
    mask = soft_mask(mix_mag, ref_template_magnitude(ref_audio, sample_rate))
    gain = (1.0 - strength * mask).astype(np.float32)

    # Keep mix phase; shrink magnitudes where mask is high.
    edited = np.empty_like(segment)
    for ch in range(n_ch):
        ch_stft = _stft(segment[:, ch])
        ch_wave = _istft(ch_stft * gain, length=seg_len)
        if ch_wave.shape[0] < seg_len:
            pad = np.zeros(seg_len - ch_wave.shape[0], dtype=np.float32)
            ch_wave = np.concatenate([ch_wave, pad])
        edited[:, ch] = ch_wave[:seg_len]

    # Soft splice: full edit in the middle, original bed at the edges.
    out[start_i:end_i, :] = _blend_edited_edges(segment, edited, sample_rate)

    return out.squeeze(axis=1) if mix.ndim == 1 else out
