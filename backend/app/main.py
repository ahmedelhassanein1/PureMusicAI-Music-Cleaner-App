"""
FastAPI application — upload, job status, download.

Pipeline per job:
  1. UVR instrumental separation → instrumental_raw.wav (standard bed)
  2. Optional UVR denoise (Lite or Standard) → instrumental_denoised.wav
  3. Optional choir preservation (karaoke stem + heuristics) when enabled
  4. SFX detection (generic AudioSet) + optional custom reference matching
  5. Remix (choir overlay + SFX) → final downloadable WAV (MP3 on download)
"""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import job_store
from app.pipeline.choir import extract_choir_candidate
from app.pipeline.remix import (
    ALLOWED_MP3_BITRATES,
    DEFAULT_MP3_BITRATE_KBPS,
    RemixPlan,
    export_mp3,
    finalize_instrumental,
    mp3_cache_path,
)
from app.pipeline.model_registry import (
    DEFAULT_KARAOKE_MODEL_ID,
    DENOISE_LITE_MODEL_ID,
    DENOISE_PRESET_IDS,
    REFERENCE_INSTRUMENTAL_MODEL_ID,
    ModelPreset,
    get_preset,
    is_denoise_preset,
    is_separable_preset,
    list_all_models,
    list_cleanup_presets,
    list_karaoke_presets,
    list_presets,
)
from app.pipeline.separator import (
    canonicalize_instrumental,
    denoise_instrumental,
    separate_instrumental,
)
from app.pipeline.sfx import detect_sfx_segments
from app.pipeline.custom_sfx import embed_reference_clips, match_references_in_mix
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phase 4.4 — allowed upload types (must match frontend accept list).
ALLOWED_UPLOAD_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"})
ALLOWED_UPLOAD_MIME_TYPES = frozenset(
    {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/ogg",
        "application/ogg",
        "video/mp4",  # some browsers tag .m4a this way
    }
)
_UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MB
MAX_REFERENCE_CLIPS = 10


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run startup housekeeping (expired job purge)."""
    purged = job_store.delete_expired_jobs()
    if purged:
        logger.info("Startup: removed %d expired job(s)", purged)
    yield


app = FastAPI(title="Music Cleaner API", version="0.4.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models(full: bool = Query(default=False)) -> dict:
    """
    List separation models.

    Default: curated presets from model_registry (starter, karaoke, classic, ensemble).
    ?full=true: entire audio-separator catalog (proxies --list_models).
    """
    if full:
        return {
            "source": "audio-separator",
            "models": list_all_models(),
        }
    return {
        "source": "registry",
        "models": list_presets(),
        "karaoke_models": list_karaoke_presets(),
        "cleanup_models": list_cleanup_presets(),
        "default_karaoke_model_id": DEFAULT_KARAOKE_MODEL_ID,
    }


@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str = Form(default="balanced"),
    karaoke_model_id: str = Form(default=DEFAULT_KARAOKE_MODEL_ID),
    choir_aggressiveness: float = Form(default=0.0),
    sfx_strength: float = Form(default=1.0),
    enable_denoise: bool = Form(default=False),
    denoise_model_id: str = Form(default=""),
    reference_clips: Annotated[list[UploadFile], File()] = [],
) -> dict:
    """
    Accept an audio upload and queue separation + optional choir + SFX cleanup.

    choir_aggressiveness: 0.0–1.0 — blends extracted choir back onto the bed.
    sfx_strength: 0.0–1.0 (1.0 = full attenuation in detected SFX regions).
    enable_denoise: legacy flag; if true with empty denoise_model_id, uses Lite.
    denoise_model_id: cleanup preset id (``denoise_lite`` / ``denoise``) or empty.
    reference_clips: optional short SFX samples for custom matching (Phase 2b).
    """
    preset = get_preset(model_id)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = _validate_upload_file(file)

    choir_aggressiveness = float(max(0.0, min(1.0, choir_aggressiveness)))
    sfx_strength = float(max(0.0, min(1.0, sfx_strength)))
    enable_denoise = _as_bool(enable_denoise)
    try:
        denoise_model_id = _resolve_denoise_model_id(denoise_model_id, enable_denoise)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    enable_denoise = bool(denoise_model_id)

    karaoke_preset = get_preset(karaoke_model_id)
    if choir_aggressiveness > 0 and (
        karaoke_preset is None or not karaoke_preset.is_karaoke
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or invalid karaoke_model_id: {karaoke_model_id}",
        )

    # Drop empty placeholders some browsers send when the multi-file input is unused.
    ref_uploads = [clip for clip in reference_clips if clip.filename]
    if len(ref_uploads) > MAX_REFERENCE_CLIPS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many reference clips (max {MAX_REFERENCE_CLIPS}).",
        )
    for clip in ref_uploads:
        _validate_upload_file(clip)

    status = job_store.create_job(model_id=model_id, original_filename=file.filename)
    job_id = status["id"]
    job_store.update_job(
        job_id,
        sfx_strength=sfx_strength,
        karaoke_model_id=karaoke_model_id,
        choir_aggressiveness=choir_aggressiveness,
        enable_denoise=enable_denoise,
        denoise_model_id=denoise_model_id or None,
        custom_sfx_clip_count=len(ref_uploads),
    )

    directory = job_store.job_dir(job_id)
    input_path = directory / f"input{suffix}"
    try:
        bytes_written = await _save_upload_limited(file, input_path, settings.max_upload_bytes)
    except _UploadTooLarge as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    logger.info("Saved upload for job %s (%d bytes, %s)", job_id, bytes_written, suffix)

    reference_paths: list[Path] = []
    if ref_uploads:
        refs_dir = directory / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index, clip in enumerate(ref_uploads):
                ref_suffix = Path(clip.filename or "").suffix.lower() or ".wav"
                # Keep original stem so MatchSegment labels stay readable.
                safe_stem = (
                    "".join(
                        ch if ch.isalnum() or ch in "-_ " else "_"
                        for ch in Path(clip.filename or f"ref_{index}").stem
                    ).strip()[:80]
                    or f"ref_{index}"
                )
                ref_path = refs_dir / f"{index:02d}_{safe_stem}{ref_suffix}"
                await _save_upload_limited(clip, ref_path, settings.max_upload_bytes)
                reference_paths.append(ref_path)
        except _UploadTooLarge as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except HTTPException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    background_tasks.add_task(
        _run_pipeline,
        job_id,
        input_path,
        preset.id,
        karaoke_model_id,
        choir_aggressiveness,
        sfx_strength,
        reference_paths,
        enable_denoise,
        denoise_model_id,
    )
    return {"job_id": job_id, "status": status["status"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    status = job_store.get_job(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/api/jobs/{job_id}/download")
def download(
    job_id: str,
    output_format: str = Query(default="mp3", alias="format", pattern="^(mp3|wav)$"),
    bitrate: int = Query(default=DEFAULT_MP3_BITRATE_KBPS),
) -> FileResponse:
    """
    Download the processed instrumental.

    Default: MP3 via ffmpeg. Pass ``?format=wav`` for the lossless WAV master.
    MP3 bitrates: ``?bitrate=192`` (default) or ``?bitrate=320``.
    """
    status = job_store.get_job(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_name = status.get("output_filename")
    if not output_name:
        raise HTTPException(status_code=404, detail="Output file missing")

    wav_path = job_store.job_dir(job_id) / output_name
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    download_stem = Path(status["original_filename"]).stem

    if output_format == "wav":
        return FileResponse(
            path=wav_path,
            media_type="audio/wav",
            filename=f"instrumental_{download_stem}.wav",
        )

    if bitrate not in ALLOWED_MP3_BITRATES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bitrate {bitrate}; use 192 or 320",
        )

    try:
        mp3_path = export_mp3(
            wav_path,
            mp3_cache_path(wav_path, bitrate),
            bitrate_kbps=bitrate,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=mp3_path,
        media_type="audio/mpeg",
        filename=f"instrumental_{download_stem}.mp3",
    )


class _UploadTooLarge(Exception):
    """Raised when an upload exceeds settings.max_upload_bytes."""


def _validate_upload_file(file: UploadFile) -> str:
    """
    4.4 — Reject unsupported extensions and non-audio MIME types.

    Returns the normalized lowercase suffix (e.g. ``.mp3``).
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or '(none)'}'. Allowed: {allowed}",
        )

    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in ALLOWED_UPLOAD_MIME_TYPES:
        if not content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type '{content_type}'. Upload an audio file.",
            )

    return suffix


async def _save_upload_limited(
    file: UploadFile,
    dest: Path,
    max_bytes: int,
) -> int:
    """Stream upload to disk and enforce a maximum byte size."""
    total = 0
    with dest.open("wb") as buffer:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise _UploadTooLarge(
                    f"File too large. Maximum upload size is {max_bytes // (1024 * 1024)} MB."
                )
            buffer.write(chunk)
    return total


def _resolve_pipeline_presets(
    model_id: str,
    karaoke_model_id: str,
    choir_aggressiveness: float,
) -> tuple[ModelPreset, ModelPreset | None, float, bool]:
    """
    Map UI/API model picks to (standard bed, karaoke stem, aggressiveness, choir_on).

    When the main model is karaoke, balanced is used as the standard bed and choir
    preservation is enabled automatically unless aggressiveness is explicitly 0.
    """
    preset = get_preset(model_id)
    if preset is None:
        raise ValueError(f"Invalid model_id: {model_id}")

    karaoke_preset = get_preset(karaoke_model_id)
    aggressiveness = choir_aggressiveness

    if preset.category == "karaoke":
        karaoke_preset = preset
        reference = get_preset(REFERENCE_INSTRUMENTAL_MODEL_ID)
        if reference is None or not is_separable_preset(reference):
            raise ValueError("Reference instrumental preset is not available")
        standard_preset = reference
        if aggressiveness <= 0.0:
            aggressiveness = 1.0
    else:
        standard_preset = preset
        if not is_separable_preset(standard_preset):
            raise ValueError(f"Model {model_id} cannot run separation")

    choir_enabled = (
        aggressiveness > 0.0
        and karaoke_preset is not None
        and karaoke_preset.category == "karaoke"
        and is_separable_preset(karaoke_preset)
    )
    return standard_preset, karaoke_preset if choir_enabled else None, aggressiveness, choir_enabled


def _run_pipeline(
    job_id: str,
    input_path: Path,
    model_id: str,
    karaoke_model_id: str,
    choir_aggressiveness: float,
    sfx_strength: float,
    reference_paths: list[Path] | None = None,
    enable_denoise: bool = False,
    denoise_model_id: str = "",
) -> None:
    """
    Background worker: UVR → optional denoise → optional choir → SFX → remix.

    Download defaults to instrumental_raw.wav. instrumental_denoised.wav is used
    when denoise ran and no SFX remix file is produced. instrumental.wav is used
    when SFX regions were attenuated during remix.
    """
    reference_paths = list(reference_paths or [])
    try:
        denoise_model_id = _resolve_denoise_model_id(denoise_model_id, enable_denoise)
    except ValueError as exc:
        job_store.update_job(job_id, status="failed", stage="error", error=str(exc))
        return
    enable_denoise = bool(denoise_model_id)
    try:
        standard_preset, karaoke_preset, choir_aggressiveness, choir_enabled = (
            _resolve_pipeline_presets(model_id, karaoke_model_id, choir_aggressiveness)
        )
    except ValueError as exc:
        job_store.update_job(job_id, status="failed", stage="error", error=str(exc))
        return

    directory = job_store.job_dir(job_id)

    try:
        # --- Step 1: standard instrumental separation ---
        job_store.set_stage(job_id, "separating", progress=5, status="processing")

        def on_standard_separate(percent: int, _stage: str) -> None:
            if enable_denoise:
                cap = 28
            elif choir_enabled:
                cap = 35
            else:
                cap = 65
            mapped = 5 + int((percent / 100) * (cap - 5))
            job_store.update_job(job_id, progress=mapped)

        uvr_output = separate_instrumental(
            input_path=input_path,
            output_dir=directory,
            preset=standard_preset,
            progress_callback=on_standard_separate,
        )
        work_path = canonicalize_instrumental(uvr_output, directory)
        choir_candidate_path: Path | None = None

        # --- Step 1b: optional denoise on the instrumental bed ---
        if enable_denoise:
            job_store.set_stage(job_id, "denoising", progress=30)

            def on_denoise(percent: int, _stage: str) -> None:
                mapped = 30 + int((percent / 100) * 6)
                job_store.update_job(job_id, progress=mapped)

            work_path = denoise_instrumental(
                input_path=work_path,
                output_dir=directory,
                progress_callback=on_denoise,
                model_id=denoise_model_id,
            )
            logger.info(
                "Job %s: denoise (%s) applied → %s",
                job_id,
                denoise_model_id,
                work_path.name,
            )

        # --- Step 2: optional choir candidate extraction ---
        if choir_enabled and karaoke_preset is not None:
            job_store.set_stage(job_id, "separating_karaoke", progress=38)

            def on_karaoke_separate(percent: int, _stage: str) -> None:
                mapped = 38 + int((percent / 100) * 22)
                job_store.update_job(job_id, progress=mapped)

            karaoke_uvr = separate_instrumental(
                input_path=input_path,
                output_dir=directory,
                preset=karaoke_preset,
                progress_callback=on_karaoke_separate,
            )
            karaoke_path = canonicalize_instrumental(
                karaoke_uvr,
                directory,
                dest_name="karaoke_instrumental_raw.wav",
            )

            job_store.set_stage(job_id, "preserving_choir", progress=62)
            choir_candidate_path = directory / "choir_candidate.wav"
            extract_choir_candidate(
                karaoke_path,
                work_path,
                choir_candidate_path,
            )
            logger.info(
                "Job %s: choir candidate extracted (karaoke=%s)",
                job_id,
                karaoke_preset.id,
            )

        # --- Step 3: Detect SFX (PANNs / AudioSet denylist) ---
        job_store.set_stage(job_id, "detecting_sfx", progress=75)
        sfx_segments, fired_labels = detect_sfx_segments(work_path)
        job_store.update_job(
            job_id,
            sfx_segment_count=len(sfx_segments),
            sfx_classes_detected=fired_labels,
        )
        if fired_labels:
            logger.info("Job %s SFX classes detected: %s", job_id, fired_labels)

        # --- Step 3b: optional custom reference matching (Phase 2b) ---
        custom_matches = []
        if reference_paths:
            job_store.set_stage(job_id, "matching_custom_sfx", progress=82)
            refs = embed_reference_clips(reference_paths)
            custom_matches = match_references_in_mix(work_path, refs)
            job_store.update_job(
                job_id,
                custom_sfx_segment_count=len(custom_matches),
                custom_sfx_labels=[m.label for m in custom_matches],
            )
            logger.info(
                "Job %s: custom SFX %d match(es) from %d reference(s)",
                job_id,
                len(custom_matches),
                len(refs),
            )

        # --- Step 4: Remix stems → final WAV ---
        apply_sfx = (
            (bool(sfx_segments) or bool(custom_matches)) and sfx_strength > 0
        )
        if apply_sfx:
            download_name = "instrumental.wav"
        elif enable_denoise:
            # Keep instrumental_raw.wav as the undenoised UVR bed for A/B.
            download_name = "instrumental_denoised.wav"
        else:
            download_name = "instrumental_raw.wav"
        final_path = directory / download_name

        job_store.set_stage(job_id, "remixing", progress=88)
        finalize_instrumental(
            RemixPlan(
                instrumental_path=work_path,
                output_path=final_path,
                choir_path=choir_candidate_path,
                choir_gain=choir_aggressiveness if choir_enabled else 0.0,
                sfx_segments=tuple(sfx_segments),
                sfx_strength=sfx_strength if apply_sfx else 0.0,
                custom_sfx_segments=tuple(custom_matches),
            )
        )

        if not apply_sfx:
            logger.info(
                "Job %s: no SFX remix — download will use %s",
                job_id,
                download_name,
            )

        job_store.set_stage(
            job_id,
            "done",
            progress=100,
            status="completed",
            output_filename=download_name,
        )
    except Exception as exc:  # noqa: BLE001 — surface error to client via status.json
        logger.exception("Job %s failed", job_id)
        job_store.update_job(job_id, status="failed", stage="error", error=str(exc))


def _as_bool(value: object) -> bool:
    """Normalize multipart form booleans (bool or common string forms)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_denoise_model_id(denoise_model_id: object, enable_denoise: bool) -> str:
    """
    Normalize denoise selection to a cleanup preset id or "".

    Empty / off → "". Legacy ``enable_denoise=true`` with no model → Lite.
    """
    model_id = str(denoise_model_id or "").strip()
    if model_id.lower() in {"", "none", "off", "false", "0"}:
        model_id = ""
    if not model_id and enable_denoise:
        model_id = DENOISE_LITE_MODEL_ID
    if model_id and not is_denoise_preset(model_id):
        raise ValueError(
            f"Unknown denoise_model_id: {model_id}. "
            f"Use one of: {', '.join(sorted(DENOISE_PRESET_IDS))}."
        )
    return model_id