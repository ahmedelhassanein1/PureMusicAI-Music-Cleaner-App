"""
FastAPI application — upload, job status, download.

Pipeline per job:
  1. UVR instrumental separation → instrumental_raw.wav (standard bed)
  2. Optional choir preservation (karaoke stem + heuristics) when enabled
  3. SFX detection
  4. Remix (choir overlay + SFX) → final downloadable WAV (MP3 on download)
"""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

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
    REFERENCE_INSTRUMENTAL_MODEL_ID,
    ModelPreset,
    get_preset,
    is_separable_preset,
    list_all_models,
    list_karaoke_presets,
    list_presets,
)
from app.pipeline.separator import canonicalize_instrumental, separate_instrumental
from app.pipeline.sfx import detect_sfx_segments
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
) -> dict:
    """
    Accept an audio upload and queue separation + optional choir + SFX cleanup.

    choir_aggressiveness: 0.0–1.0 — blends extracted choir back onto the bed.
    sfx_strength: 0.0–1.0 (1.0 = full attenuation in detected SFX regions).
    """
    preset = get_preset(model_id)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = _validate_upload_file(file)

    choir_aggressiveness = float(max(0.0, min(1.0, choir_aggressiveness)))
    sfx_strength = float(max(0.0, min(1.0, sfx_strength)))

    karaoke_preset = get_preset(karaoke_model_id)
    if choir_aggressiveness > 0 and (
        karaoke_preset is None or not karaoke_preset.is_karaoke
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown or invalid karaoke_model_id: {karaoke_model_id}",
        )

    status = job_store.create_job(model_id=model_id, original_filename=file.filename)
    job_id = status["id"]
    job_store.update_job(
        job_id,
        sfx_strength=sfx_strength,
        karaoke_model_id=karaoke_model_id,
        choir_aggressiveness=choir_aggressiveness,
    )

    directory = job_store.job_dir(job_id)
    input_path = directory / f"input{suffix}"
    try:
        bytes_written = await _save_upload_limited(file, input_path, settings.max_upload_bytes)
    except _UploadTooLarge as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    logger.info("Saved upload for job %s (%d bytes, %s)", job_id, bytes_written, suffix)

    background_tasks.add_task(
        _run_pipeline,
        job_id,
        input_path,
        preset.id,
        karaoke_model_id,
        choir_aggressiveness,
        sfx_strength,
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
) -> None:
    """
    Background worker: UVR separation → optional choir → SFX scan → remix finalize.

    Download defaults to instrumental_raw.wav. instrumental.wav is used when SFX
    regions were detected and attenuated during the remix step.
    """
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
            cap = 35 if choir_enabled else 65
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

        # --- Step 4: Remix stems → final WAV ---
        apply_sfx = bool(sfx_segments) and sfx_strength > 0
        download_name = "instrumental.wav" if apply_sfx else "instrumental_raw.wav"
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
            )
        )

        if not apply_sfx:
            logger.info("Job %s: no SFX to remove — download will use UVR stem", job_id)

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
