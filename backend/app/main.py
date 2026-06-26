"""
FastAPI application — upload, job status, download.

Pipeline per job:
  1. UVR instrumental separation → instrumental_raw.wav (download default)
  2. SFX detection + attenuation → instrumental.wav (only when SFX are found)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import job_store
from app.pipeline.model_registry import get_preset, list_presets
from app.pipeline.separator import canonicalize_instrumental, separate_instrumental
from app.pipeline.sfx import attenuate_sfx, detect_sfx_segments
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Cleaner API", version="0.2.3")

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
def models() -> dict:
    return {"models": list_presets()}


@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: str = Form(default="balanced"),
    sfx_strength: float = Form(default=1.0),
) -> dict:
    """
    Accept an audio upload and queue separation + SFX cleanup.

    sfx_strength: 0.0–1.0 (1.0 = full attenuation in detected SFX regions).
    """
    preset = get_preset(model_id)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    sfx_strength = float(max(0.0, min(1.0, sfx_strength)))

    status = job_store.create_job(model_id=model_id, original_filename=file.filename)
    job_id = status["id"]
    job_store.update_job(job_id, sfx_strength=sfx_strength)

    directory = job_store.job_dir(job_id)
    suffix = Path(file.filename).suffix or ".wav"
    input_path = directory / f"input{suffix}"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        _run_pipeline,
        job_id,
        input_path,
        preset.id,
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
def download(job_id: str) -> FileResponse:
    status = job_store.get_job(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_name = status.get("output_filename")
    if not output_name:
        raise HTTPException(status_code=404, detail="Output file missing")

    output_path = job_store.job_dir(job_id) / output_name
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"instrumental_{status['original_filename']}",
    )


def _run_pipeline(
    job_id: str,
    input_path: Path,
    model_id: str,
    sfx_strength: float,
) -> None:
    """
    Background worker: UVR separation → optional SFX cleanup.

    Download defaults to instrumental_raw.wav (pure UVR stem). instrumental.wav
    is only served when SFX regions were actually detected and attenuated.
    """
    preset = get_preset(model_id)
    if preset is None:
        job_store.update_job(job_id, status="failed", stage="error", error="Invalid model")
        return

    directory = job_store.job_dir(job_id)

    try:
        # --- Step 1: UVR instrumental separation ---
        job_store.set_stage(job_id, "separating", progress=5, status="processing")

        def on_separate(percent: int, _stage: str) -> None:
            mapped = 5 + int((percent / 100) * 65)
            job_store.update_job(job_id, progress=mapped)

        uvr_output = separate_instrumental(
            input_path=input_path,
            output_dir=directory,
            preset=preset,
            progress_callback=on_separate,
        )
        work_path = canonicalize_instrumental(uvr_output, directory)
        download_name = "instrumental_raw.wav"

        # --- Step 2: Detect SFX (PANNs / AudioSet denylist) ---
        job_store.set_stage(job_id, "detecting_sfx", progress=75)
        sfx_segments, fired_labels = detect_sfx_segments(work_path)
        job_store.update_job(
            job_id,
            sfx_segment_count=len(sfx_segments),
            sfx_classes_detected=fired_labels,
        )
        if fired_labels:
            logger.info("Job %s SFX classes detected: %s", job_id, fired_labels)

        # --- Step 3: Attenuate SFX only when something was detected ---
        if sfx_segments and sfx_strength > 0:
            job_store.set_stage(job_id, "removing_sfx", progress=88)
            final_path = directory / "instrumental.wav"
            attenuate_sfx(
                work_path,
                final_path,
                strength=sfx_strength,
                segments=sfx_segments,
            )
            download_name = "instrumental.wav"
        else:
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
