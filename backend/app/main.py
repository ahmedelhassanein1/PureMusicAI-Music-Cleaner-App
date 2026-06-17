"""FastAPI application — upload, job status, download."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import job_store
from app.pipeline.model_registry import get_preset, list_presets
from app.pipeline.separator import separate_instrumental
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Cleaner API", version="0.1.0")

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
) -> dict:
    preset = get_preset(model_id)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    status = job_store.create_job(model_id=model_id, original_filename=file.filename)
    job_id = status["id"]
    directory = job_store.job_dir(job_id)

    suffix = Path(file.filename).suffix or ".wav"
    input_path = directory / f"input{suffix}"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(_run_separation, job_id, input_path, preset.id)
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


def _run_separation(job_id: str, input_path: Path, model_id: str) -> None:
    """Background worker: runs ML separation and updates job status."""
    preset = get_preset(model_id)
    if preset is None:
        job_store.update_job(job_id, status="failed", error="Invalid model")
        return

    try:
        job_store.update_job(job_id, status="processing", stage="separating", progress=5)
        directory = job_store.job_dir(job_id)

        def on_progress(percent: int, stage: str) -> None:
            job_store.update_job(job_id, progress=percent, stage=stage)

        instrumental = separate_instrumental(
            input_path=input_path,
            output_dir=directory,
            preset=preset,
            progress_callback=on_progress,
        )

        final_path = directory / "instrumental.wav"
        if instrumental != final_path:
            shutil.move(str(instrumental), str(final_path))

        job_store.update_job(
            job_id,
            status="completed",
            stage="done",
            progress=100,
            output_filename="instrumental.wav",
        )
    except Exception as exc:  # noqa: BLE001 — surface error to client via status.json
        logger.exception("Job %s failed", job_id)
        job_store.update_job(job_id, status="failed", stage="error", error=str(exc))
