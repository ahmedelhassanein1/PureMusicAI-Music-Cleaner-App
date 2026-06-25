"""
FastAPI application — upload, job status, download.

Phase 2 pipeline per job:
  1. UVR stem separation (vocals removed)
  2. Speech detection + attenuation (OmniVAD)
  3. SFX detection + attenuation (PANNs / AudioSet denylist)
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
from app.pipeline.separator import separate_instrumental
from app.pipeline.speech import attenuate_speech, detect_speech_segments
from app.pipeline.sfx import attenuate_sfx, detect_sfx_segments
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Cleaner API", version="0.2.0")

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
    speech_strength: float = Form(default=1.0),
    sfx_strength: float = Form(default=1.0),
) -> dict:
    """
    Accept an audio upload and queue the full Phase 2 pipeline.

    speech_strength / sfx_strength: 0.0–1.0 (1.0 = full removal).
    UI sliders (tasks 2.9–2.10) will send these; defaults are 100% for now.
    """
    preset = get_preset(model_id)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    speech_strength = float(max(0.0, min(1.0, speech_strength)))
    sfx_strength = float(max(0.0, min(1.0, sfx_strength)))

    status = job_store.create_job(model_id=model_id, original_filename=file.filename)
    job_id = status["id"]
    job_store.update_job(
        job_id,
        speech_strength=speech_strength,
        sfx_strength=sfx_strength,
    )

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
        speech_strength,
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
    speech_strength: float,
    sfx_strength: float,
) -> None:
    """
    Background worker: UVR separation → speech cleanup → SFX cleanup.

    Writes intermediate WAVs in the job folder, then final instrumental.wav.
    Updates status.json at each sub-stage so the UI can show progress.
    """
    preset = get_preset(model_id)
    if preset is None:
        job_store.update_job(job_id, status="failed", stage="error", error="Invalid model")
        return

    directory = job_store.job_dir(job_id)

    try:
        # --- Step 1: UVR vocal separation ---
        job_store.set_stage(job_id, "separating", progress=5, status="processing")

        def on_separate(percent: int, _stage: str) -> None:
            # Map separator progress (10–90) into 5–55 overall
            mapped = 5 + int((percent / 100) * 50)
            job_store.update_job(job_id, progress=mapped)

        instrumental = separate_instrumental(
            input_path=input_path,
            output_dir=directory,
            preset=preset,
            progress_callback=on_separate,
        )
        work_path = directory / "instrumental_raw.wav"
        if instrumental != work_path:
            shutil.move(str(instrumental), str(work_path))

        # --- Step 2: Detect speech (OmniVAD) ---
        job_store.set_stage(job_id, "detecting_speech", progress=58)
        speech_segments = detect_speech_segments(work_path)
        job_store.update_job(job_id, speech_segment_count=len(speech_segments))

        # --- Step 3: Attenuate speech regions ---
        job_store.set_stage(job_id, "removing_speech", progress=68)
        after_speech = directory / "after_speech.wav"
        attenuate_speech(
            work_path,
            after_speech,
            strength=speech_strength,
            segments=speech_segments,
        )

        # --- Step 4: Detect SFX (PANNs / AudioSet denylist) ---
        job_store.set_stage(job_id, "detecting_sfx", progress=78)
        sfx_segments, fired_labels = detect_sfx_segments(after_speech)
        job_store.update_job(
            job_id,
            sfx_segment_count=len(sfx_segments),
            sfx_classes_detected=fired_labels,
        )
        if fired_labels:
            logger.info("Job %s SFX classes detected: %s", job_id, fired_labels)

        # --- Step 5: Attenuate SFX regions ---
        job_store.set_stage(job_id, "removing_sfx", progress=88)
        final_path = directory / "instrumental.wav"
        attenuate_sfx(
            after_speech,
            final_path,
            strength=sfx_strength,
            segments=sfx_segments,
        )

        job_store.set_stage(
            job_id,
            "done",
            progress=100,
            status="completed",
            output_filename="instrumental.wav",
        )
    except Exception as exc:  # noqa: BLE001 — surface error to client via status.json
        logger.exception("Job %s failed", job_id)
        job_store.update_job(job_id, status="failed", stage="error", error=str(exc))
