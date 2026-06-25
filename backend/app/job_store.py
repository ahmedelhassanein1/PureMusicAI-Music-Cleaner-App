"""
Filesystem-backed job store — one folder per job with a status.json file.

Phase 2 adds explicit pipeline stages (detecting_speech, removing_sfx, etc.)
so the frontend can show where a long-running job is in the pipeline.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import settings

# Valid `stage` values written to status.json during processing.
PIPELINE_STAGES = (
    "queued",
    "separating",
    "detecting_sfx",
    "removing_sfx",
    "done",
    "error",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(model_id: str, original_filename: str) -> dict[str, Any]:
    """
    Create a new job folder and return its initial status dict.

    Each job gets a UUID directory under jobs/ with status.json tracking
    progress through the pipeline (UVR → SFX).
    """
    job_id = str(uuid.uuid4())
    job_dir = settings.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    status: dict[str, Any] = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "model_id": model_id,
        "original_filename": original_filename,
        "output_filename": None,
        "error": None,
        # SFX metadata — filled in as the pipeline runs
        "sfx_segment_count": None,
        "sfx_classes_detected": [],
        "sfx_strength": 1.0,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_status(job_dir, status)
    return status


def get_job(job_id: str) -> dict[str, Any] | None:
    """Read and return a job's status.json, or None if the job does not exist."""
    path = settings.jobs_dir / job_id / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    """
    Merge new fields into status.json and bump updated_at.

    Example: update_job(id, progress=50, speech_segment_count=3)
    """
    job_dir = settings.jobs_dir / job_id
    status = get_job(job_id)
    if status is None:
        raise FileNotFoundError(f"Job {job_id} not found")

    status.update(fields)
    status["updated_at"] = _utc_now()
    _write_status(job_dir, status)
    return status


def set_stage(
    job_id: str,
    stage: str,
    *,
    progress: int | None = None,
    status: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Update the current pipeline stage shown to the frontend.

    Phase 2 stages: detecting_sfx → removing_sfx.
    Optional progress (0–100) and extra fields (e.g. speech_segment_count) can
    be passed in the same write so the UI gets one consistent snapshot.
    """
    fields: dict[str, Any] = {"stage": stage}
    if progress is not None:
        fields["progress"] = progress
    if status is not None:
        fields["status"] = status
    fields.update(extra)
    return update_job(job_id, **fields)


def job_dir(job_id: str) -> Path:
    """Return the filesystem path for a job's working directory."""
    return settings.jobs_dir / job_id


def _write_status(job_dir: Path, status: dict[str, Any]) -> None:
    path = job_dir / "status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
