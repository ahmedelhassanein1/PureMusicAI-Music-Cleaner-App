"""Filesystem-backed job store — one folder per job with a status.json file."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(model_id: str, original_filename: str) -> dict[str, Any]:
    """Create a new job folder and return its initial status dict."""
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
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_status(job_dir, status)
    return status


def get_job(job_id: str) -> dict[str, Any] | None:
    path = settings.jobs_dir / job_id / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    job_dir = settings.jobs_dir / job_id
    status = get_job(job_id)
    if status is None:
        raise FileNotFoundError(f"Job {job_id} not found")

    status.update(fields)
    status["updated_at"] = _utc_now()
    _write_status(job_dir, status)
    return status


def job_dir(job_id: str) -> Path:
    return settings.jobs_dir / job_id


def _write_status(job_dir: Path, status: dict[str, Any]) -> None:
    path = job_dir / "status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
