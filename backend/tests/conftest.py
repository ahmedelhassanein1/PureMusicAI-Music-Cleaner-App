"""Shared pytest fixtures for API tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.settings import settings


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    """Point job storage at a temp folder so tests never touch /app/jobs."""
    path = tmp_path / "jobs"
    path.mkdir()
    monkeypatch.setattr(settings, "jobs_dir", path)
    return path


@pytest.fixture
def client(jobs_dir):
    """HTTP client that exercises the FastAPI app (runs lifespan + background tasks)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Replace ML-heavy pipeline steps with fast fakes.

    Patches symbols where main.py imports them so _run_pipeline never loads
    PyTorch, UVR, or PANNs during tests.
    """

    def fake_separate(input_path, output_dir, preset, progress_callback=None):
        if progress_callback:
            progress_callback(100, "separating")
        out = output_dir / "uvr_stem.wav"
        shutil.copy2(input_path, out)
        return out

    def fake_canonicalize(uvr_path, output_dir, dest_name="instrumental_raw.wav"):
        dest = output_dir / dest_name
        shutil.copy2(uvr_path, dest)
        return dest

    def fake_detect_sfx(_work_path):
        return [], []

    def fake_finalize(plan):
        if plan.instrumental_path.resolve() != plan.output_path.resolve():
            shutil.copy2(plan.instrumental_path, plan.output_path)
        return plan.output_path

    def fake_extract_choir(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.main.separate_instrumental", fake_separate)
    monkeypatch.setattr("app.main.canonicalize_instrumental", fake_canonicalize)
    monkeypatch.setattr("app.main.detect_sfx_segments", fake_detect_sfx)
    monkeypatch.setattr("app.main.finalize_instrumental", fake_finalize)
    monkeypatch.setattr("app.main.extract_choir_candidate", fake_extract_choir)


def make_upload(
    client: TestClient,
    *,
    filename: str = "track.wav",
    content: bytes = b"RIFF----WAVEfmt ",
    content_type: str = "audio/wav",
    **form_fields: str | float,
):
    """POST /api/upload with multipart form data."""
    data = {"model_id": "balanced", **form_fields}
    files = {"file": (filename, content, content_type)}
    return client.post("/api/upload", data=data, files=files)


@pytest.fixture
def completed_job(client, mock_pipeline):
    """Upload a file and return job_id after the mocked pipeline finishes."""
    response = make_upload(client)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "completed"
    return job_id
