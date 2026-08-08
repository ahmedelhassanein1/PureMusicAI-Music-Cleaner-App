"""API route tests — ML pipeline is mocked via conftest.mock_pipeline."""

from __future__ import annotations

from pathlib import Path

from app import job_store
from tests.conftest import make_upload


def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_registry_presets(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "registry"
    model_ids = {m["id"] for m in body["models"]}
    assert "balanced" in model_ids
    assert "fast" in model_ids
    assert body["default_karaoke_model_id"]


def test_upload_unknown_model_returns_400(client):
    response = make_upload(client, model_id="not-a-real-model")
    assert response.status_code == 400
    assert "Unknown model_id" in response.json()["detail"]


def test_upload_bad_extension_returns_400(client):
    response = make_upload(
        client,
        filename="notes.txt",
        content=b"hello",
        content_type="text/plain",
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_bad_mime_returns_400(client):
    response = make_upload(
        client,
        filename="track.wav",
        content=b"RIFF----WAVEfmt ",
        content_type="application/pdf",
    )
    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]


def test_upload_file_too_large_returns_413(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_upload_bytes", 32)
    response = make_upload(client, content=b"x" * 64)
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_upload_queues_job_and_pipeline_completes(client, mock_pipeline, jobs_dir):
    response = make_upload(client, filename="my_song.mp3", content_type="audio/mpeg")
    assert response.status_code == 200
    body = response.json()
    job_id = body["job_id"]
    assert body["status"] == "queued"

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "completed"
    assert status["stage"] == "done"
    assert status["progress"] == 100
    assert status["output_filename"] == "instrumental_raw.wav"

    output_path = jobs_dir / job_id / "instrumental_raw.wav"
    assert output_path.exists()


def test_job_status_not_found_returns_404(client):
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_download_before_complete_returns_400(client, jobs_dir):
    status = job_store.create_job(model_id="balanced", original_filename="x.wav")
    job_id = status["id"]
    job_store.update_job(job_id, status="processing", stage="separating")

    response = client.get(f"/api/jobs/{job_id}/download")
    assert response.status_code == 400
    assert response.json()["detail"] == "Job not completed yet"


def test_download_wav_after_complete(client, completed_job):
    response = client.get(f"/api/jobs/{completed_job}/download?format=wav")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert "instrumental_track.wav" in response.headers.get("content-disposition", "")


def test_download_mp3_after_complete(client, completed_job, monkeypatch):
    def fake_export_mp3(wav_path: Path, mp3_path: Path, bitrate_kbps: int = 192) -> Path:
        mp3_path.write_bytes(b"ID3-fake-mp3")
        return mp3_path

    monkeypatch.setattr("app.main.export_mp3", fake_export_mp3)

    response = client.get(f"/api/jobs/{completed_job}/download?format=mp3&bitrate=320")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content == b"ID3-fake-mp3"


def test_download_invalid_bitrate_returns_400(client, completed_job):
    response = client.get(f"/api/jobs/{completed_job}/download?bitrate=128")
    assert response.status_code == 400
    assert "bitrate" in response.json()["detail"].lower()
