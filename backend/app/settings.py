"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Where uploaded files and job status live on disk.
    jobs_dir: Path = Path("/app/jobs")
    # Where UVR model weights are cached.
    models_dir: Path = Path("/app/models")
    # Allowed browser origin for local dev (Vite default port).
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # PyTorch device: "cuda" if GPU available, else "cpu".
    device: str = "cpu"


settings = Settings()

# Create directories at import time so the app can start cleanly.
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
