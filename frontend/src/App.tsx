import { useEffect, useState } from "react";
import {
  downloadUrl,
  fetchJob,
  fetchModels,
  ModelPreset,
  uploadAudio,
} from "./api/client";

const STAGE_LABELS: Record<string, string> = {
  queued: "Waiting in queue",
  loading_model: "Loading model weights",
  separating: "Separating vocals from music",
  detecting_sfx: "Scanning for sound effects",
  removing_sfx: "Reducing detected sound effects",
  finalizing: "Finalizing output",
  done: "Processing complete",
  error: "Error",
};

function formatStage(stage: string): string {
  return STAGE_LABELS[stage] ?? stage.replace(/_/g, " ");
}

function progressBarLabel(stage: string, status: string): string {
  if (status === "uploading") return "Uploading audio…";
  if (status === "completed") return STAGE_LABELS.done;
  if (status === "failed") return STAGE_LABELS.error;
  if (stage) return formatStage(stage);
  if (status === "queued") return STAGE_LABELS.queued;
  return "Processing…";
}

export default function App() {
  const [models, setModels] = useState<ModelPreset[]>([]);
  const [modelId, setModelId] = useState("balanced");
  const [sfxStrength, setSfxStrength] = useState(100);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [status, setStatus] = useState<string>("idle");
  const [error, setError] = useState<string | null>(null);

  const isBusy =
    status === "uploading" ||
    status === "queued" ||
    status === "processing";

  useEffect(() => {
    fetchModels()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setModelId(list[0].id);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!jobId || status === "completed" || status === "failed") return;

    let cancelled = false;

    async function poll() {
      try {
        const job = await fetchJob(jobId!);
        if (cancelled) return;

        setProgress(job.progress);
        setStage(job.stage);
        setStatus(job.status);
        setError(job.error);

        if (job.status === "completed") {
          setError(null);
        }
      } catch {
        if (cancelled) return;
      }
    }

    poll();
    const interval = setInterval(poll, 2000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [jobId, status]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || isBusy) return;

    setError(null);
    setJobId(null);
    setProgress(0);
    setStage("");
    setStatus("uploading");

    try {
      const result = await uploadAudio(file, modelId, sfxStrength / 100);
      setJobId(result.job_id);
      setStatus("queued");
    } catch (err) {
      setError(String(err));
      setStatus("idle");
    }
  }

  const selectedModel = models.find((m) => m.id === modelId);

  return (
    <div className="app-shell">
      <header className="uvr-header">
        <span className="uvr-badge">Phase 1 · Local</span>
        <h1>Music Cleaner</h1>
        <p>
          Free, open-source vocal separation in your browser — powered by
          Ultimate Vocal Remover models
        </p>
      </header>

      <form onSubmit={handleSubmit} className="main-panel">
        <section className="panel-section">
          <h2 className="section-title">Input audio</h2>
          <label
            className={`file-drop${file ? " has-file" : ""}`}
            htmlFor="audio-upload"
          >
            <input
              id="audio-upload"
              type="file"
              accept="audio/*,.mp3,.wav,.flac,.m4a"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              disabled={isBusy}
            />
            <div className="file-drop-icon" aria-hidden="true">
              ♪
            </div>
            <p className="file-drop-title">
              {file ? "Audio file ready" : "Select audio file"}
            </p>
            <p className="file-drop-hint">
              MP3 · WAV · FLAC · M4A — runs locally on your machine
            </p>
            {file && <span className="file-name">{file.name}</span>}
          </label>
        </section>

        <section className="panel-section">
          <h2 className="section-title">Separation model</h2>
          <div className="model-grid" role="radiogroup" aria-label="Model">
            {models.map((m) => (
              <label
                key={m.id}
                className={`model-option${modelId === m.id ? " selected" : ""}`}
              >
                <input
                  type="radio"
                  name="model"
                  value={m.id}
                  checked={modelId === m.id}
                  onChange={() => setModelId(m.id)}
                  disabled={isBusy}
                />
                <div className="model-option-body">
                  <strong>{m.name}</strong>
                  <span>{m.description}</span>
                  <span className="model-arch">{m.arch}</span>
                </div>
              </label>
            ))}
          </div>
        </section>

        <section className="panel-section">
          <h2 className="section-title">Sound effect removal</h2>
          <label className="slider-control" htmlFor="sfx-strength">
            <div className="slider-header">
              <span>SFX reduction strength</span>
              <span className="slider-value">{sfxStrength}%</span>
            </div>
            <input
              id="sfx-strength"
              type="range"
              min={0}
              max={100}
              step={5}
              value={sfxStrength}
              onChange={(e) => setSfxStrength(Number(e.target.value))}
              disabled={isBusy}
            />
            <p className="slider-hint">
              0% keeps all SFX · 100% fully mutes detected explosions, whooshes,
              and similar effects
            </p>
          </label>
        </section>

        <div className="action-row">
          <span className="action-hint">
            {selectedModel
              ? `${selectedModel.name} selected`
              : "Loading models…"}
          </span>
          <button
            type="submit"
            className="btn-primary"
            disabled={!file || isBusy || models.length === 0}
          >
            {status === "uploading"
              ? "Uploading…"
              : isBusy
                ? "Processing…"
                : "Start processing"}
          </button>
        </div>
      </form>

      {jobId && status !== "idle" && (
        <section className="status-panel" aria-live="polite">
          <div className="status-header">
            <h2>Process status</h2>
            <span className={`status-badge ${status}`}>{status}</span>
          </div>

          <div className="status-meta">
            <div className="meta-item">
              <span className="meta-label">Job ID</span>
              <span className="meta-value">{jobId.slice(0, 8)}…</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Stage</span>
              <span className="meta-value">
                {stage ? formatStage(stage) : "—"}
              </span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Model</span>
              <span className="meta-value">
                {selectedModel?.name ?? modelId}
              </span>
            </div>
          </div>

          <div className="progress-wrap">
            <div className="progress-labels">
              <span className="progress-stage">
                {progressBarLabel(stage, status)}
              </span>
              <span>{progress}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {status === "completed" && (
            <a className="btn-download" href={downloadUrl(jobId)} download>
              ↓ Save instrumental
            </a>
          )}

          {error && <div className="error-box">{error}</div>}
        </section>
      )}

      {error && !jobId && <div className="error-box">{error}</div>}

      <p className="footer-note">
        Inspired by{" "}
        <a
          href="https://ultimatevocalremover.com/"
          target="_blank"
          rel="noreferrer"
        >
          Ultimate Vocal Remover
        </a>{" "}
        · Runs locally via WSL + Docker
      </p>
    </div>
  );
}
