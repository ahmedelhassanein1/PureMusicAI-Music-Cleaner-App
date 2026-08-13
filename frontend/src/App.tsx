import { useEffect, useState } from "react";
import {
  downloadUrl,
  fetchJob,
  fetchModels,
  ModelPreset,
  Mp3Bitrate,
  uploadAudio,
} from "./api/client";

const STAGE_LABELS: Record<string, string> = {
  queued: "Waiting in queue",
  loading_model: "Loading model weights",
  separating: "Separating vocals from music",
  separating_karaoke: "Running karaoke separation for choir",
  preserving_choir: "Preserving backing vocals and choir",
  detecting_sfx: "Scanning for sound effects",
  remixing: "Mixing stems into final output",
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

function isMainGridModel(model: ModelPreset): boolean {
  return model.category !== "karaoke";
}

const ACTIVE_JOB_KEY = "music-cleaner-active-job";

export default function App() {
  const [models, setModels] = useState<ModelPreset[]>([]);
  const [karaokeModels, setKaraokeModels] = useState<ModelPreset[]>([]);
  const [modelId, setModelId] = useState("balanced");
  const [karaokeModelId, setKaraokeModelId] = useState("karaoke_mdx_kara2");
  const [choirAggressiveness, setChoirAggressiveness] = useState(0);
  const [sfxStrength, setSfxStrength] = useState(100);
  const [mp3Bitrate, setMp3Bitrate] = useState<Mp3Bitrate>(192);
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobModelId, setJobModelId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [status, setStatus] = useState<string>("idle");
  const [error, setError] = useState<string | null>(null);

  const mainModels = models.filter(isMainGridModel);

  const isBusy =
    status === "uploading" ||
    status === "queued" ||
    status === "processing";

  // Restore in-progress job after a page reload (Vite HMR, docker restart, etc.).
  useEffect(() => {
    const saved = sessionStorage.getItem(ACTIVE_JOB_KEY);
    if (saved) {
      setJobId(saved);
      setStatus("processing");
    }
  }, []);

  useEffect(() => {
    if (jobId) {
      sessionStorage.setItem(ACTIVE_JOB_KEY, jobId);
    } else {
      sessionStorage.removeItem(ACTIVE_JOB_KEY);
    }
  }, [jobId]);

  useEffect(() => {
    fetchModels()
      .then((data) => {
        const list = data.models;
        const karaoke = data.karaoke_models ?? list.filter((m) => m.is_karaoke);
        setModels(list);
        setKaraokeModels(karaoke);

        const mains = list.filter(isMainGridModel);
        if (mains.length > 0) {
          setModelId((current) =>
            mains.some((m) => m.id === current)
              ? current
              : (mains.find((m) => m.id === "balanced") ?? mains[0]).id,
          );
        }

        const defaultKaraoke =
          data.default_karaoke_model_id ??
          karaoke.find((m) => m.id === "karaoke_mdx_kara2")?.id ??
          karaoke[0]?.id;
        if (defaultKaraoke) {
          setKaraokeModelId((current) =>
            karaoke.some((m) => m.id === current) ? current : defaultKaraoke,
          );
        }
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!jobId || status === "completed" || status === "failed") return;

    let cancelled = false;
    let pollFailures = 0;

    async function poll() {
      try {
        const job = await fetchJob(jobId!);
        if (cancelled) return;

        pollFailures = 0;
        setProgress(job.progress);
        setStage(job.stage);
        setStatus(job.status);
        setJobModelId(job.model_id);
        setError(job.error);

        if (job.status === "completed") {
          setError(null);
        }
        if (job.status === "failed") {
          sessionStorage.removeItem(ACTIVE_JOB_KEY);
        }
      } catch (err) {
        if (cancelled) return;
        pollFailures += 1;
        const message = String(err);
        if (message.includes("Job not found")) {
          setStatus("failed");
          setError(
            "Job not found — the jobs folder may have been cleared while this was running.",
          );
          sessionStorage.removeItem(ACTIVE_JOB_KEY);
          return;
        }
        if (pollFailures >= 5) {
          setError(
            "Lost contact with the backend. The job may still be running — check docker compose logs. Do not refresh.",
          );
        }
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
    setJobModelId(null);
    setProgress(0);
    setStage("");
    setStatus("uploading");

    try {
      const result = await uploadAudio(file, modelId, {
        sfxStrength: sfxStrength / 100,
        karaokeModelId,
        choirAggressiveness: choirAggressiveness / 100,
      });
      setJobId(result.job_id);
      setStatus("queued");
    } catch (err) {
      setError(String(err));
      setStatus("idle");
    }
  }

  const selectedModel = mainModels.find((m) => m.id === modelId);
  const selectedKaraoke = karaokeModels.find((m) => m.id === karaokeModelId);
  const activeJobModel =
    models.find((m) => m.id === jobModelId) ??
    mainModels.find((m) => m.id === jobModelId);
  const isEnsembleJob =
    jobModelId?.startsWith("ensemble_") ||
    activeJobModel?.category === "ensemble";

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <img
            src="/logo.png"
            alt=""
            className="brand-logo"
            width={56}
            height={56}
          />
          <h1 className="brand-title">PureMusic AI - Music Cleaner</h1>
        </div>
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
            {mainModels.map((m) => (
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
          <h2 className="section-title">Choir preservation</h2>
          <label className="field-control" htmlFor="karaoke-model">
            <span className="field-label">Karaoke sub-model</span>
            <select
              id="karaoke-model"
              value={karaokeModelId}
              onChange={(e) => setKaraokeModelId(e.target.value)}
              disabled={isBusy || karaokeModels.length === 0}
            >
              {karaokeModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <p className="slider-hint">
              {selectedKaraoke?.description ??
                "Karaoke UVR model used to extract backing vocals and choir."}
            </p>
          </label>

          <label className="slider-control" htmlFor="choir-aggressiveness">
            <div className="slider-header">
              <span>Choir aggressiveness</span>
              <span className="slider-value">{choirAggressiveness}%</span>
            </div>
            <input
              id="choir-aggressiveness"
              type="range"
              min={0}
              max={100}
              step={5}
              value={choirAggressiveness}
              onChange={(e) =>
                setChoirAggressiveness(Number(e.target.value))
              }
              disabled={isBusy}
            />
            <p className="slider-hint">
              0% skips choir preservation · 100% fully re-adds detected backing
              and choir from the karaoke stem
            </p>
          </label>
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

        <section className="panel-section">
          <h2 className="section-title">Download format</h2>
          <fieldset className="bitrate-options" disabled={isBusy}>
            <legend className="field-label">MP3 bitrate (default download)</legend>
            <label className="bitrate-option">
              <input
                type="radio"
                name="mp3-bitrate"
                value={192}
                checked={mp3Bitrate === 192}
                onChange={() => setMp3Bitrate(192)}
              />
              192 kbps — smaller files
            </label>
            <label className="bitrate-option">
              <input
                type="radio"
                name="mp3-bitrate"
                value={320}
                checked={mp3Bitrate === 320}
                onChange={() => setMp3Bitrate(320)}
              />
              320 kbps — higher quality
            </label>
          </fieldset>
          <p className="slider-hint">
            Processed audio is saved as WAV internally; your download is MP3 at
            the selected bitrate. Use “Download WAV” after processing for the
            lossless file.
          </p>
        </section>

        <div className="action-row">
          <span className="action-hint">
            {selectedModel
              ? choirAggressiveness > 0
                ? `${selectedModel.name} + ${selectedKaraoke?.name ?? "karaoke"} choir`
                : `${selectedModel.name} selected`
              : "Loading models…"}
          </span>
          <button
            type="submit"
            className="btn-primary"
            disabled={!file || isBusy || mainModels.length === 0}
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
                {activeJobModel?.name ?? jobModelId ?? selectedModel?.name ?? modelId}
              </span>
            </div>
            {choirAggressiveness > 0 && (
              <div className="meta-item">
                <span className="meta-label">Choir</span>
                <span className="meta-value">
                  {selectedKaraoke?.name ?? karaokeModelId} ·{" "}
                  {choirAggressiveness}%
                </span>
              </div>
            )}
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
            {isEnsembleJob && status === "processing" && progress < 70 && (
              <p className="slider-hint">
                Ensemble models run two full separation passes — progress stays
                low for a long time. This can take 30–60+ minutes on long tracks.
              </p>
            )}
          </div>

          {status === "completed" && (
            <div className="download-actions">
              <a
                className="btn-download"
                href={downloadUrl(jobId, { format: "mp3", bitrate: mp3Bitrate })}
                download
              >
                ↓ Save instrumental (MP3 {mp3Bitrate}k)
              </a>
              <a
                className="btn-download btn-download-secondary"
                href={downloadUrl(jobId, { format: "wav" })}
                download
              >
                Download WAV
              </a>
            </div>
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
