import { useEffect, useState } from "react";
import {
  downloadUrl,
  fetchJob,
  fetchModels,
  ModelPreset,
  uploadAudio,
} from "./api/client";

export default function App() {
  const [models, setModels] = useState<ModelPreset[]>([]);
  const [modelId, setModelId] = useState("balanced");
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [status, setStatus] = useState<string>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then((list) => {
        setModels(list);
        if (list.length > 0) setModelId(list[0].id);
      })
      .catch((err) => setError(String(err)));
  }, []);

  // Poll the backend every 2 seconds while a job is running.
  useEffect(() => {
    if (!jobId || status === "completed" || status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const job = await fetchJob(jobId);
        setProgress(job.progress);
        setStage(job.stage);
        setStatus(job.status);
        if (job.error) setError(job.error);
      } catch (err) {
        setError(String(err));
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, status]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;

    setError(null);
    setStatus("uploading");

    try {
      const result = await uploadAudio(file, modelId);
      setJobId(result.job_id);
      setStatus("queued");
      setProgress(0);
    } catch (err) {
      setError(String(err));
      setStatus("idle");
    }
  }

  return (
    <main className="container">
      <header>
        <h1>Music Cleaner</h1>
        <p>Upload a song and download an instrumental (vocals removed).</p>
      </header>

      <form onSubmit={handleSubmit} className="card">
        <label>
          Audio file
          <input
            type="file"
            accept="audio/*,.mp3,.wav,.flac,.m4a"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>

        <label>
          Separation model
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} — {m.description}
              </option>
            ))}
          </select>
        </label>

        <button type="submit" disabled={!file || status === "uploading"}>
          {status === "uploading" ? "Uploading…" : "Remove vocals"}
        </button>
      </form>

      {jobId && status !== "idle" && (
        <section className="card">
          <h2>Job {jobId.slice(0, 8)}…</h2>
          <p>
            Status: <strong>{status}</strong>
            {stage ? ` — ${stage}` : ""}
          </p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <p>{progress}%</p>

          {status === "completed" && (
            <a className="download-btn" href={downloadUrl(jobId)} download>
              Download instrumental
            </a>
          )}

          {error && <p className="error">{error}</p>}
        </section>
      )}
    </main>
  );
}
