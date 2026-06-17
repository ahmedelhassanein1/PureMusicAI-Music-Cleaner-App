export type ModelPreset = {
  id: string;
  name: string;
  description: string;
  arch: string;
};

export type JobStatus = {
  id: string;
  status: "queued" | "processing" | "completed" | "failed";
  stage: string;
  progress: number;
  model_id: string;
  original_filename: string;
  output_filename: string | null;
  error: string | null;
};

const API_BASE = "/api";

export async function fetchModels(): Promise<ModelPreset[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  const data = await res.json();
  return data.models;
}

export async function uploadAudio(
  file: File,
  modelId: string,
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Upload failed");
  }
  return res.json();
}

export async function fetchJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error("Failed to fetch job status");
  return res.json();
}

export function downloadUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/download`;
}
