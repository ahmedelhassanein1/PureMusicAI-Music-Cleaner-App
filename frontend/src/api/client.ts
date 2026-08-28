export type ModelPreset = {
  id: string;
  name: string;
  description: string;
  arch: string;
  category: string;
  is_karaoke: boolean;
};

export type ModelsResponse = {
  source: string;
  models: ModelPreset[];
  karaoke_models?: ModelPreset[];
  cleanup_models?: ModelPreset[];
  default_karaoke_model_id?: string;
};

export type DenoiseModelId = "" | "denoise_lite" | "denoise";

export type JobStatus = {
  id: string;
  status: "queued" | "processing" | "completed" | "failed";
  stage: string;
  progress: number;
  model_id: string;
  karaoke_model_id?: string;
  choir_aggressiveness?: number;
  enable_denoise?: boolean;
  denoise_model_id?: string | null;
  original_filename: string;
  output_filename: string | null;
  error: string | null;
};

const API_BASE = "/api";

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function uploadAudio(
  file: File,
  modelId: string,
  options: {
    sfxStrength?: number;
    karaokeModelId?: string;
    choirAggressiveness?: number;
    enableDenoise?: boolean;
    denoiseModelId?: DenoiseModelId;
    referenceClips?: File[];
  } = {},
): Promise<{ job_id: string }> {
  const {
    sfxStrength = 1.0,
    karaokeModelId = "karaoke_mdx_kara2",
    choirAggressiveness = 0,
    enableDenoise = false,
    denoiseModelId = "",
    referenceClips = [],
  } = options;

  const resolvedDenoiseId: DenoiseModelId =
    denoiseModelId || (enableDenoise ? "denoise_lite" : "");

  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);
  form.append("karaoke_model_id", karaokeModelId);
  form.append(
    "choir_aggressiveness",
    String(Math.max(0, Math.min(1, choirAggressiveness))),
  );
  form.append("sfx_strength", String(Math.max(0, Math.min(1, sfxStrength))));
  form.append("enable_denoise", resolvedDenoiseId ? "true" : "false");
  form.append("denoise_model_id", resolvedDenoiseId);
  for (const clip of referenceClips) {
    form.append("reference_clips", clip);
  }

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
  if (res.status === 404) {
    throw new Error("Job not found");
  }
  if (!res.ok) throw new Error("Failed to fetch job status");
  return res.json();
}

export type DownloadFormat = "mp3" | "wav";
export type Mp3Bitrate = 192 | 320;

export function downloadUrl(
  jobId: string,
  options: { format?: DownloadFormat; bitrate?: Mp3Bitrate } = {},
): string {
  const { format = "mp3", bitrate = 192 } = options;
  const params = new URLSearchParams();
  params.set("format", format);
  if (format === "mp3") {
    params.set("bitrate", String(bitrate));
  }
  return `${API_BASE}/jobs/${jobId}/download?${params.toString()}`;
}
