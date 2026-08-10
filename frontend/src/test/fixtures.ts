import type { JobStatus, ModelPreset, ModelsResponse } from "../api/client";

export const mockMainModel: ModelPreset = {
  id: "balanced",
  name: "Balanced",
  description: "MDX-Net instrumental model.",
  arch: "mdx",
  category: "starter",
  is_karaoke: false,
};

export const mockFastModel: ModelPreset = {
  id: "fast",
  name: "Fast",
  description: "Quick VR Arch model.",
  arch: "vr",
  category: "starter",
  is_karaoke: false,
};

export const mockKaraokeModel: ModelPreset = {
  id: "karaoke_mdx_kara2",
  name: "Karaoke MDX (KARA 2)",
  description: "MDX karaoke model.",
  arch: "mdx",
  category: "karaoke",
  is_karaoke: true,
};

export const mockModelsResponse: ModelsResponse = {
  source: "registry",
  models: [mockMainModel, mockFastModel, mockKaraokeModel],
  karaoke_models: [mockKaraokeModel],
  default_karaoke_model_id: "karaoke_mdx_kara2",
};

export function makeJob(overrides: Partial<JobStatus> = {}): JobStatus {
  return {
    id: "job-123",
    status: "processing",
    stage: "separating",
    progress: 42,
    model_id: "balanced",
    original_filename: "track.mp3",
    output_filename: null,
    error: null,
    ...overrides,
  };
}
