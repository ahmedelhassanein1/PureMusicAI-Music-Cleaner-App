import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  downloadUrl,
  fetchJob,
  fetchModels,
  uploadAudio,
} from "./client";

describe("downloadUrl", () => {
  it("defaults to MP3 at 192 kbps", () => {
    expect(downloadUrl("abc-123")).toBe(
      "/api/jobs/abc-123/download?format=mp3&bitrate=192",
    );
  });

  it("builds a WAV download URL", () => {
    expect(downloadUrl("abc-123", { format: "wav" })).toBe(
      "/api/jobs/abc-123/download?format=wav",
    );
  });

  it("includes a custom MP3 bitrate", () => {
    expect(downloadUrl("abc-123", { format: "mp3", bitrate: 320 })).toBe(
      "/api/jobs/abc-123/download?format=mp3&bitrate=320",
    );
  });
});

describe("fetchModels", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns parsed JSON on success", async () => {
    const payload = { source: "registry", models: [] };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => payload,
      }),
    );

    await expect(fetchModels()).resolves.toEqual(payload);
    expect(fetch).toHaveBeenCalledWith("/api/models");
  });

  it("throws when the response is not ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500 }),
    );

    await expect(fetchModels()).rejects.toThrow("Failed to load models");
  });
});

describe("uploadAudio", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("POSTs multipart form data with normalized strengths", async () => {
    const file = new File(["audio"], "track.mp3", { type: "audio/mpeg" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "job-456" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await uploadAudio(file, "balanced", {
      sfxStrength: 0.5,
      choirAggressiveness: 1.5,
      karaokeModelId: "karaoke_mdx_kara2",
    });

    expect(result).toEqual({ job_id: "job-456" });
    expect(fetchMock).toHaveBeenCalledWith("/api/upload", {
      method: "POST",
      body: expect.any(FormData),
    });

    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("file")).toBe(file);
    expect(body.get("model_id")).toBe("balanced");
    expect(body.get("karaoke_model_id")).toBe("karaoke_mdx_kara2");
    expect(body.get("sfx_strength")).toBe("0.5");
    expect(body.get("choir_aggressiveness")).toBe("1");
    expect(body.get("enable_denoise")).toBe("false");
    expect(body.get("denoise_model_id")).toBe("");
  });

  it("sends denoise_model_id for standard DeNoise", async () => {
    const file = new File(["audio"], "track.mp3", { type: "audio/mpeg" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "job-denoise" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await uploadAudio(file, "balanced", { denoiseModelId: "denoise" });

    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("enable_denoise")).toBe("true");
    expect(body.get("denoise_model_id")).toBe("denoise");
  });

  it("maps legacy enableDenoise=true to denoise_lite", async () => {
    const file = new File(["audio"], "track.mp3", { type: "audio/mpeg" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "job-denoise-lite" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await uploadAudio(file, "balanced", { enableDenoise: true });

    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get("enable_denoise")).toBe("true");
    expect(body.get("denoise_model_id")).toBe("denoise_lite");
  });

  it("throws with server detail on failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        text: async () => "Unknown model_id",
      }),
    );

    const file = new File(["audio"], "track.mp3", { type: "audio/mpeg" });
    await expect(uploadAudio(file, "bad-model")).rejects.toThrow(
      "Unknown model_id",
    );
  });
});

describe("fetchJob", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns job JSON on success", async () => {
    const job = { id: "job-1", status: "completed", progress: 100 };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => job,
      }),
    );

    await expect(fetchJob("job-1")).resolves.toEqual(job);
    expect(fetch).toHaveBeenCalledWith("/api/jobs/job-1");
  });

  it("throws a clear error for 404", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404 }),
    );

    await expect(fetchJob("missing")).rejects.toThrow("Job not found");
  });

  it("throws on other HTTP errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500 }),
    );

    await expect(fetchJob("job-1")).rejects.toThrow(
      "Failed to fetch job status",
    );
  });
});
