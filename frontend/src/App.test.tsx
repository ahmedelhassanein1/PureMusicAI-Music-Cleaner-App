import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import * as api from "./api/client";
import { makeJob, mockModelsResponse } from "./test/fixtures";

vi.mock("./api/client", async () => {
  const actual = await vi.importActual<typeof api>("./api/client");
  return {
    ...actual,
    fetchModels: vi.fn(),
    fetchJob: vi.fn(),
    uploadAudio: vi.fn(),
    downloadUrl: vi.fn(),
  };
});

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.fetchModels).mockResolvedValue(mockModelsResponse);
    vi.mocked(api.downloadUrl).mockImplementation(
      (jobId, options = {}) => actualDownloadUrl(jobId, options),
    );
  });

  function actualDownloadUrl(
    jobId: string,
    options: { format?: "mp3" | "wav"; bitrate?: 192 | 320 } = {},
  ) {
    const { format = "mp3", bitrate = 192 } = options;
    const params = new URLSearchParams({ format });
    if (format === "mp3") params.set("bitrate", String(bitrate));
    return `/api/jobs/${jobId}/download?${params.toString()}`;
  }

  it("renders the main heading and loads models", async () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { name: /puremusic ai.*music cleaner/i }),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole("radio", { name: /balanced/i })).toBeInTheDocument();
      expect(screen.getByRole("radio", { name: /fast/i })).toBeInTheDocument();
    });
    expect(api.fetchModels).toHaveBeenCalledTimes(1);
  });

  it("keeps Start processing disabled until a file is selected", async () => {
    render(<App />);
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /balanced/i })).toBeInTheDocument(),
    );

    const submit = screen.getByRole("button", { name: "Start processing" });
    expect(submit).toBeDisabled();
  });

  it("shows a load error when models fail to fetch", async () => {
    vi.mocked(api.fetchModels).mockRejectedValue(new Error("Network down"));
    render(<App />);

    expect(await screen.findByText(/Network down/)).toBeInTheDocument();
  });

  it("uploads audio and shows progress while processing", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup();

    vi.mocked(api.uploadAudio).mockResolvedValue({ job_id: "job-123" });
    vi.mocked(api.fetchJob)
      .mockResolvedValueOnce(makeJob({ status: "processing", progress: 10 }))
      .mockResolvedValue(
        makeJob({
          status: "completed",
          stage: "done",
          progress: 100,
          output_filename: "instrumental_raw.wav",
        }),
      );

    render(<App />);
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /balanced/i })).toBeInTheDocument(),
    );

    const file = new File(["audio"], "track.mp3", { type: "audio/mpeg" });
    await user.upload(screen.getByLabelText(/select audio file/i), file);
    await user.click(screen.getByRole("button", { name: "Start processing" }));

    expect(api.uploadAudio).toHaveBeenCalledWith(
      file,
      "balanced",
      expect.objectContaining({
        karaokeModelId: "karaoke_mdx_kara2",
        choirAggressiveness: 0,
        sfxStrength: 1,
      }),
    );

    expect(await screen.findByText(/process status/i)).toBeInTheDocument();
    expect(screen.getByText(/job-123/i)).toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(2000);

    expect(
      await screen.findByRole("link", { name: /save instrumental/i }),
    ).toHaveAttribute(
      "href",
      "/api/jobs/job-123/download?format=mp3&bitrate=192",
    );
    expect(screen.getByRole("link", { name: /download wav/i })).toHaveAttribute(
      "href",
      "/api/jobs/job-123/download?format=wav",
    );

    vi.useRealTimers();
  });

  it("uses the selected MP3 bitrate in the download link", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup();

    vi.mocked(api.uploadAudio).mockResolvedValue({ job_id: "job-999" });
    vi.mocked(api.fetchJob).mockResolvedValue(
      makeJob({
        id: "job-999",
        status: "completed",
        stage: "done",
        progress: 100,
        output_filename: "instrumental_raw.wav",
      }),
    );

    render(<App />);
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /balanced/i })).toBeInTheDocument(),
    );

    await user.click(screen.getByRole("radio", { name: /320 kbps/i }));
    const file = new File(["audio"], "track.mp3", { type: "audio/mpeg" });
    await user.upload(screen.getByLabelText(/select audio file/i), file);
    await user.click(screen.getByRole("button", { name: "Start processing" }));

    await vi.advanceTimersByTimeAsync(2000);

    expect(
      await screen.findByRole("link", { name: /save instrumental/i }),
    ).toHaveAttribute(
      "href",
      "/api/jobs/job-999/download?format=mp3&bitrate=320",
    );

    vi.useRealTimers();
  });
});
