# PureMusic AI - Music Cleaner

Clean up audio in your browser — remove lead vocals, spoken dialogue, and sound effects, while trying to keep choir and instrumental backing.

Runs **locally** on your PC via WSL + Docker. Free, unlimited, no account required. Your files stay on your machine.

---

## What it does

| Removed | Kept |
|---------|------|
| Lead vocals | Instrumental backing |
| Spoken dialogue | Choir / group vocals *(when choir preservation is enabled)* |
| Sound effects (explosions, whooshes, etc.) | Musical content |

**How it works in one sentence:** upload a song → AI separates and cleans stems → download an instrumental (MP3 or WAV).

**Choir caveat:** Perfect choir preservation is hard. Standard vocal-removal models may still reduce choir. Use the **choir aggressiveness** slider and a **karaoke sub-model** to blend backing vocals back in — results vary by track.

---

## How to use the app

1. **Start the app** (see [Running the app](#running-the-app) below) and open **http://localhost:5173**.
2. **Select an audio file** — MP3, WAV, FLAC, or M4A (max **100 MB**).
3. **Pick a separation model:**
   - **Fast** — quickest, good for CPU-only machines
   - **Balanced** — default quality/speed tradeoff
   - **High Quality** — Roformer; slower, cleaner output (GPU recommended)
   - **Classic** / **Ensemble** presets — more options for advanced users
4. **Optional — Choir preservation:** set **Choir aggressiveness** above 0% and pick a **Karaoke sub-model**.
5. **Optional — SFX reduction:** adjust the **SFX reduction strength** slider (100% = strongest attenuation in detected SFX regions).
6. Click **Start processing** and watch the progress panel.
7. When status is **completed**, download **MP3** (192k or 320k) or **WAV**.

**Tips**

- First run downloads model weights (hundreds of MB) — expect a longer wait.
- **Ensemble** models run multiple full separation passes; a 4-minute song can take 30–60+ minutes.
- Refreshing the page during a job should resume polling (job ID is stored in the browser for the session).
- Jobs on disk are auto-deleted after **24 hours**.

---

## Running the app

### Prerequisites

- **WSL 2** with Ubuntu (or native Linux/macOS with Docker)
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** with WSL integration enabled
- **8 GB+ RAM** (16 GB recommended)
- **Optional:** NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for faster separation

### Start

```bash
cd ~/projects/music-cleaner
docker compose up --build
```

Open **http://localhost:5173** in your browser.

Verify the backend: `curl http://localhost:8000/api/health` → `{"status":"ok"}`

### Stop

Press `Ctrl+C` in the terminal, or run `docker compose down`.

### No NVIDIA GPU?

Edit `docker-compose.yml`: comment out `gpus: all` and set `DEVICE=cpu` under the backend `environment` block.

### Optional environment overrides

Copy `.env.example` to `.env` and adjust paths or `DEVICE` if running components outside Docker.

---

## Hardware requirements

### System

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 5 GB free | 10 GB+ free (model cache in `backend/models/`) |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | None (CPU works) | NVIDIA 6 GB+ VRAM (e.g. RTX 3050 Ti) |

### Model choice vs your machine

| Preset | GPU helpful? | Rough time (4-min song) |
|--------|--------------|-------------------------|
| **Fast** | Optional | 3–8 min (CPU) · 20–60 s (GPU) |
| **Balanced** | Optional | 2–6 min (CPU) · 30–90 s (GPU) |
| **High Quality** (Roformer) | Strongly recommended | 20–40 min (CPU) · 5–15 min (GPU) |
| **Ensemble** | Recommended | 30–90+ min (GPU) |

Times are approximate — track length and system load matter.

### What runs on GPU vs CPU

| Workload | Typical device |
|----------|----------------|
| VR / Roformer separation | GPU when `DEVICE=cuda` |
| MDX-Net (Balanced) | Often CPU (ONNX) |
| SFX detection (PANNs) | CPU |
| MP3 export (ffmpeg) | CPU |
| Choir heuristics + remix | CPU |

---

## How it works (pipeline)

```mermaid
flowchart LR
  upload[Upload] --> separate[UVR_separation]
  separate --> choir[Choir_optional]
  separate --> sfx[SFX_detection]
  choir --> remix[Remix_and_normalize]
  sfx --> remix
  remix --> download[Download_MP3_or_WAV]
```

Each job is a folder under `backend/jobs/` with a `status.json` file tracking progress. No database — everything is on disk.

---

## Project structure

```
music-cleaner/
├── docker-compose.yml       # Start frontend + backend
├── .env.example             # Optional env overrides
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── requirements-dev.txt # pytest dependencies
│   ├── pytest.ini
│   ├── tests/               # API tests (ML mocked)
│   ├── jobs/                # Uploads + job status (gitignored)
│   ├── models/              # Downloaded UVR weights (gitignored)
│   └── app/
│       ├── main.py          # FastAPI routes + pipeline worker
│       ├── settings.py
│       ├── job_store.py
│       └── pipeline/
│           ├── model_registry.py
│           ├── separator.py
│           ├── choir.py
│           ├── sfx.py
│           └── remix.py
└── frontend/
    ├── Dockerfile
    ├── vite.config.ts
    └── src/
        ├── App.tsx          # Main UI
        ├── App.test.tsx
        ├── api/client.ts    # API helpers
        ├── api/client.test.ts
        └── test/            # Vitest setup + fixtures
```

---

## Tech stack

| Layer | Tools | Purpose |
|-------|-------|---------|
| **UI** | React 18, TypeScript, Vite | Upload form, progress, download links |
| **API** | FastAPI, Uvicorn, Pydantic | REST endpoints, file upload, background jobs |
| **ML** | PyTorch, audio-separator, UVR models | Vocal/instrumental separation |
| **Audio** | librosa, soundfile, NumPy, ffmpeg | Analysis, WAV I/O, MP3 export |
| **Detection** | PANNs (SFX), OmniVAD (speech) | Find non-musical regions to attenuate |
| **Storage** | Local filesystem + `status.json` | Job state (no Redis/Postgres) |
| **Packaging** | Docker, Docker Compose | Reproducible dev environment on WSL |
| **Tests** | pytest, Vitest, Testing Library | Automated API + UI tests |

**Not used:** accounts, cloud storage, Redis, Celery, or a SQL database.

---

## Running tests

**Backend (pytest)** — use Docker:

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend \
  sh -c "pip install -q -r requirements-dev.txt && pytest -v"
```

**Frontend (Vitest)** — on the host:

```bash
cd frontend
npm install
npm test
```

---

## Manual smoke test checklist

Use a **30–60 second** clip first.

- [ ] `docker compose up --build` succeeds
- [ ] UI loads at `http://localhost:5173`
- [ ] `/api/health` returns OK
- [ ] Upload completes with **Balanced**
- [ ] Progress updates through stages to **completed**
- [ ] MP3 and WAV downloads play correctly
- [ ] Bad file type rejected; file over 100 MB rejected
- [ ] `pytest` and `npm test` pass

---

## Troubleshooting

### Docker and startup

| Problem | Fix |
|---------|-----|
| `docker compose` not found | Start Docker Desktop; enable WSL integration |
| Frontend works, upload fails | Wait for backend; check `http://localhost:8000/api/health` |
| `gpus: all` error | Remove GPU block in `docker-compose.yml`; set `DEVICE=cpu` |

### During processing

| Problem | Fix |
|---------|-----|
| Very slow first job | Model download; weights cache in `backend/models/` |
| Progress stuck low on **Ensemble** | Normal — multiple passes; can take 30–60+ min |
| **Job not found** | Don't delete `backend/jobs/` while running; jobs expire after 24h |
| UI lost job after refresh | Re-upload if needed; check `docker compose logs backend` |

### Downloads and uploads

| Problem | Fix |
|---------|-----|
| Download **400** | Wait until status is **completed** |
| Download **404** | Job expired or folder cleared — re-upload |
| Upload **413** | File over 100 MB — trim or compress |
| Upload **400** | Use `.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, or `.ogg` |
| CORS error | Open `http://localhost:5173`, not `:8000` directly |

### Performance

| Problem | Fix |
|---------|-----|
| Out of memory | Use **Fast**; shorter clip; close other apps |
| Slow on laptop | Use **Fast** or **Balanced**; enable GPU if available |

### Debug logs

```bash
docker compose logs -f backend
cat backend/jobs/<job-id>/status.json
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/models` | Curated presets + karaoke models |
| `GET` | `/api/models?full=true` | Full audio-separator catalog |
| `POST` | `/api/upload` | Upload audio → `{ job_id }` |
| `GET` | `/api/jobs/{id}` | Job status and progress |
| `GET` | `/api/jobs/{id}/download` | Download result (default MP3; `?format=wav`; `?bitrate=192\|320`) |

---

## Credits

Vocal separation uses [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) models via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT license).
