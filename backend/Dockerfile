FROM python:3.11-slim

# ffmpeg + build tools: audio-separator pulls in diffq, which compiles a C extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# PyTorch stack from one index so torch / torchvision / torchaudio versions match.
# Mismatch causes: RuntimeError: operator torchvision::nms does not exist
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV JOBS_DIR=/app/jobs
ENV MODELS_DIR=/app/models

RUN mkdir -p /app/jobs /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]