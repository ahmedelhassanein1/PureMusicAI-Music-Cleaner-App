FROM python:3.11-slim

# ffmpeg + build tools: audio-separator pulls in diffq, which compiles a C extension.
# wget: download PANNs label CSV + model weights for panns-inference (Phase 2 SFX).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    wget \
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
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

# panns-inference expects these under /root/panns_data (not bundled in pip package).
RUN mkdir -p /root/panns_data \
    && wget -q -O /root/panns_data/class_labels_indices.csv \
        http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv \
    && wget -q -O /root/panns_data/Cnn14_mAP=0.431.pth \
        "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

COPY app ./app
COPY scripts ./scripts

ENV JOBS_DIR=/app/jobs
ENV MODELS_DIR=/app/models

RUN mkdir -p /app/jobs /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]