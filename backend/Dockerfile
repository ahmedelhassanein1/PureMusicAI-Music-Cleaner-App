FROM python:3.11-slim

# ffmpeg is required by audio-separator for non-WAV inputs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV JOBS_DIR=/app/jobs
ENV MODELS_DIR=/app/models

RUN mkdir -p /app/jobs /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
