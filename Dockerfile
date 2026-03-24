FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    libsndfile1 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install "setuptools<81" wheel

# Install CPU-only torch stack explicitly to avoid CUDA-linked torchaudio.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchaudio==2.2.2

# Install app dependencies. Stage 4 adds madmom from GitHub for Python 3.11 compatibility.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install whisper separately without build isolation so it uses setuptools<81.
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

COPY app /app/app

RUN mkdir -p /app/data/uploads /app/data/outputs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
