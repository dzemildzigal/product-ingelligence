# syntax=docker/dockerfile:1

# Safe, minimal image for FastAPI + OCR + FAISS + pyzbar
FROM python:3.13.7 AS runtime

ENV PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	TOKENIZERS_PARALLELISM=false

# System dependencies:
# - tesseract-ocr: required by pytesseract
# - libzbar0: required by pyzbar (barcode reading)
# - curl: used for HEALTHCHECK
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   tesseract-ocr libzbar0 curl \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
	&& pip install -r requirements.txt

# Copy application code and data
COPY app ./app
COPY tools ./tools
COPY data ./data
COPY README.md ./

# Optionally prebuild the FAISS index at image build time to speed startup
# Set at build time: --build-arg PREBUILD_INDEX=true
ARG PREBUILD_INDEX=false
RUN if [ "$PREBUILD_INDEX" = "true" ]; then \
	  python tools/build_faiss_index.py || echo "[docker] Index build failed; will attempt at runtime"; \
	fi

# Create non-root user for safety
RUN useradd -m -u 10001 appuser \
	&& chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Basic healthcheck against FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/healthz || exit 1

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

