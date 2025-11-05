# Product Intelligence API

A FastAPI microservice for product recognition from images and question answering using local docs (RAG with FAISS) and an optional LLM.

## Features

- OCR-based text extraction (+ optional barcode parsing) from product photos
- Catalog matching to identify likely products from `data/catalog.csv`
- Retrieval-Augmented Generation (RAG)
	- FAISS index built from `data/docs/*.txt` with semantic chunking
	- Retrieves relevant passages per product and question
- LLM answering via an OpenAI-compatible Chat Completions API
	- External (OpenAI) or local (Ollama) selected by a simple boolean flag
- Unified endpoint to recognize and answer in one call

## Quickstart

1) Install dependencies

	 - Python 3.13.7
	 - On Windows, install Visual C++ Build Tools if needed for some packages
	 - Then install:

	 ```bash
	 pip install -r requirements.txt
	 ```

2) Prepare data

	 - Product catalog: `data/catalog.csv`
	 - Documentation: `data/docs/*.txt`

3) Run the API

	 ```bash
	 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
	 ```

At startup, the app will build a FAISS index under `data/index/` if it's missing.

Alternatively, you can build the index manually:

```bash
python tools/build_faiss_index.py
```

## Docker

You can run this service in a container. The image installs system dependencies (Tesseract OCR and zbar) required for OCR and barcode reading, and it exposes port 8000 with a healthcheck on `/healthz`.

Build (without prebuilding the FAISS index):

```bash
docker build -t product-intel .
```

Build with prebuilt FAISS index (larger image, faster startup):

```bash
docker build -t product-intel --build-arg PREBUILD_INDEX=true .
```

Run the container:

```bash
docker run --rm -p 8000:8000 product-intel
```

Pass LLM configuration for external provider (optional):

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -e OPENAI_MODEL=gpt-4o-mini \
  product-intel
```

Health check locally:

```bash
curl http://localhost:8000/healthz
```

Notes:

- If `PREBUILD_INDEX=true` is set during build, the index is created inside the image from `data/docs`. If you change docs later, rebuild the image or let the app rebuild at container start.
- The image is based on Python 3.13.7 and runs the service as a non-root user.

## Environment variables

LLM selection is controlled only by a boolean flag in requests; set these env vars to configure providers:

- External (OpenAI):
	- `OPENAI_API_KEY` (or `LLM_API_KEY`)
	- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
	- `OPENAI_MODEL` (default: `gpt-4o-mini`)

- Local (Ollama):
	- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
	- `OLLAMA_MODEL` (default: `gemma3n:e2b`)

If the provider is unreachable or misconfigured, the service will fall back to returning the top retrieved passage snippet.

## API

- GET `/healthz` — liveness/health probe

- POST `/recognize` — recognize product from an image
	- multipart/form-data body: `image` (file: .jpg/.png)
	- Response: top catalog candidates with `best_product_id`

- POST `/products/{product_id}/answer` — answer a question about a product
	- JSON body: `{ "question": "...", "use_external_llm": false }`
	- Retrieves product-specific passages then answers with LLM or falls back

- POST `/recognize-and-answer` — one call to recognize then answer
	- multipart/form-data body:
		- `image` (file: .jpg/.png)
		- `question` (optional string)
		- `use_external_llm` (optional boolean; default false)
	- Response model includes: `candidates`, `best_product_id`, and optionally `answer`, `context_sources`

OpenAPI docs are available at `/docs` when running locally.

## Example requests

bash-friendly curl examples:

- Recognize (image-only):

```bash
curl -X POST -F "image=@path/to/your/photo.jpg" http://localhost:8000/recognize
```

Expected response (example):

```json
{
	"candidates": [
		{"product_id": "iphone-15-pro", "title": "iPhone 15 Pro", "score": 0.82, "evidence": ["brand match (apple) x 1.00", "model match (15 pro) x 1.00"]},
		{"product_id": "iphone-15", "title": "iPhone 15", "score": 0.54, "evidence": ["brand match (apple) x 1.00"]}
	],
	"best_product_id": "iphone-15-pro"
}
```

- Q&A about a known product (JSON body):

```bash
curl -X POST http://localhost:8000/products/iphone-15-pro/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "What chip does it use?", "use_external_llm": false}'
```

Example response:

```json
{
	"answer": "It uses the A17 Pro chip.",
	"context_sources": ["iphone-15-pro.txt#c2", "iphone-15-pro.txt#c3"]
}
```

- One-shot recognize + answer (multipart form):

```bash
curl -X POST http://localhost:8000/recognize-and-answer \
  -F "image=@path/to/your/photo.jpg" \
  -F "question=What is the display size?" \
  -F "use_external_llm=false"
```

Example response:

```json
{
	"candidates": [{"product_id": "iphone-15-pro", "title": "iPhone 15 Pro", "score": 0.82, "evidence": ["brand match (apple) x 1.00"]}],
	"best_product_id": "iphone-15-pro",
	"answer": "It has a 6.1-inch display.",
	"context_sources": ["iphone-15-pro.txt#c1"]
}
```

## Local LLM (Ollama)

To use the local LLM path (use_external_llm = false), install and run Ollama:

1) Install Ollama
	 - Windows/macOS: https://ollama.com/download
	 - Linux: follow the official instructions (curl script) or package manager

2) Start the server (usually auto-starts)

```bash
ollama serve
```

3) Pull a model (default model in app)

```bash
ollama pull gemma3n:e2b
```

4) Test the API

```bash
curl http://localhost:11434/api/version
```

The app calls an OpenAI-compatible endpoint, so set:

```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
```

### Container networking: reaching Ollama from the API

If the API runs inside Docker and Ollama runs on your host, `localhost` inside the container is not your host. Use one of these:

- Windows/macOS Docker Desktop:
	- Set `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`

- Linux:
	- Run the container with host gateway entry and target `host.docker.internal`:
		- `docker run --add-host host.docker.internal:host-gateway ...`
		- `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`
	- Or run with `--network host` (Linux only), then `http://localhost:11434/v1` works

## Data and indexing

- Index artifacts are written to `data/index/`:
	- `index.faiss`, `passages.jsonl`, `meta.json`
- The builder performs semantic chunking and enforces a token budget per chunk to improve retrieval quality.

## Notes

- OCR uses `pytesseract` and Pillow; ensure Tesseract OCR is installed and on PATH if needed.
- Barcode parsing uses `pyzbar` if installed; otherwise it's skipped gracefully.
- If you change files under `data/docs`, rebuild the index or restart the API to trigger auto-build.

## Models used and rationale

- OCR: Tesseract via pytesseract (robust open-source OCR; widely available on all platforms)
- Barcode: pyzbar + system libzbar (optional; gracefully skipped if not present)
- Embeddings: intfloat/e5-base-v2 (sentence-transformers)
	- Strong retrieval quality at reasonable speed and footprint
	- Normalized embeddings enable cosine similarity with FAISS IndexFlatIP
- FAISS: IndexFlatIP for efficient vector search on normalized vectors (cosine via dot product)
- Chunking: semantic chunking with sentence-aware splits; tokenizer-enforced ~512 token max per chunk for better LLM context fit
- LLM:
	- External: OpenAI (default model gpt-4o-mini) when use_external_llm=true
	- Local: Ollama (default model gemma3n:e2b, small and high quality, popular choice for resilient, on-edge-devices deployment) via OpenAI-compatible /v1/chat/completions when use_external_llm=false
	- If a provider is unreachable, the app falls back to returning a snippet from the top retrieved passage

