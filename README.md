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

	 - Python 3.10+
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

## Data and indexing

- Index artifacts are written to `data/index/`:
	- `index.faiss`, `passages.jsonl`, `meta.json`
- The builder performs semantic chunking and enforces a token budget per chunk to improve retrieval quality.

## Notes

- OCR uses `pytesseract` and Pillow; ensure Tesseract OCR is installed and on PATH if needed.
- Barcode parsing uses `pyzbar` if installed; otherwise it's skipped gracefully.
- If you change files under `data/docs`, rebuild the index or restart the API to trigger auto-build.

