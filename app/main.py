from fastapi import FastAPI
from pathlib import Path
from contextlib import asynccontextmanager
from app.routers import products, recognize

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure FAISS index exists, then initialize RAG
    try:
        root = Path(__file__).resolve().parents[1]
        index_dir = root / "data" / "index"
        index_path = index_dir / "index.faiss"
        passages_path = index_dir / "passages.jsonl"
        meta_path = index_dir / "meta.json"

        needs_build = not (index_path.exists() and passages_path.exists() and meta_path.exists())
        if needs_build:
            try:
                from tools.build_faiss_index import load_corpus, build_index  # type: ignore
                passages = load_corpus()
                build_index(passages)
            except Exception as e:
                print(f"[lifespan] FAISS index build skipped due to error: {e}")

        try:
            from app.services import rag
            rag.initialize()
        except Exception as e:
            print(f"[lifespan] RAG initialize error: {e}")
    except Exception as e:
        print(f"[lifespan] Unexpected startup error: {e}")

    yield

    # Shutdown: nothing to clean up currently


app = FastAPI(
    title="Product Intelligence API",
    description="An API for product recognition and catalog management",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(products.router)
app.include_router(recognize.router)


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)