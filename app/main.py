from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List
from pydantic import BaseModel
from app.schemas import Candidate, RecognizeAndAnswerParams, RecognizeAndAnswerResponse
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



@app.post("/recognize-and-answer", response_model=RecognizeAndAnswerResponse)
async def recognize_and_answer(
    image: UploadFile = File(...),
    params: RecognizeAndAnswerParams = Depends(RecognizeAndAnswerParams.as_form),
) -> RecognizeAndAnswerResponse:
    """Recognize product from image and optionally answer a question in one call.

    - image: uploaded image (jpg/png)
    - question: optional question to answer about the recognized product
    - use_external_llm: True = OpenAI cloud, False = local Ollama
    """
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only .jpg and/or .png files are allowed")

    # Lazy imports to avoid circulars and heavy deps at import time
    from app.services.ocr import extract_from_image_bytes
    from app.services.matcher import rank_catalog
    from app.services import rag
    from app.services.llm import answer_question

    # Process the uploaded image
    img_bytes = await image.read()
    try:
        ocr_result = extract_from_image_bytes(img_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build query text: prefer OCR text, include barcodes if present
    extracted_text = str(ocr_result.get("extracted_text") or "").strip()
    raw_barcodes = ocr_result.get("barcodes")
    barcodes: List[str] = raw_barcodes if isinstance(raw_barcodes, list) else []
    if extracted_text:
        query_text = (" ".join([*barcodes, extracted_text]) if barcodes else extracted_text)
    else:
        query_text = " ".join(barcodes) if barcodes else ""

    candidates: List[dict] = rank_catalog(query_text, top_k=3) if query_text else []
    best_product_id = candidates[0]["product_id"] if candidates else None

    response = {
        "candidates": candidates,
        "best_product_id": best_product_id,
    }

    # If a question is provided and a product was recognized, run RAG + LLM
    if params.question and best_product_id:
        try:
            rag.initialize()
            contexts = rag.retrieve_for_product(str(best_product_id), params.question, top_k=1)
            ans = answer_question(params.question, contexts, use_external_llm=params.use_external_llm)
            response.update({
                "answer": ans,
                "context_sources": [str(c.get("source")) for c in contexts],
            })
        except Exception as e:
            # Don't fail the entire request if answering fails; return recognition at least
            response.update({
                "answer_error": str(e),
            })

    return RecognizeAndAnswerResponse(**response)


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)