from fastapi import APIRouter, HTTPException
from typing import List, Dict
from app.services import rag
from app.services.llm import answer_question
from app.schemas import ProductAnswerRequest, ProductAnswerResponse

router = APIRouter(
    prefix="/products",
    tags=["products"]
)


@router.post("/{product_id}/answer", response_model=ProductAnswerResponse)
async def answer_for_product(product_id: str, body: ProductAnswerRequest) -> ProductAnswerResponse:
    """Answer a question about a specific product via a simple RAG pipeline."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # Ensure RAG is initialized
    rag.initialize()

    contexts = rag.retrieve_for_product(product_id, question, top_k=3)
    answer = answer_question(question, contexts, use_external_llm=body.use_external_llm)

    return ProductAnswerResponse(
        answer=answer,
        context_sources=[str(c.get("source")) for c in contexts],
    )