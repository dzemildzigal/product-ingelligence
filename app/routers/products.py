from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.services import rag
from app.services.llm import answer_question

router = APIRouter(
    prefix="/products",
    tags=["products"]
)


class ProductQuery(BaseModel):
    question: str
    use_external_llm: bool = False


class AnswerResponse(BaseModel):
    answer: str
    context_sources: List[str]


@router.post("/{product_id}/answer", response_model=AnswerResponse)
async def answer_for_product(product_id: str, body: ProductQuery) -> AnswerResponse:
    """Answer a question about a specific product via a simple RAG pipeline."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # Ensure RAG is initialized
    rag.initialize()

    contexts = rag.retrieve_for_product(product_id, question, top_k=3)
    answer = answer_question(question, contexts, use_external_llm=body.use_external_llm)

    return AnswerResponse(
        answer=answer,
        context_sources=[str(c.get("source")) for c in contexts],
    )