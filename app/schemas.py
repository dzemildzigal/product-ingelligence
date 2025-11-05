from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel
from fastapi import Form


class Candidate(BaseModel):
    product_id: Optional[str] = None
    title: Optional[str] = None
    score: float
    evidence: List[str]


class RecognizeResponse(BaseModel):
    candidates: List[Candidate]
    best_product_id: Optional[str] = None


class RecognizeAndAnswerParams(BaseModel):
    question: Optional[str] = None
    use_external_llm: bool = False

    @classmethod
    def as_form(
        cls,
        question: Optional[str] = Form(None),
        use_external_llm: bool = Form(False),
    ) -> "RecognizeAndAnswerParams":
        return cls(question=question, use_external_llm=use_external_llm)


class RecognizeAndAnswerResponse(BaseModel):
    candidates: List[Candidate]
    best_product_id: Optional[str] = None
    answer: Optional[str] = None
    context_sources: Optional[List[str]] = None
    answer_error: Optional[str] = None


# Optional params for the /recognize endpoint (multipart form alongside the image)
class RecognizeParams(BaseModel):
    top_k: int = 3

    @classmethod
    def as_form(
        cls,
        top_k: int = Form(3),
    ) -> "RecognizeParams":
        return cls(top_k=top_k)


# Product Q&A schemas
class ProductAnswerRequest(BaseModel):
    question: str
    use_external_llm: bool = False


class ProductAnswerResponse(BaseModel):
    answer: str
    context_sources: List[str]
