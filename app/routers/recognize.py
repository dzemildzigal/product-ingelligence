from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List
from app.services.ocr import extract_from_image_bytes
from app.services.matcher import rank_catalog
from app.schemas import RecognizeResponse, RecognizeParams


router = APIRouter(
    prefix="/recognize",
    tags=["recognition"]
)

@router.post("", response_model=RecognizeResponse)
async def recognize_product_from_image(
    image: UploadFile = File(...),
    params: RecognizeParams = Depends(RecognizeParams.as_form),
) -> RecognizeResponse:
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only .jpg and/or .png files are allowed")
    
    # Process the uploaded image
    image_data = await image.read()
    try:
        result = extract_from_image_bytes(image_data)
    except RuntimeError as e:
        # Raised when optional dependencies like Pillow are missing
        raise HTTPException(status_code=500, detail=str(e))

    # Build a query text for matching: prefer OCR text, else join barcodes
    extracted_raw = result.get("extracted_text")
    extracted_text = extracted_raw.strip() if isinstance(extracted_raw, str) else ""
    raw_barcodes = result.get("barcodes")
    barcodes: List[str] = raw_barcodes if isinstance(raw_barcodes, list) else []
    if extracted_text:
        query_text = " ".join(barcodes + [extracted_text]) if barcodes else extracted_text
    else:
        query_text = " ".join(barcodes) if barcodes else ""

    candidates: List[dict]
    if query_text:
        top_k = max(1, int(params.top_k)) if isinstance(params.top_k, int) else 3
        candidates = rank_catalog(query_text, top_k=top_k)
    else:
        candidates = []

    best_product_id = candidates[0]["product_id"] if candidates else None

    # Coerce to typed Candidate models for clearer validation
    from app.schemas import Candidate
    typed_candidates = [Candidate(**c) for c in candidates]
    return RecognizeResponse(candidates=typed_candidates, best_product_id=best_product_id)