"""
OCR and barcode extraction service utilities.

- Opens images from bytes using Pillow
- Runs OCR via Tesseract (pytesseract)
- Detects barcodes using pyzbar when available; falls back to regex over OCR text

This module is dependency-tolerant: if optional libs are missing, it
returns informative errors rather than crashing.
"""
from __future__ import annotations

from io import BytesIO
from typing import List, Optional, Dict, Tuple
import re

# Optional imports with graceful fallbacks
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from pytesseract import TesseractNotFoundError  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
    class TesseractNotFoundError(Exception):
        pass

try:
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore
except Exception:
    zbar_decode = None  # type: ignore


DEFAULT_TESSERACT_CONFIG = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-+&$#"


def _open_image(image_bytes: bytes):
    if Image is None:
        raise RuntimeError("Pillow (PIL) is not installed. Install 'pillow' to enable image processing.")
    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _run_ocr(img, config: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return (extracted_text, ocr_error)."""
    if pytesseract is None:
        return None, "pytesseract not installed. Add 'pytesseract' and install Tesseract OCR binary."
    try:
        text = pytesseract.image_to_string(img)
        return (text.strip() if text is not None else None), None
    except TesseractNotFoundError:
        return None, "Tesseract OCR binary not found on system. Install it and ensure it's on PATH."
    except Exception as e:
        return None, f"OCR failed: {e}"


def _decode_barcodes(img) -> List[str]:
    """Use pyzbar if available; otherwise return empty list for fallback."""
    results: List[str] = []
    if zbar_decode is None:
        return results
    try:
        decoded = zbar_decode(img)
        for d in decoded:
            try:
                data = d.data.decode("utf-8") if hasattr(d, "data") and d.data is not None else None
            except Exception:
                data = None
            if data:
                results.append(data)
    except Exception:
        # swallow pyzbar errors; we'll fallback to regex
        pass
    return results


def _barcode_regex_from_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    # Common barcode lengths: EAN-8 (8), UPC-A (12), EAN-13 (13), ITF-14 (14)
    matches = re.findall(r"(?<!\\d)(?:\\d{8}|\\d{12,14})(?!\\d)", text)
    seen = set()
    out: List[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def extract_from_image_bytes(image_bytes: bytes, *, tesseract_config: Optional[str] = None) -> Dict[str, Optional[object]]:
    """
    High-level API to extract OCR text and barcode candidates from image bytes.

    Returns a dictionary with keys:
    - extracted_text: Optional[str]
    - barcodes: List[str]
    - barcode_method: Optional[str] ("pyzbar" or "ocr-regex")
    - ocr_error: Optional[str]
    """
    img = _open_image(image_bytes)

    extracted_text, ocr_error = _run_ocr(img, config=tesseract_config)

    barcodes = _decode_barcodes(img)
    barcode_method: Optional[str] = None
    if barcodes:
        barcode_method = "pyzbar"
    else:
        # fallback to OCR-based regex
        barcodes = _barcode_regex_from_text(extracted_text)
        if barcodes:
            barcode_method = "ocr-regex"

    return {
        "extracted_text": extracted_text,
        "barcodes": barcodes,
        "barcode_method": barcode_method,
        "ocr_error": ocr_error,
    }
