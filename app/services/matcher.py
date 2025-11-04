"""
Matcher service utilities.

This module loads the product catalog CSV into memory at import time and
exposes simple accessors so other modules (e.g., routers) can use it
without re-reading the file.
"""

from __future__ import annotations

from pathlib import Path
import csv
import logging
import math
from typing import Dict, List, Optional, Tuple, Set
import re
 


logger = logging.getLogger(__name__)

# Resolve catalog path relative to repository root: <repo>/data/catalog.csv
CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog.csv"

# In-memory catalog structures
CATALOG_ROWS: List[Dict[str, str]] = []
CATALOG_BY_ID: Dict[str, Dict[str, str]] = {}

# Token caches
_TITLE_TOKENS: List[Set[str]] = []
_BRAND_TOKENS: List[Set[str]] = []
_MODEL_TOKENS: List[Set[str]] = []
_DESC_TOKENS: List[Set[str]] = []



def _load_catalog() -> None:
	"""Load the product catalog CSV into memory (rows and by-id index)."""
	global CATALOG_ROWS, CATALOG_BY_ID
	try:
		with CATALOG_PATH.open("r", encoding="utf-8-sig", newline="") as f:
			reader = csv.DictReader(f)
			rows: List[Dict[str, str]] = []
			for row in reader:
				# Skip completely empty lines
				if not row or not any((value or "").strip() for value in row.values()):
					continue
				rows.append(row)
			CATALOG_ROWS = rows
			CATALOG_BY_ID = {row.get("product_id", ""): row for row in rows if row.get("product_id")}
		logger.info("Loaded catalog: %d rows from %s", len(CATALOG_ROWS), CATALOG_PATH)
	except FileNotFoundError:
		CATALOG_ROWS = []
		CATALOG_BY_ID = {}
		logger.warning("Catalog file not found at %s; continuing with empty catalog.", CATALOG_PATH)
	except Exception as e:
		# On unexpected errors, keep empty catalog but log the issue
		CATALOG_ROWS = []
		CATALOG_BY_ID = {}
		logger.exception("Failed to load catalog from %s: %s", CATALOG_PATH, e)


def get_catalog_rows() -> List[Dict[str, str]]:
	"""Accessor for catalog rows (list of dicts)."""
	return CATALOG_ROWS


def get_catalog_by_id() -> Dict[str, Dict[str, str]]:
	"""Accessor for catalog indexed by product_id."""
	return CATALOG_BY_ID


def reload_catalog() -> None:
	"""Reload the catalog from disk (callable from other modules/tests)."""
	_load_catalog()
	_build_indices()


# Load catalog at import time
_load_catalog()
 

# ------------------------- Matching utilities ------------------------------

_TOKENIZER = re.compile(r"[a-z0-9]+")


def _normalize(text: Optional[str]) -> str:
	return (text or "").lower()


def _tokens(text: Optional[str]) -> Set[str]:
	return set(_TOKENIZER.findall(_normalize(text)))


def _jaccard(a: Set[str], b: Set[str]) -> float:
	if not a or not b:
		return 0.0
	inter = a & b
	union = a | b
	return len(inter) / len(union)


def _coverage_ratio(needles: Set[str], haystack: Set[str]) -> float:
	if not needles:
		return 0.0
	return sum(1 for n in needles if n in haystack) / len(needles)


def _build_indices() -> None:
	"""Build token and embedding indices for the current catalog rows."""
	global _TITLE_TOKENS, _BRAND_TOKENS, _MODEL_TOKENS, _DESC_TOKENS
	_TITLE_TOKENS = []
	_BRAND_TOKENS = []
	_MODEL_TOKENS = []
	_DESC_TOKENS = []
	for r in CATALOG_ROWS:
		_TITLE_TOKENS.append(_tokens(r.get("title")))
		_BRAND_TOKENS.append(_tokens(r.get("brand")))
		_MODEL_TOKENS.append(_tokens(r.get("model")))
		_DESC_TOKENS.append(_tokens(r.get("description")))



# Build indices at import time
_build_indices()


def rank_catalog(query_text: str, *, top_k: int = 3, brand_boost: float = 0.3,
				 model_boost: float = 0.5, lex_boost: float = 0.2) -> List[Dict[str, object]]:
	"""
	Rank catalog products against query_text and return top_k results.

	Score components (each in [0,1]):
	- s_brand: token coverage of catalog brand tokens within the query tokens.
	- s_model: token coverage of catalog model tokens within the query tokens.
	- s_lex: Jaccard similarity between query tokens and product title tokens.

	Final score = weighted sum of components, renormalized so weights
	sum to 1.0. Returns list of dicts with keys:
	- product_id, title, score, evidence (list[str])
	"""
	q_tok = _tokens(query_text)

	results: List[Tuple[int, float, List[str]]] = []  # (index, score, evidence)

	# Normalize weights so they sum to 1.0
	components = [("brand", brand_boost), ("model", model_boost), ("lex", lex_boost)]
	total_weight = sum(w for _, w in components)
	norm_weights = {k: (w / total_weight if total_weight > 0 else 0.0) for k, w in components}

	for i, row in enumerate(CATALOG_ROWS):
		ev: List[str] = []
		score = 0.0

		# Brand coverage
		s_brand = _coverage_ratio(_BRAND_TOKENS[i], q_tok)
		if s_brand > 0:
			ev.append(f"brand match ({row.get('brand','')}) x {s_brand:.2f}")

		# Model coverage
		s_model = _coverage_ratio(_MODEL_TOKENS[i], q_tok)
		if s_model > 0:
			ev.append(f"model match ({row.get('model','')}) x {s_model:.2f}")

		# Lexical overlap with title
		s_lex = _jaccard(_TITLE_TOKENS[i], q_tok)
		if s_lex > 0:
			ev.append(f"title token overlap x {s_lex:.2f}")

		# Weighted sum with normalized weights
		comp_vals = {
			"brand": s_brand,
			"model": s_model,
			"lex": s_lex,
		}
		score = sum(norm_weights.get(k, 0.0) * comp_vals[k] for k in comp_vals)

		results.append((i, score, ev))

	# Sort and select top_k
	results.sort(key=lambda x: x[1], reverse=True)
	top = results[: max(0, top_k)]

	out: List[Dict[str, object]] = []
	for i, s, ev in top:
		row = CATALOG_ROWS[i]
		out.append({
			"product_id": row.get("product_id"),
			"title": row.get("title"),
			"score": round(float(s), 4),
			"evidence": ev,
		})
	return out


