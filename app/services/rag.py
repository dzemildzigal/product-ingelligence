"""
RAG retrieval utilities with FAISS-first strategy and simple fallbacks.

Public API:
- initialize() -> None
- retrieve(query: str, top_k: int = 3) -> List[Dict[str, object]]
- retrieve_for_product(product_id: str, query: str, top_k: int = 3) -> List[Dict[str, object]]

Behavior:
- If a FAISS index built by tools/build_faiss_index.py exists under data/index and
  sentence-transformers is available, use it for retrieval.
- Otherwise, load docs from data/docs, chunk them, embed in-memory if the model is
  available, or fall back to token Jaccard similarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
import re
import json
import numpy as np

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _ROOT / "data" / "docs"
_INDEX_DIR = _ROOT / "data" / "index"
_FAISS_INDEX_PATH = _INDEX_DIR / "index.faiss"
_PASSAGES_PATH = _INDEX_DIR / "passages.jsonl"
_META_PATH = _INDEX_DIR / "meta.json"

_TOKENIZER = re.compile(r"[a-z0-9]+")
_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL", "intfloat/e5-base-v2")

# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------
try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
	SentenceTransformer = None  # type: ignore

# Note: LangChain semantic chunker support removed for simplicity and stability.

_FAISS = None  # Lazy-import faiss


def _ensure_faiss():
	global _FAISS
	if _FAISS is None:
		try:
			import faiss  # type: ignore
			_FAISS = faiss
		except Exception:
			_FAISS = None
	return _FAISS


# --------------------------------------------------------------------------------------
# Runtime state
# --------------------------------------------------------------------------------------
_EMBED_MODEL = None  # sentence-transformers model
_PASSAGES: List[str] = []
_SOURCES: List[str] = []  # passage sources, aligned with _PASSAGES
_EMB: Optional[np.ndarray] = None  # in-memory passage embeddings (N x D)
_FAISS_INDEX = None  # loaded faiss index


def _ensure_model():
	global _EMBED_MODEL
	if _EMBED_MODEL is None and SentenceTransformer is not None:
		try:
			_EMBED_MODEL = SentenceTransformer(_MODEL_NAME)
		except Exception:
			_EMBED_MODEL = None
	return _EMBED_MODEL


def _normalize(text: str) -> str:
	return text.replace("\r", "").strip()


def _sent_tokenize(text: str) -> List[str]:
	parts = re.split(r"[\n\.]+", text)
	return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, max_chars: int = 600) -> List[str]:
	chunks: List[str] = []
	cur: List[str] = []
	cur_len = 0
	for sent in _sent_tokenize(_normalize(text)):
		if not sent:
			continue
		if cur_len + len(sent) + 1 > max_chars:
			if cur:
				chunks.append(" ".join(cur))
			cur = [sent]
			cur_len = len(sent)
		else:
			cur.append(sent)
			cur_len += len(sent) + 1
	if cur:
		chunks.append(" ".join(cur))
	return chunks


def _load_docs() -> None:
	"""Load and chunk docs from data/docs into _PASSAGES and _SOURCES."""
	global _PASSAGES, _SOURCES
	_PASSAGES = []
	_SOURCES = []
	if not _DOCS_DIR.exists():
		return
	for path in sorted(_DOCS_DIR.glob("*.txt")):
		try:
			text = path.read_text(encoding="utf-8")
		except Exception:
			continue

		# Simple chunking to avoid extra optional dependencies
		chunks = _chunk_text(text)

		for idx, ch in enumerate(chunks, start=1):
			_PASSAGES.append(ch)
			_SOURCES.append(f"{path.name}#c{idx}")


def _embed_passages() -> None:
	"""Compute in-memory embeddings for _PASSAGES if model available."""
	global _EMB
	model = _ensure_model()
	if model is None or not _PASSAGES:
		_EMB = None
		return
	try:
		texts = [f"passage: {t}" for t in _PASSAGES]
		emb = model.encode(texts, normalize_embeddings=True)
		_EMB = emb if isinstance(emb, np.ndarray) else np.asarray(emb)
	except Exception:
		_EMB = None


def _tokens(s: str) -> set[str]:
	return set(_TOKENIZER.findall(s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
	if not a or not b:
		return 0.0
	inter = len(a & b)
	union = len(a | b)
	return inter / union


def _load_faiss_index() -> None:
	"""Load FAISS index and passages if artifacts exist."""
	global _FAISS_INDEX, _PASSAGES, _SOURCES
	faiss_mod = _ensure_faiss()
	if faiss_mod is None:
		return
	try:
		if not (_FAISS_INDEX_PATH.exists() and _PASSAGES_PATH.exists() and _META_PATH.exists()):
			return
		_FAISS_INDEX = faiss_mod.read_index(str(_FAISS_INDEX_PATH))
		# Load passages and sources
		_PASSAGES = []
		_SOURCES = []
		with _PASSAGES_PATH.open("r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				d = json.loads(line)
				_PASSAGES.append(d.get("text", ""))
				_SOURCES.append(str(d.get("source", "")))
	except Exception:
		_FAISS_INDEX = None


def initialize() -> None:
	"""Initialize retrieval backends: prefer FAISS, else in-memory docs+embeddings."""
	if _FAISS_INDEX is None:
		_load_faiss_index()
	if _FAISS_INDEX is None:
		_load_docs()
		_embed_passages()


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, object]]:
	"""Return top_k passages with scores and sources. Prefer FAISS if usable."""
	if not _PASSAGES and _FAISS_INDEX is None:
		initialize()

	faiss_mod = _ensure_faiss()
	model = _ensure_model()

	# FAISS path
	if _FAISS_INDEX is not None and model is not None and faiss_mod is not None:
		try:
			q = model.encode([f"query: {query}"], normalize_embeddings=True)
			q = q if isinstance(q, np.ndarray) else np.asarray(q)
			q = q.astype("float32")
			D, I = _FAISS_INDEX.search(q, top_k)
			out: List[Dict[str, object]] = []
			for idx, score in zip(I[0], D[0]):
				if idx < 0:
					continue
				out.append({
					"text": _PASSAGES[idx],
					"score": round(float(score), 4),
					"source": _SOURCES[idx],
				})
			return out
		except Exception:
			pass

	# In-memory embeddings path
	if model is not None and _EMB is not None:
		try:
			q = model.encode([f"query: {query}"], normalize_embeddings=True)
			q = q if isinstance(q, np.ndarray) else np.asarray(q)
			q = q[0]
			scores = _EMB @ q  # cosine similarity (normalized)
			idxs = np.argsort(-scores)[: max(0, top_k)]
			return [
				{"text": _PASSAGES[i], "score": round(float(scores[i]), 4), "source": _SOURCES[i]}
				for i in idxs
			]
		except Exception:
			pass

	# Token Jaccard fallback
	results: List[Tuple[int, float]] = []
	qtok = _tokens(query)
	for i, p in enumerate(_PASSAGES):
		results.append((i, _jaccard(qtok, _tokens(p))))
	results.sort(key=lambda x: x[1], reverse=True)
	top = results[: max(0, top_k)]
	return [{"text": _PASSAGES[i], "score": round(float(s), 4), "source": _SOURCES[i]} for i, s in top]


def retrieve_for_product(product_id: str, query: str, top_k: int = 3) -> List[Dict[str, object]]:
	"""Retrieve passages restricted to documents matching product_id in filename."""
	if not _PASSAGES and _FAISS_INDEX is None:
		initialize()

	pid = (product_id or "").lower()
	faiss_mod = _ensure_faiss()
	model = _ensure_model()

	# FAISS path with filtering
	if _FAISS_INDEX is not None and model is not None and faiss_mod is not None:
		try:
			q = model.encode([f"query: {query}"], normalize_embeddings=True)
			q = q if isinstance(q, np.ndarray) else np.asarray(q)
			q = q.astype("float32")
			D, I = _FAISS_INDEX.search(q, max(50, top_k))
			filtered: List[Dict[str, object]] = []
			for idx, score in zip(I[0], D[0]):
				if idx < 0:
					continue
				src = _SOURCES[idx]
				fname = src.split('#', 1)[0].lower()
				if pid and pid in fname:
					filtered.append({
						"text": _PASSAGES[idx],
						"score": round(float(score), 4),
						"source": src,
					})
				if len(filtered) >= top_k:
					break
			if filtered:
				return filtered
		except Exception:
			pass

	# In-memory embeddings path
	pairs: List[Tuple[int, float]] = []
	if model is not None and _EMB is not None:
		try:
			q = model.encode([f"query: {query}"], normalize_embeddings=True)
			q = q if isinstance(q, np.ndarray) else np.asarray(q)
			q = q[0]
			scores = _EMB @ q
			pairs = [(i, float(scores[i])) for i in range(len(_PASSAGES))]
		except Exception:
			pairs = []

	# Token Jaccard fallback if needed
	if not pairs:
		qtok = _tokens(query)
		for i, p in enumerate(_PASSAGES):
			pairs.append((i, _jaccard(qtok, _tokens(p))))

	# Filter for product id
	filtered_pairs: List[Tuple[int, float]] = []
	for i, s in pairs:
		src = _SOURCES[i]
		fname = src.split('#', 1)[0].lower()
		if pid and pid in fname:
			filtered_pairs.append((i, s))

	selected = (
		sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
		if filtered_pairs else sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
	)
	return [{"text": _PASSAGES[i], "score": round(float(s), 4), "source": _SOURCES[i]} for i, s in selected]

