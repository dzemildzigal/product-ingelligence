"""
Build a FAISS index over local docs in data/docs.

Outputs under data/index/:
- passages.jsonl: one JSON per line with {"text", "source"}
- meta.json: {"model": str, "dim": int}
- index.faiss: FAISS index file (IndexFlatIP over L2-normalized embeddings)

Usage (optional): run as a script. It will (re)build the index.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any
from pathlib import Path
import os
import json
import numpy as np

try:
	import faiss  # type: ignore
except Exception as e:
	raise RuntimeError("faiss-cpu is required to build the index. Install 'faiss-cpu'.") from e

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
	raise RuntimeError("sentence-transformers is required to build the index.") from e

# Optional semantic chunker (LangChain). We'll fall back gracefully if unavailable.
try:
	from langchain_text_splitters import SemanticChunker  # type: ignore
except Exception:
	try:
		from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
	except Exception:
		SemanticChunker = None  # type: ignore

try:
	from langchain_core.embeddings import Embeddings  # type: ignore
except Exception:
	Embeddings = None  # type: ignore

# Optional: use the model's tokenizer (from transformers) to enforce token limits
try:
	from transformers import AutoTokenizer  # type: ignore
except Exception:
	AutoTokenizer = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PASSAGES_PATH = OUT_DIR / "passages.jsonl"
META_PATH = OUT_DIR / "meta.json"
INDEX_PATH = OUT_DIR / "index.faiss"

MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL", "intfloat/e5-base-v2")
MAX_TOKENS = int(os.environ.get("EMBEDDINGS_MAX_TOKENS", "512"))


def _normalize_text(s: str) -> str:
	return s.replace("\r", "").strip()
# --------------------------------------------------------------------------------------
# Global model & chunker (initialized once)
# --------------------------------------------------------------------------------------
_ST_MODEL: Optional[SentenceTransformer] = None
_SEM_CHUNKER: Optional[Any] = None
_TOKENIZER: Optional[Any] = None


def ensure_model() -> SentenceTransformer:
	global _ST_MODEL
	if _ST_MODEL is None:
		_ST_MODEL = SentenceTransformer(MODEL_NAME)
	return _ST_MODEL


def ensure_semantic_chunker():
	"""Initialize and cache a single SemanticChunker instance when possible."""
	global _SEM_CHUNKER
	if _SEM_CHUNKER is not None:
		return _SEM_CHUNKER
	if SemanticChunker is None or Embeddings is None:
		return None
	try:
		model = ensure_model()
		class _STEmb(Embeddings):  # type: ignore
			def __init__(self, m):
				self.m = m
			def embed_documents(self, texts: List[str]) -> List[List[float]]:
				return self.m.encode(texts, normalize_embeddings=True).tolist()
			def embed_query(self, text: str) -> List[float]:
				return self.m.encode([text], normalize_embeddings=True)[0].tolist()

		_SEM_CHUNKER = SemanticChunker(_STEmb(model))  # type: ignore
		return _SEM_CHUNKER
	except Exception:
		_SEM_CHUNKER = None
		return None


def ensure_tokenizer():
	"""Return cached tokenizer aligned with MODEL_NAME when available."""
	global _TOKENIZER
	if _TOKENIZER is not None:
		return _TOKENIZER
	if AutoTokenizer is None:
		return None
	try:
		_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
		return _TOKENIZER
	except Exception:
		_TOKENIZER = None
		return None



def _sent_split(s: str) -> List[str]:
	import re
	# Split on sentence-ending punctuation or newlines
	parts = re.split(r"[\n\.!?]+", s)
	return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, max_chars: int = 600) -> List[str]:
	"""Semantic chunking if available; otherwise sentence-based fixed-size chunks.

	Note: SemanticChunker creates variable-length chunks by semantic coherence. We keep
	max_chars as a soft limit only for the fallback splitter.
	"""
	chunker = ensure_semantic_chunker()
	if chunker is not None:
		try:
			chunks = chunker.split_text(text)
			if chunks:
				return chunks
		except Exception:
			pass

	# Fallback: simple sentence-based chunking using max_chars window
	chunks: List[str] = []
	cur: List[str] = []
	cur_len = 0
	for sent in _sent_split(_normalize_text(text)):
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


def _token_count(text: str) -> int:
	tok = ensure_tokenizer()
	if tok is None:
		# Rough fallback: whitespace token count
		return max(1, len(text.split()))
	try:
		ids = tok.encode(text, add_special_tokens=False)
		return len(ids)
	except Exception:
		return max(1, len(text.split()))


def _split_overlong(text: str, max_tokens: int) -> List[str]:
	"""Split a text into <=max_tokens chunks, preferring sentence boundaries near the limit."""
	if _token_count(text) <= max_tokens:
		return [text]
	# Try sentence-accumulation until hitting limit
	sents = _sent_split(text)
	if not sents:
		# fallback: naive mid split
		n = len(text)
		mid = n // 2
		return [text[:mid].strip(), text[mid:].strip()]
	acc: List[str] = []
	acc_tokens = 0
	for i, s in enumerate(sents):
		if not s:
			continue
		cand = (" ".join(acc + [s])).strip()
		cand_tokens = _token_count(cand)
		if cand_tokens > max_tokens:
			left = " ".join(acc).strip()
			right = " ".join(sents[i:]).strip()
			left = left or s.strip()
			right = right if left != s.strip() else " ".join(sents[i+1:]).strip()
			parts: List[str] = []
			parts.extend(_split_overlong(left, max_tokens))
			if right:
				parts.extend(_split_overlong(right, max_tokens))
			return [p for p in parts if p]
		acc.append(s)
		acc_tokens = cand_tokens
	# Entire text within limit after accumulation (shouldn't happen since early return), but keep safe
	final = " ".join(acc).strip()
	return [final] if final else []


def load_corpus() -> List[Dict[str, str]]:
	passages: List[Dict[str, str]] = []
	if not DOCS_DIR.exists():
		return passages
	for p in sorted(DOCS_DIR.glob("*.txt")):
		try:
			text = p.read_text(encoding="utf-8")
		except Exception:
			continue
		chunks = _chunk_text(text)
		# Enforce model token limit per chunk
		final_chunks: List[str] = []
		for ch in chunks:
			final_chunks.extend(_split_overlong(ch, MAX_TOKENS))
		chunks = [c for c in final_chunks if c]
		for idx, ch in enumerate(chunks, start=1):
			passages.append({
				"text": ch,
				"source": f"{p.name}#c{idx}",
			})
	return passages


def build_index(passages: List[Dict[str, str]]) -> None:
	if not passages:
		# write empty files to indicate no data
		PASSAGES_PATH.write_text("", encoding="utf-8")
		META_PATH.write_text(json.dumps({"model": MODEL_NAME, "dim": 0}), encoding="utf-8")
		if INDEX_PATH.exists():
			INDEX_PATH.unlink()
		return

	model = ensure_model()
	texts = [f"passage: {d['text']}" for d in passages] # Source: https://huggingface.co/intfloat/e5-base-v2 , passages should be embedded with a prefix passage:
	embeddings = model.encode(texts, normalize_embeddings=True)
	embeddings = np.asarray(embeddings, dtype=np.float32)
	embeddings = np.ascontiguousarray(embeddings)
	dim = embeddings.shape[1]

	index = faiss.IndexFlatIP(dim)
	# Add vectors to index. Handle both python bindings styles.
	add_fn = getattr(index, "add")
	try:
		add_fn(embeddings)
	except TypeError:
		# Some faiss bindings expect (n, x)
		add_fn(embeddings.shape[0], embeddings)

	# Persist
	with PASSAGES_PATH.open("w", encoding="utf-8") as f:
		for d in passages:
			f.write(json.dumps(d, ensure_ascii=False) + "\n")
	META_PATH.write_text(json.dumps({"model": MODEL_NAME, "dim": int(dim)}), encoding="utf-8")
	faiss.write_index(index, str(INDEX_PATH))


def main() -> None:
	passages = load_corpus()
	build_index(passages)
	print(f"Built FAISS index with {len(passages)} passages -> {INDEX_PATH}")


if __name__ == "__main__":
	main()

