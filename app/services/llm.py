"""
LLM answering utilities with OpenAI-compatible chat completions.

Selection via boolean flag:
- use_external_llm = True  -> Call OpenAI servers (https://api.openai.com/v1)
- use_external_llm = False -> Call local Ollama server (http://localhost:11434/v1)

Environment variables:
- OPENAI_BASE_URL (optional): defaults to https://api.openai.com/v1
- OPENAI_API_KEY: required for OpenAI cloud
- OPENAI_MODEL: model name for OpenAI (default: gpt-4o-mini)
- OLLAMA_BASE_URL (optional): defaults to http://localhost:11434/v1
- OLLAMA_MODEL (optional): model name for Ollama (default: llama3.1)
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any
import re
import os
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def _openai_chat_completion(messages: List[Dict[str, str]], *, model: str, base_url: str, api_key: Optional[str], max_tokens: int = 400, temperature: float = 0.2, timeout: int = 30) -> Optional[str]:
	"""Minimal OpenAI-compatible chat completions call using stdlib HTTP.

	Works with OpenAI cloud or local Ollama. Returns assistant message content or None.
	"""
	url = base_url.rstrip("/") + "/chat/completions"
	payload = {
		"model": model,
		"messages": messages,
		"temperature": temperature,
		"max_tokens": max_tokens,
		"stream": False,
	}
	data = json.dumps(payload).encode("utf-8")
	headers = {"Content-Type": "application/json"}
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"
	req = Request(url, data=data, headers=headers, method="POST")
	try:
		with urlopen(req, timeout=timeout) as resp:
			body = resp.read().decode("utf-8")
			obj = json.loads(body)
			choices = obj.get("choices") or []
			if choices and isinstance(choices, list):
				msg = choices[0].get("message") or {}
				content = msg.get("content")
				if isinstance(content, str):
					return content.strip()
	except (HTTPError, URLError, TimeoutError, ValueError) as e:
		print(f"[llm] chat completion error: {e}")
	return None


def answer_question(question: str, contexts: List[Dict[str, Any]], use_external_llm: bool = False) -> str:
	texts = [str(c.get("text", "")) for c in contexts]

	# Build compact context block (shared for both providers)
	context_lines: List[str] = []
	for c in contexts:
		src = str(c.get("source", ""))
		txt = str(c.get("text", "")).strip()
		if not txt:
			continue
		context_lines.append(f"- [{src}] {txt}")
	context_block = "\n".join(context_lines) if context_lines else "(no context)"

	# Select provider via boolean flag
	if use_external_llm:
		base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
		api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
		model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
	else:
		base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
		api_key = os.environ.get("OLLAMA_API_KEY")  # typically not needed for local
		model = os.environ.get("OLLAMA_MODEL", "gemma3n:e2b")

	messages = [
		{
			"role": "system",
			"content": (
				"You are a helpful assistant that answers strictly using the provided context. "
				"If the answer cannot be found in the context, say you don't know. Keep answers concise."
			),
		},
		{
			"role": "user",
			"content": f"Question: {question}\n\nContext:\n{context_block}",
		},
	]

	completion = _openai_chat_completion(
		messages,
		model=model,
		base_url=base_url,
		api_key=api_key,
		max_tokens=400,
		temperature=0.2,
	)
	if isinstance(completion, str) and completion.strip():
		return completion.strip()

	# Heuristic fallback: return the first/best passage trimmed
	if texts:
		snippet = texts[0].strip()
		return snippet

	return "I'm not seeing enough information to answer from the local docs."

