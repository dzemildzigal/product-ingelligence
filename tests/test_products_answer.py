from __future__ import annotations

import time
import unittest
from typing import List

from fastapi.testclient import TestClient

# Import the FastAPI app
from app.main import app


class TestProductAnswer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_product_answer_latency_accuracy(self):
        """
        Route: POST /products/{product_id}/answer

        - Measures simple latency (wall-clock) and asserts it's under a loose threshold.
        - Stubs out LLM call to avoid network dependency and heavy model loads.
        - Forces retrieval to use lightweight token Jaccard fallback (no embeddings model).
        - Checks a minimal accuracy criterion: expected keyword appears in the answer.
        """
        import app.services.llm as llm
        import app.services.rag as rag
        from unittest.mock import patch

        product_id = "iphone-15-pro"
        payload = {"question": "What chip does it use?", "use_external_llm": False}

        with patch.object(rag, "_ensure_model", return_value=None), \
             patch.object(llm, "_openai_chat_completion", return_value=None):

            t0 = time.perf_counter()
            resp = self.client.post(f"/products/{product_id}/answer", json=payload)
            dt = time.perf_counter() - t0

        self.assertEqual(resp.status_code, 200, msg=resp.text)

        data = resp.json()
        answer = data.get("answer", "")

        # Latency metric (seconds)
        latency_s = dt

        # Simple accuracy metric: keyword presence
        expected_keywords: List[str] = ["a17"]  # from docs, expect A17 Pro mention
        ans_lower = answer.lower()
        keyword_hits = sum(1 for k in expected_keywords if k in ans_lower)
        accuracy = keyword_hits / max(1, len(expected_keywords))

        # Print metrics for visibility in test output
        print({
            "latency_s": round(latency_s, 3),
            "accuracy": round(accuracy, 3),
            "answer_preview": answer[:120],
        })

        # Assertions with conservative thresholds
        self.assertLess(latency_s, 2.0, msg=f"Route too slow: {latency_s:.3f}s")
        self.assertGreaterEqual(accuracy, 1.0, msg=f"Expected keywords missing in answer: {answer}")
