from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_observability import AgentObservability
from context_compaction import MessageCompactor
from research_harness import CircuitBreaker, LoopDetector, ToolHarness
from research_memory import HybridMemoryStore, bm25_rank


class FakeEmbedder:
    model_id = "fake-semantic-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if any(word in text for word in ("论文", "文献", "学术", "资料")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(word in text for word in ("咖啡", "饮品")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class LoopDetectorTests(unittest.TestCase):
    def test_repetition_injects_hint_but_changed_result_resets_progress(self):
        detector = LoopDetector(repeat_threshold=3)
        for _ in range(2):
            self.assertIsNone(detector.observe("search", {"q": "agent"}, {"hits": [1]}, "success"))
        self.assertIn(
            "没有新进展",
            detector.observe("search", {"q": "agent"}, {"hits": [1]}, "success"),
        )
        self.assertIsNone(detector.observe("search", {"q": "agent"}, {"hits": [2]}, "success"))

    def test_arguments_and_sessions_are_resettable(self):
        detector = LoopDetector(repeat_threshold=2)
        detector.observe("search", {"page": 1}, [], "success")
        self.assertIsNone(detector.observe("search", {"page": 2}, [], "success"))
        detector.reset()
        self.assertIsNone(detector.observe("search", {"page": 1}, [], "success"))


class CircuitBreakerTests(unittest.TestCase):
    def test_open_half_open_and_recovery(self):
        breaker = CircuitBreaker(failure_threshold=2, reset_seconds=10)
        breaker.record_failure("web", now=1)
        self.assertEqual(breaker.status("web"), "closed")
        breaker.record_failure("web", now=2)
        self.assertEqual(breaker.status("web"), "open")
        self.assertFalse(breaker.allow("web", now=5))
        self.assertTrue(breaker.allow("web", now=12))
        self.assertEqual(breaker.status("web"), "half_open")
        breaker.record_success("web")
        self.assertEqual(breaker.status("web"), "closed")

    def test_harness_returns_structured_error_and_short_circuits(self):
        harness = ToolHarness(circuit_breaker=CircuitBreaker(failure_threshold=2))

        def fail():
            raise RuntimeError("temporary failure")

        try:
            self.assertEqual(harness.execute("search_web", {}, fail).state, "error")
            self.assertEqual(harness.execute("search_web", {}, fail).state, "error")
            stopped = harness.execute("search_web", {}, fail)
            self.assertEqual(stopped.state, "circuit_open")
            self.assertIn("熔断器", stopped.model_text())
        finally:
            harness.close()


class ContextCompactionTests(unittest.TestCase):
    def test_only_old_tool_body_is_compacted(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "x" * 5000},
            {"role": "assistant", "content": "continue"},
            {"role": "user", "content": "latest"},
        ]
        result = MessageCompactor(
            trigger_chars=100,
            keep_recent_messages=2,
            tool_preview_chars=200,
        ).compact(messages)
        self.assertEqual(result.compacted_tools, 1)
        self.assertLess(result.after_chars, result.before_chars)
        self.assertEqual(result.messages[2]["tool_calls"][0]["id"], "call-1")
        self.assertIn('"compacted": true', result.messages[3]["content"])
        self.assertEqual(result.messages[-1]["content"], "latest")


class ObservabilityTests(unittest.TestCase):
    def test_trace_defaults_to_redacted_hashes(self):
        class Observation:
            def __init__(self):
                self.updates = []

            def update(self, **kwargs):
                self.updates.append(kwargs)

        class Context:
            def __init__(self, observation):
                self.observation = observation

            def __enter__(self):
                return self.observation

            def __exit__(self, *_args):
                return False

        class Client:
            def __init__(self):
                self.options = None
                self.observation = Observation()

            def start_as_current_observation(self, **kwargs):
                self.options = kwargs
                return Context(self.observation)

            def flush(self):
                pass

        client = Client()
        tracing = AgentObservability(client)
        with tracing.span("research", input_data={"secret": "private text"}) as span:
            span.update_output({"answer": "private result"})
        self.assertTrue(client.options["input"]["content_redacted"])
        self.assertNotIn("private text", str(client.options))
        self.assertNotIn("private result", str(client.observation.updates))


class HybridMemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.path = Path(self.directory.name) / "memory.db"

    def tearDown(self):
        self.directory.cleanup()

    def test_bm25_understands_chinese_bigrams(self):
        ranked = bm25_rank("浏览器研究", ["浏览器自动化研究", "咖啡制作方法"])
        self.assertEqual(ranked[0][1], 0)

    def test_hybrid_recall_delete_and_persistence(self):
        store = HybridMemoryStore(self.path, embedder=FakeEmbedder())
        paper = store.save("建立可靠的文献检索与来源核验流程", "研究")
        store.save("手冲咖啡的水温记录", "生活")
        recalled = store.recall("怎样查找学术资料", limit=2)
        self.assertEqual(recalled["retrieval"], "hybrid_bm25_embedding")
        self.assertEqual(recalled["memories"][0]["id"], paper["memory_id"])

        reopened = HybridMemoryStore(self.path, embedder=FakeEmbedder())
        self.assertTrue(reopened.delete(paper["memory_id"])["deleted"])
        self.assertNotIn(
            paper["memory_id"],
            [item["id"] for item in reopened.list()["memories"]],
        )

    def test_reindex_adds_vectors_to_legacy_rows(self):
        legacy = HybridMemoryStore(self.path)
        legacy.save("旧的文献研究记忆", "研究")
        upgraded = HybridMemoryStore(self.path, embedder=FakeEmbedder())
        self.assertEqual(upgraded.reindex()["indexed"], 1)
        self.assertEqual(
            upgraded.recall("学术资料", 1)["retrieval"],
            "hybrid_bm25_embedding",
        )


if __name__ == "__main__":
    unittest.main()
