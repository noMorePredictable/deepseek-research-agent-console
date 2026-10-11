"""SQLite 长期记忆与 BM25 + Embedding 混合召回。"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from openai import OpenAI


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u3005]+")
_WORD = re.compile(r"[^\W_]+(?:[-_.][^\W_]+)*", re.UNICODE)


def terms(text: str) -> list[str]:
    """兼顾英文单词、完整型号与中文二元词项的轻量分词。"""
    value = unicodedata.normalize("NFKC", text).casefold()
    tokens = _WORD.findall(_CJK.sub(" ", value))
    for segment in _CJK.findall(value):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def bm25_rank(query: str, documents: Sequence[str]) -> list[tuple[float, int]]:
    """返回 ``(score, document_index)``，不依赖额外搜索库。"""
    if not documents:
        return []
    query_terms = set(terms(query))
    if not query_terms:
        return []
    counters = [Counter(terms(document)) for document in documents]
    lengths = [sum(counter.values()) for counter in counters]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency = Counter(term for counter in counters for term in counter)
    scored: list[tuple[float, int]] = []
    k1, b = 1.2, 0.75
    for index, (counter, length) in enumerate(zip(counters, lengths)):
        score = 0.0
        for term in query_terms:
            frequency = counter[term]
            if not frequency:
                continue
            count = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - count + 0.5) / (count + 0.5),
            )
            denominator = frequency + k1 * (
                1 - b + b * length / max(average_length, 1)
            )
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        if score > 0:
            scored.append((score, index))
    return sorted(scored, key=lambda item: (-item[0], item[1]))


def _valid_vector(vector: object) -> bool:
    return (
        isinstance(vector, list)
        and bool(vector)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        )
        and any(vector)
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    weights: Sequence[float],
    k: int = 60,
) -> list[tuple[float, int]]:
    """使用 RRF 融合不同量纲的 BM25 分数与余弦相似度排名。"""
    if len(rankings) != len(weights):
        raise ValueError("rankings 与 weights 数量必须一致")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, memory_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[memory_id] = scores.get(memory_id, 0.0) + weight / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], -item[0]))
    return [(score, memory_id) for memory_id, score in ordered]


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本编码为等维向量。"""


class OpenAIEmbeddingProvider:
    """调用任意 OpenAI 兼容的 ``/embeddings`` 服务。"""

    def __init__(self, *, api_key: str, model: str, base_url: str = "") -> None:
        self.model_id = model
        options = {"api_key": api_key}
        if base_url:
            options["base_url"] = base_url.rstrip("/")
        self._client = OpenAI(**options)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model_id, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts) or any(not _valid_vector(vector) for vector in vectors):
            raise ValueError("Embedding 服务返回了无效向量")
        return vectors


def embedding_provider_from_env() -> EmbeddingProvider | None:
    """显式配置才启用语义向量；不假设聊天模型服务同时支持 Embedding。"""
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not api_key or not model:
        return None
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
    )


class HybridMemoryStore:
    """兼容原 memories 表，并增量增加向量字段和记忆 CRUD。"""

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self._init_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """,
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "embedding" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
            if "embedding_model" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN embedding_model TEXT")

    def save(self, content: str, tags: str = "") -> dict:
        content = content[:8000]
        tags = tags[:300]
        vector: list[float] | None = None
        embedding_error = ""
        if self.embedder is not None:
            try:
                vector = self.embedder.embed([content])[0]
            except Exception as error:  # 记忆仍可由 BM25 检索，向量服务不能阻断写入。
                embedding_error = type(error).__name__

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories(
                    content, tags, created_at, embedding, embedding_model
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    content,
                    tags,
                    datetime.now().isoformat(timespec="seconds"),
                    json.dumps(vector) if vector is not None else None,
                    self.embedder.model_id if vector is not None and self.embedder else None,
                ),
            )
        result = {
            "saved": True,
            "memory_id": cursor.lastrowid,
            "embedding_stored": vector is not None,
        }
        if embedding_error:
            result["embedding_fallback"] = embedding_error
        return result

    @staticmethod
    def _public_row(row: sqlite3.Row | dict, *, score: float | None = None) -> dict:
        item = {
            "id": row["id"],
            "content": row["content"],
            "tags": row["tags"],
            "created_at": row["created_at"],
        }
        if score is not None:
            item["score"] = round(score, 6)
        return item

    def list(self, limit: int = 20) -> dict:
        bounded = max(1, min(limit, 100))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return {"memories": [self._public_row(row) for row in rows]}

    def delete(self, memory_id: int) -> dict:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
        return {"deleted": cursor.rowcount == 1, "memory_id": memory_id}

    def recall(self, query: str, limit: int = 5) -> dict:
        bounded = max(1, min(limit, 8))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT 500",
            ).fetchall()
        if not rows:
            return {"memories": [], "retrieval": "empty"}
        if not query.strip():
            return {
                "memories": [self._public_row(row) for row in rows[:bounded]],
                "retrieval": "recent",
            }

        documents = [f"{row['content']} {row['tags']}" for row in rows]
        lexical = bm25_rank(query, documents)
        lexical_ids = [rows[index]["id"] for _, index in lexical]

        vector_pairs: list[tuple[float, int]] = []
        if self.embedder is not None:
            try:
                query_vector = self.embedder.embed([query])[0]
                for row in rows:
                    if row["embedding_model"] != self.embedder.model_id:
                        continue
                    try:
                        vector = json.loads(row["embedding"] or "null")
                    except (TypeError, ValueError):
                        continue
                    if not _valid_vector(vector):
                        continue
                    similarity = _cosine(query_vector, vector)
                    if similarity > 0:
                        vector_pairs.append((similarity, row["id"]))
                vector_pairs.sort(key=lambda item: (-item[0], -item[1]))
            except Exception:
                vector_pairs = []

        row_by_id = {row["id"]: row for row in rows}
        if vector_pairs:
            fused = reciprocal_rank_fusion(
                [lexical_ids, [memory_id for _, memory_id in vector_pairs]],
                weights=[0.55, 0.45],
            )
            retrieval = "hybrid_bm25_embedding"
        elif lexical_ids:
            fused = reciprocal_rank_fusion([lexical_ids], weights=[1.0])
            retrieval = "bm25"
        else:
            fused = [(1 / (60 + rank), row["id"]) for rank, row in enumerate(rows, 1)]
            retrieval = "recent_fallback"

        selected = [
            self._public_row(row_by_id[memory_id], score=score)
            for score, memory_id in fused[:bounded]
        ]
        return {"memories": selected, "retrieval": retrieval}

    def reindex(self, limit: int = 100) -> dict:
        """为旧记忆或模型变更后的记忆补建向量，不改写事实正文。"""
        if self.embedder is None:
            return {"indexed": 0, "enabled": False}
        bounded = max(1, min(limit, 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE embedding IS NULL OR embedding_model IS NULL OR embedding_model != ?
                ORDER BY id DESC LIMIT ?
                """,
                (self.embedder.model_id, bounded),
            ).fetchall()
        if not rows:
            return {"indexed": 0, "enabled": True}

        vectors = self.embedder.embed([row["content"] for row in rows])
        if len(vectors) != len(rows) or any(not _valid_vector(vector) for vector in vectors):
            raise ValueError("Embedding 重建结果无效，未写入数据库")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row, vector in zip(rows, vectors):
                connection.execute(
                    "UPDATE memories SET embedding = ?, embedding_model = ? WHERE id = ?",
                    (json.dumps(vector), self.embedder.model_id, row["id"]),
                )
        return {"indexed": len(rows), "enabled": True}
