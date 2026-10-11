"""可选 Langfuse Trace；未配置时是零网络请求的 No-Op。"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


def _summary(value: Any) -> dict[str, Any]:
    """默认只记录长度与哈希，避免把问题、网页正文和工具结果上传。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "characters": len(encoded),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "content_redacted": True,
    }


@dataclass
class ObservationHandle:
    observation: Any = None
    capture_content: bool = False

    def update_output(self, value: Any) -> None:
        if self.observation is None:
            return
        payload = value if self.capture_content else _summary(value)
        self.observation.update(output=payload)


class AgentObservability:
    """为 Agent、LLM generation 和工具调用建立父子 Trace。"""

    def __init__(self, client: Any = None, *, capture_content: bool = False) -> None:
        self.client = client
        self.capture_content = capture_content

    @classmethod
    def from_env(cls) -> "AgentObservability":
        required = (
            os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
            os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
            os.getenv("LANGFUSE_BASE_URL", "").strip(),
        )
        if not all(required):
            return cls()
        try:
            from langfuse import get_client

            client = get_client()
        except (ImportError, RuntimeError, ValueError):
            # 可观测系统不可用不能阻断研究主链路，也不能在异常中打印密钥。
            return cls()
        capture = os.getenv("LANGFUSE_CAPTURE_CONTENT", "0") in {
            "1",
            "true",
            "True",
        }
        return cls(client, capture_content=capture)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        as_type: str = "span",
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[ObservationHandle]:
        if self.client is None:
            yield ObservationHandle()
            return

        payload = input_data if self.capture_content else _summary(input_data)
        options: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
            "input": payload,
            "metadata": metadata or {},
        }
        if model:
            options["model"] = model
        with self.client.start_as_current_observation(**options) as observation:
            handle = ObservationHandle(observation, self.capture_content)
            try:
                yield handle
            except BaseException as error:
                observation.update(
                    level="ERROR",
                    status_message=type(error).__name__,
                )
                raise

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                pass
