"""研究 Agent 的轻量 Harness：循环检测、工具超时与熔断。

这些能力放在业务工具外层，不改变搜索、网页读取或记忆本身。Harness 发现模型
重复调用同一工具时只注入收敛提示，让主 Agent 自己换策略或结束回答；工具连续
失败时则短暂熔断，避免把同一种外部故障重复灌入上下文。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable


CONVERGENCE_HINT = (
    "检测到工具 {tool} 使用相同参数连续得到相同结果 {count} 次，任务没有新进展。"
    "请不要原样重试：改用不同检索词或其他工具；若现有证据已经足够就直接作答，"
    "否则明确说明缺少什么信息。"
)


def _fingerprint(value: Any) -> str:
    """对结构化值做稳定哈希，Detector 中不保留可能含网页正文的原文。"""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            pass
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class LoopDetector:
    """使用滑动窗口识别同工具、同参数、同结果的无进展循环。"""

    repeat_threshold: int = 3
    window: int = 8
    _history: deque[tuple[str, str, str, str]] = field(init=False)

    def __post_init__(self) -> None:
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold 至少为 2")
        if self.window < self.repeat_threshold:
            raise ValueError("window 不能小于 repeat_threshold")
        self._history = deque(maxlen=self.window)

    def observe(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        state: str,
    ) -> str | None:
        """记录一次已完成调用；结果变化、参数变化或状态变化都视为有进展。"""
        item = (
            tool_name,
            _fingerprint(arguments),
            _fingerprint(result),
            state,
        )
        self._history.append(item)

        count = 0
        for previous in reversed(self._history):
            if previous[0] != item[0] or previous[1] != item[1]:
                continue
            if previous[2:] != item[2:]:
                break
            count += 1
        if count >= self.repeat_threshold:
            return CONVERGENCE_HINT.format(
                tool=tool_name,
                count=count,
            )
        return None

    def reset(self) -> None:
        self._history.clear()


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None
    half_open_probe: bool = False


@dataclass
class CircuitBreaker:
    """按工具名维护 closed/open/half-open 三态熔断器。"""

    failure_threshold: int = 3
    reset_seconds: float = 30.0
    _states: dict[str, _CircuitState] = field(default_factory=dict)

    def _state(self, tool_name: str) -> _CircuitState:
        return self._states.setdefault(tool_name, _CircuitState())

    def status(self, tool_name: str) -> str:
        state = self._state(tool_name)
        if state.opened_at is None:
            return "closed"
        return "half_open" if state.half_open_probe else "open"

    def allow(self, tool_name: str, *, now: float | None = None) -> bool:
        state = self._state(tool_name)
        if state.opened_at is None:
            return True
        current = time.monotonic() if now is None else now
        if current - state.opened_at < self.reset_seconds:
            return False
        if state.half_open_probe:
            return False
        state.half_open_probe = True
        return True

    def record_success(self, tool_name: str) -> None:
        self._states[tool_name] = _CircuitState()

    def record_failure(self, tool_name: str, *, now: float | None = None) -> None:
        state = self._state(tool_name)
        current = time.monotonic() if now is None else now
        if state.half_open_probe:
            state.opened_at = current
            state.half_open_probe = False
            return
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failure_threshold:
            state.opened_at = current


@dataclass(frozen=True)
class ToolExecution:
    """一次工具执行的结构化结果，方便主循环注入提示和记录 Trace。"""

    tool_name: str
    result: Any
    state: str
    elapsed_ms: int
    circuit: str
    convergence_hint: str | None = None

    def model_payload(self) -> dict[str, Any]:
        if isinstance(self.result, dict):
            payload = dict(self.result)
        else:
            payload = {"result": self.result}
        payload["_runtime"] = {
            "state": self.state,
            "elapsed_ms": self.elapsed_ms,
            "circuit": self.circuit,
        }
        if self.convergence_hint:
            payload["_harness_hint"] = self.convergence_hint
        return payload

    def model_text(self) -> str:
        return json.dumps(self.model_payload(), ensure_ascii=False, default=str)


class ToolHarness:
    """统一执行工具，并提供超时、熔断与循环检测。"""

    def __init__(
        self,
        *,
        timeouts: dict[str, float] | None = None,
        loop_detector: LoopDetector | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.timeouts = timeouts or {
            "search_web": 25.0,
            "read_page": 25.0,
            "recall_memory": 10.0,
            "list_memories": 10.0,
            "save_memory": 15.0,
            "delete_memory": 10.0,
        }
        self.loop_detector = loop_detector or LoopDetector()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="research-tool",
        )

    def reset_loop(self) -> None:
        self.loop_detector.reset()

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        function: Callable[..., Any],
    ) -> ToolExecution:
        started = time.monotonic()
        if not self.circuit_breaker.allow(tool_name):
            return ToolExecution(
                tool_name=tool_name,
                result={
                    "error": "工具连续失败，熔断器暂时打开，请换用其他工具或稍后重试。",
                },
                state="circuit_open",
                elapsed_ms=0,
                circuit="open",
            )

        timeout = self.timeouts.get(tool_name, 30.0)
        future = self._executor.submit(function, **arguments)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            self.circuit_breaker.record_failure(tool_name)
            state = "timeout"
            result = {"error": f"工具执行超过 {timeout:g} 秒，已按超时处理。"}
        except Exception as error:  # 工具异常必须变成 observation，不能打穿 Agent Loop。
            self.circuit_breaker.record_failure(tool_name)
            state = "error"
            result = {
                "error": str(error)[:500],
                "error_type": type(error).__name__,
            }
        else:
            self.circuit_breaker.record_success(tool_name)
            state = "success"

        elapsed_ms = round((time.monotonic() - started) * 1000)
        hint = self.loop_detector.observe(tool_name, arguments, result, state)
        return ToolExecution(
            tool_name=tool_name,
            result=result,
            state=state,
            elapsed_ms=elapsed_ms,
            circuit=self.circuit_breaker.status(tool_name),
            convergence_hint=hint,
        )

    def close(self) -> None:
        # wait=False 避免已经超时的外部请求在退出时继续阻塞主线程。
        self._executor.shutdown(wait=False, cancel_futures=True)
