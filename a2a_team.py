"""ResearchAgent 与 BrowserAgent 的本地 A2A 协作入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import httpx
import uvicorn
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
)
from a2a.utils.constants import TransportProtocol
from a2a.utils.errors import UnsupportedOperationError
from agentscope.agent import A2AAgent, Agent
from agentscope.console import launch_console
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, ToolResultBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import FunctionTool, ToolResponse
from starlette.applications import Starlette

from browser_agent import (
    MCP_TOOL_PREFIX,
    BrowserSmokeModel,
    browser_agent_session,
)
from research_agent import build_console_agent


A2A_HOST = "127.0.0.1"
A2A_PORT = 8765
A2A_URL = f"http://{A2A_HOST}:{A2A_PORT}"
A2A_RPC_PATH = "/a2a/browser"


class BrowserAgentExecutor(AgentExecutor):
    """把现有 BrowserAgent 适配为 A2A Server 的执行器。"""

    def __init__(self, browser_agent: Agent) -> None:
        self.browser_agent = browser_agent
        self._lock = asyncio.Lock()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        async with self._lock:
            result = await self.browser_agent.reply(
                UserMsg(name="Researcher", content=context.get_user_input()),
            )

        await event_queue.enqueue_event(
            Message(
                message_id=str(uuid4()),
                context_id=context.context_id or str(uuid4()),
                role=Role.ROLE_AGENT,
                parts=[Part(text=result.get_text_content())],
            ),
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        del context, event_queue
        raise UnsupportedOperationError()


def build_browser_a2a_app(browser_agent: Agent) -> Starlette:
    """创建 BrowserAgent 的 Agent Card、JSON-RPC 路由和 A2A 服务。"""
    card = AgentCard(
        name="BrowserAgent",
        description="通过 Playwright MCP 浏览网页并返回带来源的研究结果。",
        supported_interfaces=[
            AgentInterface(
                url=A2A_URL + A2A_RPC_PATH,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version="1.0",
            ),
        ],
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="browser-research",
                name="Browser Research",
                description="打开网页、读取页面并整理可核查的来源。",
                tags=["browser", "research", "playwright"],
                examples=["打开官网并核实产品的最新功能"],
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=BrowserAgentExecutor(browser_agent),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, rpc_url=A2A_RPC_PATH),
    ]
    return Starlette(routes=routes)


@asynccontextmanager
async def run_a2a_server(app: Starlette) -> AsyncIterator[None]:
    """在当前进程后台启动并可靠关闭本地 A2A 服务。"""
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=A2A_HOST,
            port=A2A_PORT,
            log_level="warning",
        ),
    )
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("A2A 服务启动超时")
        yield
    finally:
        server.should_exit = True
        await server_task


async def connect_remote_browser() -> A2AAgent:
    """发现 BrowserAgent 的 Agent Card，并创建 AgentScope A2A 客户端。"""
    async with httpx.AsyncClient(timeout=10, trust_env=False) as http_client:
        card = await A2ACardResolver(http_client, A2A_URL).get_agent_card()
    a2a_http_client = httpx.AsyncClient(timeout=90, trust_env=False)
    client = ClientFactory(
        ClientConfig(
            streaming=False,
            polling=False,
            httpx_client=a2a_http_client,
            supported_protocol_bindings=[TransportProtocol.JSONRPC.value],
        ),
    ).create(card)
    return A2AAgent(agent_card=card, client=client)


def build_browser_delegate_tool(remote_browser: A2AAgent) -> FunctionTool:
    """将远程 BrowserAgent 包装为 ResearchAgent 可以选择的工具。"""

    async def delegate_browser_research(task: str) -> ToolResponse:
        """通过 A2A 把需要真实浏览器完成的研究任务交给 BrowserAgent。"""
        response = await remote_browser.reply(
            UserMsg(name="Researcher", content=task),
        )
        return ToolResponse(
            content=[TextBlock(text=response.get_text_content())],
            metadata={"protocol": "A2A", "agent": remote_browser.name},
        )

    return FunctionTool(
        delegate_browser_research,
        name="delegate_browser_research",
        description=(
            "通过 A2A 委派 BrowserAgent 操作真实浏览器。"
            "适合动态网页、需要点击或需要页面快照的研究任务。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "交给 BrowserAgent 的明确浏览任务",
                },
            },
            "required": ["task"],
        },
        is_read_only=True,
    )


class ResearchTeamSmokeModel(ChatModelBase):
    """无需 API Key，固定触发一次 ResearchAgent -> A2A -> BrowserAgent。"""

    class Parameters(ChatModelBase.Parameters):
        pass

    def __init__(self) -> None:
        super().__init__(
            credential=OpenAICredential(api_key="unused"),
            model="research-team-smoke",
            parameters=self.Parameters(),
            stream=False,
        )
        self.formatter = OpenAIChatFormatter()
        self._step = 0

    async def _call_api(
        self,
        model_name: str,
        messages: list,
        tools: list[dict] | None = None,
        tool_choice=None,
        **kwargs,
    ) -> ChatResponse:
        del model_name, messages, tools, tool_choice, kwargs
        if self._step == 0:
            content = [
                ToolCallBlock(
                    id="a2a-browser-1",
                    name="delegate_browser_research",
                    input=json.dumps(
                        {
                            "task": (
                                "加载 browser-research Skill，打开 "
                                "https://example.com 并读取页面快照。"
                            ),
                        },
                    ),
                ),
            ]
        else:
            content = [TextBlock(text="A2A team smoke test completed.")]
        self._step += 1
        return ChatResponse(content=content, is_last=True)


async def run_team(
    *,
    headless: bool,
    smoke: bool,
    allow_browser_actions: bool,
) -> None:
    """启动 BrowserAgent A2A 服务，并运行 ResearchAgent 协调端。"""
    if not smoke and not allow_browser_actions:
        raise SystemExit(
            "A2A BrowserAgent 没有独立的权限确认终端；"
            "请明确添加 --allow-browser-actions 后启动。",
        )

    browser_model = BrowserSmokeModel() if smoke else None
    async with browser_agent_session(
        headless=headless,
        model=browser_model,
        bypass_permissions=smoke or allow_browser_actions,
    ) as browser_agent:
        async with run_a2a_server(build_browser_a2a_app(browser_agent)):
            remote_browser = await connect_remote_browser()
            async with remote_browser:
                researcher = build_console_agent(
                    extra_tools=[build_browser_delegate_tool(remote_browser)],
                    model=ResearchTeamSmokeModel() if smoke else None,
                    system_prompt_suffix=(
                        "\n9. 遇到动态网页、点击交互或需要浏览器快照时，"
                        "调用 delegate_browser_research 委派 BrowserAgent；"
                        "收到结果后由你统一核实并汇总。"
                    ),
                    bypass_permissions=smoke,
                    api_key="unused" if smoke else None,
                )

                if smoke:
                    result = await researcher.reply(
                        UserMsg(name="user", content="验证两个 Agent 的 A2A 协作。"),
                    )
                    a2a_results = [
                        block
                        for msg in researcher.state.context
                        for block in msg.get_content_blocks()
                        if isinstance(block, ToolResultBlock)
                        and block.name == "delegate_browser_research"
                    ]
                    failed = [
                        block
                        for block in a2a_results
                        if str(getattr(block.state, "value", block.state))
                        == "error"
                    ]
                    if not a2a_results or failed:
                        raise RuntimeError("ResearchAgent 的 A2A 工具调用失败")
                    browser_results = [
                        block
                        for msg in browser_agent.state.context
                        for block in msg.get_content_blocks()
                        if isinstance(block, ToolResultBlock)
                        and block.name.startswith(MCP_TOOL_PREFIX)
                    ]
                    browser_failed = [
                        block
                        for block in browser_results
                        if str(getattr(block.state, "value", block.state))
                        == "error"
                    ]
                    if len(browser_results) < 2 or browser_failed:
                        raise RuntimeError("BrowserAgent 的 MCP 工具调用失败")
                    if result.get_text_content() != "A2A team smoke test completed.":
                        raise RuntimeError("ResearchAgent 没有完成 A2A 协作测试")
                    print("A2A_TEAM_SMOKE_OK")
                    print("flow=ResearchAgent->A2A->BrowserAgent->Playwright MCP")
                    return

                await launch_console(
                    researcher,
                    user_name="user",
                    verbosity="default",
                    max_tool_result_lines=30,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResearchAgent + BrowserAgent A2A 协作终端",
    )
    parser.add_argument("--headed", action="store_true", help="显示浏览器窗口")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="运行无需 API Key 的 A2A 端到端测试",
    )
    parser.add_argument(
        "--allow-browser-actions",
        action="store_true",
        help="允许远程 BrowserAgent 执行项目限定的 Playwright 工具",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_team(
            headless=not args.headed,
            smoke=args.smoke,
            allow_browser_actions=args.allow_browser_actions,
        ),
    )


if __name__ == "__main__":
    main()
