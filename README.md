# DeepSeek Research Agent

这是一个基于 DeepSeek 和 AgentScope 2.0.8 的轻量研究 Agent。项目保留简单清晰的 ReAct 主链路，同时加入网页分层路由、A2A BrowserAgent、MCP 沙箱、Agent Harness、上下文压缩、混合长期记忆和可选 Langfuse Trace。

## 核心能力

- `research_agent.py`：手写 Agent Loop、Function Calling、DDGS 搜索、HTTP 静态页解析和来源管理。
- `browser_agent.py`：通过 Playwright MCP 操作需要 JavaScript、点击、筛选或翻页的动态网页。
- `a2a_team.py`：ResearchAgent 通过 Agent Card、JSON-RPC 和 AgentExecutor 委派 BrowserAgent。
- `research_harness.py`：滑动窗口 LoopDetector、工具超时、三态熔断和收敛提示注入。
- `context_compaction.py`：手写 Loop 压缩旧工具正文；AgentScope Console 使用原生 `ContextConfig` 摘要压缩。
- `research_memory.py`：SQLite 记忆 CRUD，默认 BM25；配置 Embedding 后使用 BM25 + 向量 RRF 混合召回。
- `agent_observability.py`：按 Agent → Generation → Tool 建立 Langfuse 父子 Trace，默认不上传正文。

## 安装与运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
$env:DEEPSEEK_API_KEY="你的密钥"
python research_agent.py --console
```

单次研究：

```powershell
python research_agent.py "研究 Agent Harness 的常见实现"
```

浏览器与 A2A：

```powershell
python browser_agent.py
python browser_agent.py --smoke
python a2a_team.py --allow-browser-actions
python a2a_team.py --smoke
```

## 混合长期记忆

不配置 Embedding 时，记忆使用 BM25 全文召回。要启用语义向量，在环境变量中设置一个 OpenAI 兼容的 Embedding 服务：

```powershell
$env:EMBEDDING_BASE_URL="https://你的服务/v1"
$env:EMBEDDING_API_KEY="你的密钥"
$env:EMBEDDING_MODEL="你的向量模型"
python research_agent.py --memory-reindex
```

记忆管理命令：

```powershell
python research_agent.py --memory-list
python research_agent.py --memory-delete 12
python research_agent.py --memory-reindex 100
```

## Langfuse 可观测

设置 `LANGFUSE_BASE_URL`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 后自动启用。默认的 `LANGFUSE_CAPTURE_CONTENT=0` 只记录字符数、哈希、工具名、耗时和状态；只有你明确改成 `1` 才会记录问题、网页内容和输出正文。

## Docker 浏览器沙箱

```powershell
docker build -f sandbox/Dockerfile -t deepseek-research-agent-sandbox:latest .
python browser_agent.py --sandbox
python a2a_team.py --sandbox --allow-browser-actions
```

`host_workdir=None` 时项目目录不会直接挂载进浏览器容器，Playwright MCP 与 Chromium 在临时 Workspace 中运行。

## 测试

```powershell
python -m unittest discover -s tests -v
python browser_agent.py --smoke
python a2a_team.py --smoke
```
