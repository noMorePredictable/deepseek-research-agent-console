# DeepSeek Research Agent

这是一个基于 DeepSeek 和 AgentScope 2.0.8 的轻量研究项目，支持网页搜索、SQLite 记忆，以及由 Playwright MCP 驱动的 Browser Agent。
运行 `pip install -r requirements.txt`、`npm install` 并设置 `DEEPSEEK_API_KEY` 后，可用 `python research_agent.py --console` 启动研究终端。
使用 `python browser_agent.py` 启动浏览器 Agent，或运行 `python browser_agent.py --smoke` 执行无需 API Key 的端到端测试。
使用 `python a2a_team.py --allow-browser-actions` 启动 ResearchAgent 与 BrowserAgent 的本地 A2A 协作终端；`python a2a_team.py --smoke` 可验证完整协作链路。
先运行 `docker build -f sandbox/Dockerfile -t deepseek-research-agent-sandbox:latest .` 构建隔离镜像，再给 Browser Agent 或 A2A 命令加上 `--sandbox`，即可把 Playwright MCP 和 Chromium 放进临时 Docker 容器运行。
