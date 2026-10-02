# DeepSeek Research Agent

这是一个基于 DeepSeek 和 AgentScope 2.0.8 的轻量研究项目，支持网页搜索、SQLite 记忆，以及由 Playwright MCP 驱动的 Browser Agent。
运行 `pip install -r requirements.txt`、`npm install` 并设置 `DEEPSEEK_API_KEY` 后，可用 `python research_agent.py --console` 启动研究终端。
使用 `python browser_agent.py` 启动浏览器 Agent，或运行 `python browser_agent.py --smoke` 执行无需 API Key 的端到端测试。
