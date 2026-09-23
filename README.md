# DeepSeek Research Agent

这是一个基于 DeepSeek 和 AgentScope 2.0.8 的轻量资料研究 Agent，支持网页搜索、正文读取、SQLite 长期记忆和来源编号。
运行 `pip install -r requirements.txt` 并设置 `DEEPSEEK_API_KEY` 后，可用 `python research_agent.py --console` 启动带流式事件与工具确认的终端 UI。
也可以使用 `python research_agent.py "研究问题"` 生成单次 Markdown 研究报告。
