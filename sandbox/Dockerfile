FROM mcr.microsoft.com/playwright:v1.63.0-noble

# AgentScope DockerWorkspace 的网关需要 Python；浏览器及其系统依赖只在镜像中安装。
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/playwright-mcp
COPY package.json package-lock.json ./
RUN npm ci \
    && npx playwright install chromium \
    && npm cache clean --force

WORKDIR /workspace
