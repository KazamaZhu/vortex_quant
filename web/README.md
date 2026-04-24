# Vortex Web 控制台（最小版）

与 `vortex/` 核心包分离：静态前端在 `web/frontend/`，API 在 `web/server/`。  
功能仅封装**已实现**的 CLI 能力（`init`、`data *`、`profile resolve/explain`），不修改业务逻辑。

## 依赖

```bash
uv sync --extra web
```

## 启动

仅监听本机（避免局域网暴露）：

```bash
uv run vortex-web
# 或
uv run python -m web.server.main
# 或
uv run uvicorn web.server.main:app --host 127.0.0.1 --port 8765
```

浏览器打开：<http://127.0.0.1:8765/>

API 文档：<http://127.0.0.1:8765/docs>

## 行为说明

- **工作区路径**：与 CLI 一致，通过页面里的「根目录」或各 API 的 `root` 参数传入；数据仍在 `{root}/data/`。
- **Token**：页面只显示「是否已配置」及掩码前缀，完整 Token 不会出现在 API 响应中（环境变量或 `{root}/.env`）。
- **前台 / 后台**：与 `vortex data` 相同；后台任务走既有 `_submit_data_background_task`。前台任务通过**子进程**执行 `python -m vortex data ...`，避免 `sys.exit` 结束 Web 进程。

## 隐私说明

助手对话不会自动读取你电脑上的路径或密钥；本控制台也只在浏览器请求时、由你填写的 `root` 访问该目录。
