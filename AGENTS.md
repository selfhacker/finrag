# AGENTS.md — finrag 开发指南

金融 RAG + Tool Calling 复合 Agent（LangGraph 手写 ReAct 循环），面向面试的教学型项目。
技术栈：Python ≥3.14 / uv / LangGraph / LangChain / Chroma / pydantic v2。
LLM=DeepSeek，Embedding=阿里云 DashScope（均为 OpenAI 兼容接口）。

## 文档体系分工

| 文档 | 职责 |
|---|---|
| `README.md` | 项目门面：快速上手、架构总览、压测与评估数据 |
| `docs/README.md` | docs 文档中心：全文档导航 + 思考与方案归档规范 |
| `docs/PROJECT_GUIDE.md` | 学习地图：模块分层 → 核心主题 → 动手实验 |
| `docs/思考与方案/` | 面试答辩归档：按主题分文件记录工程决策问答 |
| `docs/概念与术语.md` | 评估术语速查词汇表（Hit@k/MRR、混淆矩阵、ground truth、faithfulness 等） |
| `AGENTS.md`（本文件） | 开发约定与陷阱，供 AI 助手/协作者 |

改动代码时若影响对外描述（新模块、行为变化、数据更新），须同步上述文档，勿只改一处。

往 `docs/思考与方案/` 归档新条目时须同步：文件头主题范围、关联知识链、`docs/README.md` 的目录导航、条目间双向关联；引用代码用"函数名 + 行为"定位，不写行号（易漂移）。

## 常用命令

```bash
uv sync                        # 安装依赖（必须用 uv，不要用 pip）
./scripts/check.sh             # 质量门禁 = ruff check main.py src eval + pyrefly check
uv run python main.py --demo   # 并发压测冒烟（8 请求、信号量上限 5，会真实调用 LLM API）
uv run python -c "import finrag.agent_graph"   # 免 API 的快速导入校验
./scripts/eval.sh              # 检索评估全量 35 条（约 5 分钟，调 LLM 花钱）
./scripts/eval.sh --skip-rerank    # 秒级，仅 dense vs hybrid，不调 LLM
./scripts/eval.sh --sample 8       # 抽样快跑
```

- **无测试框架**（无 tests/、无 pytest）。改动后的验证顺序：`check.sh` → 导入校验或 `--demo` 冒烟 → 涉及检索逻辑时跑 golden set。
- 注意：`main.py` 的 `main()` 当前被调试状态固定为无条件执行 `concurrency_demo()`，交互模式被注释掉。要恢复交互模式需手动解开注释。

## 架构速记

- 入口 `main.py` → 使用 `src/finrag/agent_graph.py` 的模块级单例 `graph`。
- 四节点手写 StateGraph：`retrieve → agent ⇄ tools → generate`（ReAct 回环，条件边 `route_after_agent` 判断 `tool_calls`）。刻意不用 `AgentExecutor`/`create_react_agent` 等预封装。
- `state.py` 的 `AgentState` 是 **pydantic BaseModel（非 TypedDict）**：
  - 节点内一律属性访问（`state.messages`），dict 下标会报错；
  - 不要改回 TypedDict——langgraph 的 `StateT` 泛型边界会让 PyCharm/pyrefly 误报协议不匹配；
  - 字段必须有默认值（图入口只传 messages）。
- `tools.py`：`@tool` + pydantic `args_schema`。**工具失败绝不抛异常**，一律返回友好错误文本转成 ToolMessage——抛异常会炸掉整条 LangGraph 链路（项目核心设计约束）。
  - **工具 docstring 即发给 LLM 的接口文档**：`@tool` 会把 docstring 全文透传为请求体的 `description`。必须写"使用场景（何时调用）+ 参数口径 + 返回内容"；教学类比/实现细节只放 `#` 代码注释或 Input 类 docstring（pydantic 类 docstring 不进 JSON Schema），不得污染工具 docstring。
- `mcp_client.py`：手写最小 MCP Client（JSON-RPC 2.0 over Streamable HTTP，零新依赖，复用 langchain-openai 的传递依赖 httpx）。教学定位与"刻意不用预封装"同款——看清协议本质；生产替代是 `langchain-mcp-adapters`。
  - 错误策略阶段化：`list_tools` 属装配阶段，可抛 `McpError` fail-fast；`call_tool` 属运行期，沿用工具铁律（绝不抛异常、降级为错误文本）。
  - `to_langchain_tools()` 把远程工具转成 `StructuredTool` 接入主链路；Java 版 Server 写法示例在文件头注释。
  - 端点由 `.env` 的 `MCP_SERVER_URL` 配置（空则 demo 跳过）。
- `config.py` 把所有相对路径锚定到项目根，从任意 CWD 启动均安全；全部参数走 `.env`。
- RAG 链路在 `rag_retriever.py`：加载→分块→Chroma 持久化→Dense+BM25+RRF 混合检索→可选 LLM 重排。实测小语料下重排是负优化（见 README 第六节），由 `ENABLE_RE_RANK` 控制。
- `eval/`：L1 检索评估（`metrics.py` Hit@k/MRR@k 双口径 + `compare.py` 三路对比，确定性指标）+ L2 生成评估（`faithfulness.py` LLM-as-Judge 原子断言级忠实度）。
  - 判官模型走 `LLM_JUDGE_MODEL`（默认 deepseek-chat 非思考版）：思考模型不支持强制 tool_choice，会使 `with_structured_output` 的 function_calling 方式报 400。
  - DeepSeek 不支持 response_format(json_schema)，结构化输出必须 `method="function_calling"`。
  - judge 失败降级为 score=-1 不抛异常（同工具铁律）。

## 环境与产物

- `.env` 需两个密钥：`LLM_API_KEY`（DeepSeek）、`DASHSCOPE_API_KEY`（Embedding）。模板占位符 `sk-your-...` 会被主动拦截报错。`.env` 已 gitignore，勿提交密钥。
- `chroma_db/` 是向量索引持久化产物，可安全删除重建（首次启动自动构建，需 Embedding API）；`eval/chroma_db/` 是评估用的独立索引。
- 多轮记忆靠 MemorySaver 按 `thread_id` 隔离，仅内存态。

## 类型检查已知陷阱

- **新增/修改的代码必须对类型检查器友好**：目标是 pyrefly 零诊断（0 diagnostics）且 PyCharm 无警告。优先用标准写法从根源消除（如局部变量中转收窄 global、Pydantic state 替代 TypedDict、逐文件构造替代标注有缺陷的 loader_cls），`# pyrefly: ignore[...]` 只是最后手段，且必须带原因说明。
- PDF 加载刻意不走 `DirectoryLoader`（其 `loader_cls` 标注未含 PyPDFLoader，属 langchain_community 标注缺陷），而是逐文件直接构造 `PyPDFLoader`——不要"顺手"改回 DirectoryLoader，否则 pyrefly 报 bad-argument-type。
- `# pyrefly: ignore[...]` 只对 pyrefly 生效，PyCharm 不识别（它认 `# noqa` / `# type: ignore` / `# noinspection`）；反之 PyCharm 对 global 变量收窄弱、对 TypedDict 协议匹配误报多，修 PyCharm 警告优先用局部变量中转等标准写法，而不是堆 ignore。

## 代码风格（项目特有）

- 所有注释/docstring 用简体中文，并带「Java 工程师类比」讲解（如 Semaphore≈j.u.c.Semaphore、gather≈allOf）——这是项目的刻意教学风格，新代码须延续。
- 异步规范：全链路 async/await；并行用 `asyncio.gather`；限流用 `asyncio.Semaphore`；临界区内不得有 await。
