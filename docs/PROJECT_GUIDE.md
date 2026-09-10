# finrag 项目理解地图 —— 学习指南

> 面向"从零理解本项目"的完整导读：先看全景，再按主题深入，最后用实验检验。
> 全文基于当前代码（Python 3.14 / LangGraph 1.x / langchain-core 1.x）。
>
> **文档分工**：本文件管"怎么学会"；快速上手与实测数据看 [README.md](../README.md)；
> 深挖设计决策的问答归档在 [思考与方案/](思考与方案/)；评估术语速查见
> [概念与术语.md](概念与术语.md)；全部文档入口见 [docs/README.md](README.md)；
> 开发约定见 [AGENTS.md](../AGENTS.md)。

---

# 一、一句话定位

> 一个金融问答 Agent：**手写 LangGraph StateGraph 实现 ReAct 循环**（不用任何高级封装），
> 叠加 **RAG 混合检索管道**与 **Tool Calling**，配套**质量评估实验**与**并发压测**
> ——每一层都能单独拿出来讲工程原理。

技术栈速览：

| 维度 | 选型 |
|---|---|
| 运行时 | Python 3.14（注解默认惰性求值，无需 future import）/ uv 包管理 |
| 编排 | LangGraph `StateGraph`（手写节点 + 条件边，不用 AgentExecutor） |
| LLM | DeepSeek（OpenAI 兼容接口），Embedding = 阿里云 DashScope text-embedding-v3 |
| 向量库 | Chroma（本地持久化到 `chroma_db/`） |
| 质量工具 | ruff（风格）+ pyrefly（类型检查，`uv run pyrefly check` 应为 0 diagnostics） |

---

# 二、全景：一次提问的生命周期（最重要的一张图）

> 节点结构（Mermaid 版）见 [README.md 第二节](../README.md)；本节侧重"一次提问的完整时序"，
> 两者互补不重复。

```
用户输入 "计算600998持仓盈亏，顺便查研报"
        │
        ▼
┌─ main.py: stream_answer() ──── graph.astream(stream_mode="messages") 逐token输出
│
│   ┌─────────────────────── LangGraph 图（agent_graph.py）───────────────────────┐
│   │                                                                              │
│   │  START ──► retrieve ──► agent ──┬─(有tool_calls)──► tools ──┐                │
│   │             │            │      │                           │ 回环=ReAct循环 │
│   │             │            │      └─(无tool_calls)──► generate ──► END        │
│   │             ▼            ▼              ▲                            │
│   │     rag_retriever    SystemPrompt      asyncio.gather                 │
│   │     .retrieve()      拼RAG上下文       并行执行工具                    │
│   │     dense+BM25       bind_tools        ToolMessage回传               │
│   │     +RRF+LLM重排                                                    │
│   └──────────────────────────────────────────────────────────────────────┘
```

**核心洞察**：

- `retrieve` 每轮都重新执行——新问题重新检索，上下文永远新鲜
- `agent ⇄ tools` 构成回环——循环终止条件是 **LLM 不再发起 tool_calls**，这就是
  ReAct（Reasoning + Acting）
- `generate` 是收尾节点——把"中间态"（带工具调用痕迹的对话）整理成面向用户的最终答案

> 注：当前 `main.py` 的 `main()` 处于调试状态（固定跑并发 demo，交互模式被注释），
> 想恢复交互模式需手动解开注释。

---

# 三、分层模块地图（按职责）

> 完整文件树见 [README.md 第二节](../README.md)「目录结构」。

| 层 | 文件 | 体量 | 职责 | Java 类比 |
|---|---|---|---|---|
| 入口 | `main.py` | 小 | 流式 CLI + Semaphore 并发压测 demo | Controller + 压测脚本 |
| **编排** | `agent_graph.py` | 中 | 四节点 + 条件边组装图，ReAct 循环 | Activiti 流程定义 |
| **编排** | `state.py` | 小 | AgentState 契约（pydantic BaseModel）+ reducer 合并策略 | 强类型流程上下文对象 |
| 能力 | `tools.py` | 小 | 进程内 3 个 @tool（天气/股票/读文档） | Service + DTO 校验 |
| 能力 | `rag_retriever.py` | 大 | 建索引 → 多路召回 → RRF → 可选 LLM 重排 | ES 检索服务 |
| 扩展 | `mcp_client.py` | 中 | 手写最小 MCP Client：远程工具发现/调用 → 转 LangChain 工具；文件头含 Java 版 Server 示例 | Feign 客户端 + 协议适配层 |
| 质量 | `eval/metrics.py` + `eval/compare.py` + `eval/faithfulness.py` | 中 | Hit@k/MRR@k 双口径指标 + 三路检索策略对比 + LLM-as-Judge 忠实度（原子断言级幻觉检测） | 黄金集回归 + QA 自动评审 |
| 基建 | `config.py` | 小 | .env 配置中心 + get_llm/get_embeddings 工厂 | @Configuration + @Bean |
| 基建 | `trace.py` | 小 | @trace 装饰器打点耗时 | AOP 切面 / Span |
| 运维 | `scripts/*.sh` | — | check/chat/demo/eval 快捷命令 | Makefile |

> 注："体量"用相对大小而非行数——行数会随开发漂移，相对规模更耐放。

---

# 四、五个核心学习主题（由浅入深）

## 主题 1：State 与 reducer —— 数据怎么在节点间流动

读 `state.py`（小而关键）。注意：AgentState 是 **pydantic BaseModel 而非 TypedDict**（规避 langgraph 泛型边界的类型误报），节点内一律属性访问。两个设计：

- `messages: Annotated[list[AnyMessage], add_messages]` —— 带 **reducer**：
  节点返回**增量**，框架自动追加合并（多轮记忆的根基）
- `retrieval_docs: list[str]` —— **故意不带** reducer：
  每轮整体覆盖，防止上一轮的检索结果"串味"

> 思考题：为什么 messages 要增量而 docs 要覆盖？
> （答案就在代码注释里：对话历史要累积，检索上下文要新鲜。）

## 主题 2：ReAct 循环与条件边 —— 图的"心跳"

读 `agent_graph.py` 的三个关键函数：

```python
def route_after_agent(state) -> Literal["tools", "generate"]:
    # 看最后一条 AIMessage 有没有 tool_calls → 决定走哪条边
```

- **条件边** = 排他网关：`add_conditional_edges("agent", route_after_agent, {...})`
- **回环边**：`add_edge("tools", "agent")` —— 工具结果回传后 LLM 继续思考，
  直到不再发 tool_calls 才去 `generate` 收尾
- `MemorySaver` checkpointer 按 `thread_id` 存快照 → 多轮对话记忆的来源

> 思考题：为什么有了 tools 回环还需要单独的 generate 节点？
> （带 tool_calls 的 AIMessage 是"中间态"，generate 做一次无工具的收尾输出。）

## 主题 3：RAG 管道 —— 从"向量库=全部"到"多路召回+融合"

读 `rag_retriever.py`，按这个顺序理解 `retrieve()`：

| 步骤 | 代码位置 | 原理 |
|---|---|---|
| 建索引 | `ensure_index()` | Chroma 持久化；库已有数据则跳过构建 |
| Dense 召回 | `similarity_search_with_score` | 语义相似，懂"营收≈营业收入" |
| BM25 召回 | `_bm25.get_scores(_tokenize(query))` | 关键词精确命中，懂"600998"这种专名 |
| RRF 融合 | `score += 1/(60+rank)` | 只看排名不看分数，消除两路量纲差异 |
| LLM 重排 | `_llm_rerank()` | 精排，失败自动降级保原序（熔断思想）；小语料实测负优化，详见 README 第六节 |

配套：`retrieve_dense()` 是纯向量基线（供评估实验做对照）；
中文分词用 `_tokenize()` 的 CJK bigram 近似（可换 jieba）。

> 动手实验：`./scripts/eval.sh --sample 8` 分别跑 dense vs hybrid，
> 亲眼看到 Hit@k/MRR 差距——这就是你理解每层增益的证据。

## 主题 4：Tool Calling —— 让 LLM"长出手"

读 `tools.py`：

- pydantic Schema 就是校验器（`price: float = Field(gt=0)`），
  非法参数进不了函数体（对应 Java 的 JSR-303 + @ExceptionHandler）
- 错误降级双保险：pydantic 拦截 + 函数内 try/except，
  **任何失败都变成友好 ToolMessage 回传给 LLM**，而不是炸掉链路
  ——这是本项目的核心设计约束：工具失败绝不抛异常
- `asyncio.gather` 并行执行多个 tool_calls（≈ CompletableFuture.allOf）

## 主题 5：并发与可观测性 —— Agent 的工程外衣

- **限流**：`main.py` 的 demo 用 `asyncio.Semaphore(MAX_CONCURRENCY)` 控制并发请求数，
  日志里能看到"获得令牌/活跃数"的排队过程（≈ Java 线程池 + 信号量）
- **打点**：`trace.py` 的 `@trace("node.agent")` 同时包装 sync/async 函数
  （运行时用 `inspect.iscoroutinefunction` 分派），每个节点输出耗时 → P99 复盘的数据来源
- **惰性单例**：`config.get_llm()` / `get_retriever()` 首次调用才创建（省启动开销），
  类型标注遵循"返回具体类型 + 内部 assert 收窄"
- **路径安全**：`config.py` 把 DATA_DIR / CHROMA_PERSIST_DIR / .env 全部锚定到项目根，
  从任何 CWD 启动都正确

---

# 五、推荐学习路径（照这个顺序读代码）

| 步骤 | 做什么 | 目的 |
|---|---|---|
| 1 | `./scripts/chat.sh` 问一个问题，盯着终端日志 | 建立"节点顺序"直觉：看到 `[TRACE] node.retrieve → node.agent → node.tools → ...` |
| 2 | 读 `state.py`(40行) → `agent_graph.py` 的 `build_graph()` + `route_after_agent()` | 掌握图骨架，这是全项目的"目录" |
| 3 | 在四个节点函数里各加一行 `print(state.keys())` 再跑一次 | 亲眼看到 state 在节点间如何累积 |
| 4 | 读 `rag_retriever.retrieve()`，**手算一遍 RRF 公式**（拿两路各 3 个结果） | 彻底理解排名融合 |
| 5 | 故意问 "帮我算 600998 以 abc 元持有 -5 股的盈亏" | 观察 pydantic 拦截 → 友好降级 → LLM 解释错误 |
| 6 | `./scripts/check.sh` + `./scripts/eval.sh --sample 8 --skip-rerank` | 体验质量门禁与量化评估 |
| 7 | 读 `trace.py`、`config.py`、`main.py` 的 demo 部分 | 补齐装饰器/惰性单例/并发的工程细节 |

---

# 六、动手改造实验（检验是否真懂）

1. **改路由**：让 `route_after_agent` 永远返回 `"generate"`（跳过工具）
   → 观察天气/计算类问题的回答退化
2. **加字段**：给 `AgentState` 加一个 `retrieval_count: int`，
   在 retrieve 节点写入、agent 节点读出
3. **写新工具**：仿照 `fetch_weather` 写一个 `convert_currency(rate, amount)`，
   注册进 `ALL_TOOLS`
4. **量化实验**：`.env` 里 `ENABLE_RE_RANK=false` vs `true` 各跑一遍 eval
   → 亲眼验证"小语料下重排是负优化"（本项目实测答案 Hit@3 100%→94.3%）
5. **压测观察**：`MAX_CONCURRENCY=2` 跑 `./scripts/demo.sh` → 看排队日志

做完这 5 个实验，你就不是"读过"这个项目，而是"拥有"它了。

---

# 七、Java → Python 对照

核心对照表统一维护在 [README.md 第三节](../README.md)「三、核心机制说明（Java → Python 对照）」，
本节不再重复，避免两处内容漂移。

---

# 八、常用命令

> 完整的环境准备与密钥配置见 [README.md 第一节](../README.md)。

```bash
./scripts/check.sh                      # 质量门禁：ruff + pyrefly（应为双绿）
uv run python main.py                   # 交互模式（当前被调试态覆盖，需解开注释）
./scripts/chat.sh                       # 同上（脚本封装）
./scripts/demo.sh                       # 并发限流演示（8 请求，上限 5）
./scripts/eval.sh --skip-rerank         # 秒级评估：dense vs hybrid
./scripts/eval.sh --sample 8            # 抽样评估
uv run python eval/compare.py           # 全量 35 条评估（约 5 分钟，调 LLM 花钱）
```

---

**一句话总结**：先跑起来看现象（步骤1）→ 抓住图骨架（步骤2-3）→
深入两条能力线（RAG 管道、工具循环）→ 用 eval 和压测闭环验证。
卡在哪一步随时回来翻这份地图。

学完本地图后，去 [思考与方案/](思考与方案/) 把每个设计决策的"为什么"过一遍
（检索质量账本、workflow vs agent 判据、MCP 协议选型……）——
那是面试答辩的直接弹药库。
