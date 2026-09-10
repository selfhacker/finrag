# Agent 架构 — 问题归档

> 主题范围：workflow 与 agent 辨析、ReAct 循环、asyncio 并发模型、function calling 链路、
> MCP 接口与协议选型、CoT/Plan-and-Execute/ReAct 模式对比、规划与思考、工具 docstring 规范。
> 对应实现：`src/finrag/agent_graph.py`。

---

## Q1 · agent_graph.py 算不算工作流？

**骨架是 workflow，心脏是 agent**——混合体（agentic workflow）。准确叫法："用 workflow 编排的 ReAct agent"。

### 判据只有一句话

> **下一步走哪条路，由谁在运行时决定？**
> - 开发者在代码里写死 → **Workflow**
> - 模型自己看着办 → **Agent**

### 逐段解剖本项目图

```
START ──→ retrieve ──→ agent ⇄ tools
                          │
                          └──→ generate ──→ END
```

| 部分 | 谁做主 | 定性 |
|---|---|---|
| `START → retrieve → agent` | **代码写死**：每次必先检索 | Workflow |
| `agent ⇄ tools` 回环 | **模型做主**：输出带 `tool_calls` 进 tools，不带去 generate；循环几圈也由模型定（`route_after_agent` 只按模型输出分支路由） | **Agent** |
| `agent → generate` | 条件边分支，路径预定义 | Workflow |

### 为什么这么设计（真正的加分点）

**retrieve 前置写死是刻意的确定性设计**：RAG 场景"必须先检索"是业务硬约束。交给纯 agent 自主决定要不要检索，模型可能跳过检索凭记忆瞎答。把确定性需求固化成 workflow 节点 = 用编排保证下限；需要灵活性的部分（调什么工具、调几次）留给模型 = 用 agent 抬高上限。

一句话：**能确定的部分不浪费模型的自主性，需要自主性的部分不用代码猜。**

### 参照系

| 形态 | 控制流 | 例子 |
|---|---|---|
| 纯 Workflow | 全部代码预定 | Dify/n8n 编排、LCEL 顺序链 |
| 纯 Agent | 全部模型决定 | `create_react_agent` / AgentExecutor（无强制检索） |
| 本项目 | 固定管道 + 嵌入式 ReAct 循环 | 手写 StateGraph 四节点 |

加分细节：项目刻意手写 `StateGraph` + 条件边而不用 `create_react_agent` 预封装，正是为了做出"固定 retrieve 前缀 + agent 回环"的混合结构——预封装纯 agent 给不了这个控制粒度。

---

## Q2 · ReAct 循环是什么

**ReAct = Reasoning + Acting**（2022 年论文 *ReAct: Synergizing Reasoning and Acting in Language Models*）：

> 让 LLM 交替地「想一步 → 做一步 → 看结果」，循环往复，直到它认为可以给出最终答案。

### 三拍节奏与代码对应

```
Thought（推理）    = agent 节点（LLM 思考，产出 AIMessage）
Action（行动）     = tools 节点（并行执行 tool_calls）
Observation(观察) = ToolMessage 回填消息历史 → 回到 agent 再想
不带 tool_calls    = Final Answer → generate 收尾
```

条件边路由函数 `route_after_agent` 就是循环节拍器：看这次有没有 `tool_calls`——有就再转一圈，没有就出循环。

### Java 类比

```java
// ReAct ≈ 由模型驱动退出条件的 while 循环
while (true) {
    AIMessage msg = llm.think(messages);          // Reason
    if (msg.toolCalls.isEmpty()) break;            // 模型说"够了"
    messages.addAll(execute(msg.toolCalls));       // Act + Observe
}
return llm.finalAnswer(messages);                  // Final Answer
```

与 Chain（固定管道、一次 prompt 出结果）的本质区别：Chain 是流水线，ReAct 是**带反馈回路的状态机**。

### 范式对照

| 范式 | 特点 | 局限 |
|---|---|---|
| CoT | 只推理不行动，全靠脑补 | 拿不到实时数据 |
| Plan-and-Execute | 先列完整计划再逐项执行 | 计划赶不上变化，中途难修正 |
| **ReAct** | 走一步看一步，按观察调整 | 循环次数不定，需防死循环 |

**工程边界**：项目退出条件完全交给模型（`tool_calls` 为空即出）；生产应叠加**硬性最大迭代次数**兜底，防止"调工具→失败→再调"的死循环烧钱。又一个 demo/工程分界点。

---

## Q3 · 并行执行 tool_calls 是怎么并行的？

核心一句话：**`asyncio.gather` 把多个工具协程同时交给事件循环调度——单线程内的"并发"，靠 IO 等待时间重叠实现，不是多线程真并行**。

### 逐行拆解（`tools_node` 的并行执行段）

```python
tool_messages: list[ToolMessage] = await asyncio.gather(
    *(_run_single_tool_call(tc) for tc in tool_calls)
)
```

1. **调用 async 函数 ≠ 执行它**：`_run_single_tool_call(tc)` 只是创建协程对象（拿到未执行的 Task），此刻什么都没跑；
2. **`gather(*)` = 集体提交事件循环**：所有协程注册后交替运行；
3. **`await` 是切换点**：工具内部都是 IO（`ainvoke` 调 API、读文件），执行到 `await` 时协程挂起让出控制权——事件循环立刻切去推进下一个协程。

### 时间线

```
串行:   工具A ████░░░░ 工具B ██████░░ 工具C ████░░░░   总耗时 = tA+tB+tC
               (等API)        (等API)       (等API)

gather: 工具A ████░░░░
        工具B ██████░░   ← 三段"等API"重叠     总耗时 ≈ max(tA,tB,tC)
        工具C ████░░░░
```

工具耗时 99% 在等网络 IO，CPU 几乎闲着——不需要多核，单线程重叠等待就赚翻了。

### Java 对照

| Python | Java 对应物 |
|---|---|
| `asyncio.gather(*coros)` | `CompletableFuture.allOf(f1, f2, ...)` |
| 事件循环 | Netty `EventLoop`（单线程非阻塞） |
| `await` 挂起 | 非阻塞 IO 注册回调后返回，线程不阻塞 |
| 协程 | 虚拟线程（Java 21）/ Reactive Stream |

区别：Java 传统方案每个请求占一个池化线程傻等；Python asyncio 零线程，挂起协程只占内存不占线程。

### 易答错的追问

- **是并行吗？** 严格说是**并发（concurrency）非并行（parallelism）**：同一时刻只有一个协程在 CPU 上跑，只是 IO 等待被重叠。
- **结果会乱序吗？** 不会。`gather` 保证返回顺序与输入协程顺序一致（完成有先有后，结果按位归位），ToolMessage 与 tool_call_id 对应关系不受影响。
- **某工具卡死怎么办？** 会拖住整个 gather。生产配 `asyncio.wait_for(timeout)` 或信号量限流（main.py demo 的 `Semaphore(5)` 同款思路）。

---

## Q4 · 多个协程可以跑在多个 CPU 上吗？

**默认不能**。协程生存在一个事件循环 = 一个线程里，一个线程同一时刻只被调度到一个核。

### 两层原因

1. **架构层**：协程是**用户态调度**——切换逻辑写在事件循环代码里，不经操作系统内核调度器，自然不会分散到多核；
2. **GIL 层**：CPython 全局解释器锁保证同一进程内同一时刻只有一个线程执行 Python 字节码——开 8 个线程跑纯计算也只有一个核干活。

### 想用多核的三条路

| 方案 | 做法 | 适用 |
|---|---|---|
| **多进程部署** | uvicorn/gunicorn `--workers N`，每个 worker 独立进程+独立事件循环 | Web 服务标准做法（≈ 应用多实例横向扩） |
| **进程池下沉** | `loop.run_in_executor(ProcessPoolExecutor(), fn)` | 个别重计算步骤，其余仍 async |
| **free-threading** | Python 3.14 官方 no-GIL 构建（PEP 703，本项目正好 ≥3.14） | 实验性，需专用解释器构建+生态验证，生产慎入 |

Java 类比：GIL ≈ 进程级一把超大 `synchronized`。

### 为什么本项目不需要

判据是瓶颈类型：工具全是 **IO 密集**（调 LLM API、读文件）——单循环已把等待全部重叠（Q3 收益拿满），没有"算"的任务可分给其他核。只有出现 **CPU 密集**工具（本地 embedding 计算、解析超大 PDF、复杂量化回测）才值得引进程池。

面试一句话：**协程解决"等待太多"，进程解决"算得太慢"；IO 密集用并发，CPU 密集用并行——先分清再选型。**

---

## Q5 · Function calling 全链路——项目是怎么实现工具调用的？

**工具调用是本项目的核心机制**（项目定位就是"RAG + Tool Calling 复合 Agent"），全链路四步如下。术语先对齐：**function calling = tool calling = tool use** 同义——OpenAI 2023 年最早叫 function calling，后官方更名 tool calling，Anthropic 叫 tool use。

### 四步链路（全部有对应代码）

```python
# 1️⃣ 定义 tools.py：@tool + pydantic args_schema
@tool(args_schema=CalculateStockInput)     # pydantic 模型自动生成 JSON Schema
async def calculate_stock(...): ...

# 2️⃣ 绑定 agent_graph.py：把工具清单挂到 LLM 上
llm_with_tools = llm.bind_tools(tools.ALL_TOOLS)
#   bind_tools 本质：每个工具的"名字+描述+参数 JSON Schema"塞进请求体，
#   模型由此知道有哪些工具可用、怎么传参

# 3️⃣ 决策：模型输出带 tool_calls 的 AIMessage
response: AIMessage = await llm_with_tools.ainvoke([system, *messages])
#   response.tool_calls = [{"name": "calculate_stock",
#                           "args": {"code": "600519", ...}, "id": "call_xxx"}]

# 4️⃣ 执行+回传（agent_graph 的 tools_node）：解析并并行执行
tool_messages = await asyncio.gather(*(_run_single_tool_call(tc) for tc in tool_calls))
#   结果包成 ToolMessage 塞回消息历史 → 回到 agent 节点继续推理
```

### 两个加分细节

1. **决策权在模型，执行权在你的代码**：`route_after_agent` 只看 `tool_calls` 有无来路由，调不调、调哪个完全是模型基于 JSON Schema 的自主判断；
2. **pydantic `args_schema` 一石二鸟**：既生成给模型看的参数说明书（JSON Schema），又在 `target.ainvoke(args)` 时做本地入参校验——传错类型不用等远端报错，在进入工具前就被拦下并降级成友好 ToolMessage。

**面试一句话**："bind_tools 给模型发'菜单'，模型点菜（tool_calls），我的代码炒菜（gather 并行执行），再上菜回桌（ToolMessage）——ReAct 循环就转起来了。"

关联 → Q6/Q7（工具如何从进程内走向进程外与标准化）；本项目的进程内实现在 `src/finrag/tools.py`。

---

## Q6 · 基于 CRUD 企业级业务系统怎么开发 MCP 接口？

核心结论先行：**MCP 是加在外面的"工具适配层"，业务系统几乎不动**——如同当年给遗留系统包一层 REST API 对外服务，这次是包一层给 AI 用。

### 三种集成姿势

| 方案 | 做法 | 适用 |
|---|---|---|
| **A. 独立适配服务（推荐）** | 新建轻量 MCP Server，内部用 HTTP/RPC 反调现有系统接口 | 遗留系统零改动，部署独立、故障隔离 |
| **B. 嵌入应用内** | 单体直接挂 MCP endpoint（Spring AI `spring-ai-starter-mcp-server-webmvc` + `@Tool` 注解即成） | 允许发版的 Spring 系统，最省事 |
| **C. 网关聚合** | 一个 MCP Gateway 聚合多个内部系统 | N 个微服务想一次曝光 |

### 关键认知（最容易搞反的方向）

> **业务系统完全不需要集成 LLM。**

| 角色 | 知道什么 | 不知道什么 |
|---|---|---|
| **LLM** | 只有"工具说明书"：工具名+描述+参数 Schema（靠启动时 `tools/list` 拉取） | 你的数据库、SQL、Service 层 |
| **MCP Server** | 自己的代码和数据 | LLM 的存在！收到的只是带参数的结构化调用 |

时序：用户提问 → 客户端把「问题+工具清单」发给 LLM → 模型输出结构化 `{name, args}` → 客户端转发给 MCP Server → **纯普通 Java 代码执行 Service 方法查库返回 JSON** → 结果回填 LLM 上下文组织回答。LLM 只在头尾两次出场（决策/解读），中间执行全程无模型。

### 企业级真正难点（聊这个最加分）

CRUD 不难，难的是安全地交给概率性系统：

- **鉴权穿透**：MCP Server 代表用户还是自己？权限判断永远留在原 Service 层透传用户上下文，**适配层只转发不自建权限**；
- **只读账号兜底**：最小权限账号防模型幻觉拼出意外写操作；
- **审计日志**：谁在哪个会话调了什么工具、传了什么参数回了多少数据——给 AI 开一个全量审计的 API 账号；
- **防注入**：工具返回内容会进模型上下文，业务数据里的恶意文本可能诱导模型（prompt injection via data），敏感字段脱敏后再返回；
- **限流熔断**：LLM 可能高频重试失败调用，限流保护数据库。

### 项目落地对照

本项目已实现手写最小 MCP Client（`src/finrag/mcp_client.py`）：Java 版 Spring AI `@Tool` Server 写法就在其文件头注释；错误策略阶段化——装配期 fail-fast、运行期降级为文本（沿用工具铁律）。

一句话总结：**把 MCP Server 当作"专为 LLM 设计的 API 网关/Facade"做——业务逻辑零迁移、权限校验留原地、功夫花在工具描述和安全边界上。** Java 后端写 provider/Controller 的能力已 100% 够用。

---

## Q7 · 为什么需要 MCP？让 LLM 直接走 REST 不就好了？

先承认直觉正确：技术上 REST 可全程替代 MCP——**MCP 底层传输其实就是 JSON-RPC over stdio/HTTP，比 REST 还朴素**。它解决的不是"能不能调通"，而是集成经济学和 LLM 适用性。

### 先破除误解：LLM 从来不"直接发请求"

无论 REST 还是 MCP，执行的永远是客户端代码：LLM 只输出 `{"name", "args"}` 这段 JSON（不会动），你的代码解析后才真正发起 HTTP/MCP 调用。所以问题的准确形式是："agent 直接调我的 OpenAPI 不就完了？"——能跑通，那 MCP 赚了什么？

### 差异一：集成成本 M×N → M+N（核心卖点）

| | 无标准 REST 集成 | 有 MCP |
|---|---|---|
| 角色 | 每个 agent 团队各自写适配：读 Swagger、挑端点、拼认证、处理分页…… | 系统方实现一次 Server；agent 方实现一次 Client |
| 成本公式 | M 个 agent × N 个系统 = **M×N 套胶水代码** | **M + N** |
| 类比 | JDBC 出现前各家数据库自己连 | **JDBC**：driver 按 SPI 各写一遍，应用层换库零改码 |

Claude Desktop/opencode/Cursor 已内置 MCP Client——你实现一个 Server，这些现成"万能前端"立刻全能调你；REST 方案得一家家谈集成。

### 差异二：REST/OpenAPI 为人设计，不是为 token 设计

- **文档太肥**：OpenAPI spec 几十 KB，进 prompt 直接吃爆上下文；MCP `tools/list` 是精简的"名字+描述+参数 Schema"，专为进 prompt 优化；
- **粒度错位**：REST 给 CRUD 碎片端点，要模型自己编排多步；工具应是语义完整动作（`query_order_detail(orderId)` 一次到位）；
- **缺发现机制**：agent 怎么动态知道你有哪些能力？`tools/list` 协议化了这件事；
- **鉴权模型不同**：API key 是机器对机器；agent 场景是"代理某人"（on behalf of user）的 delegated authorization，裸 REST 要自己发明。

### 公道话：很多场景确实没必要

| 场景 | 合理选择 |
|---|---|
| 自己应用内部工具（finrag 三个 @tool） | 进程内 bind_tools——最简单，没有之一 |
| 公司内部 agent 调自有系统 | REST/gRPC 直调完全可行，薄适配即可 |
| 让任意第三方 AI 客户端即插即用你的能力 | MCP Server（生态红利）|
| 产品想消费海量第三方能力 | MCP Client（社区已有数百个现成 Server）|

类比收尾：像"有了方法调用为什么还要 RPC""有了 Socket 为什么还要 HTTP"——**功能上都能打通，标准化赢在生态位**。MCP 押注的是 AI 应用间工具互操作会成为通用需求。本项目两界都占：进程内 @tools 走主链路，mcp_client.py 演示协议外化，正好构成面试对比素材。

---

## Q8 · CoT、Plan-and-Execute、ReAct 的区别？

一句话定位：**CoT 是"只想不做"，ReAct 是"边想边做、走一步看一步"，Plan-and-Execute 是"先想清楚再动手"**。

| 维度 | CoT（思维链） | ReAct | Plan-and-Execute |
|---|---|---|---|
| 本质 | 提示工程（prompting 技巧） | Agent 循环模式 | Agent 架构模式 |
| 核心动作 | 只推理，输出中间思考步骤 | 思考→行动→观察，循环 | 一次性生成完整计划，再逐步执行（可 replan） |
| 调用工具 | ❌ 不与外部交互 | ✅ 每轮可调 | ✅ 执行阶段调 |
| 规划方式 | 无行动 | **无全局计划**，每步即时决策 | **全局计划先行** |
| LLM 调用 | 1 次 | N 次（每轮思考+决策） | 1 次规划 + N 次执行（执行可用便宜小模型） |
| 典型弱点 | 需要外部信息的题抓瞎 | 绕弯、循环打转、token 烧得多 | 计划赶不上变化，依赖 replan |

**Java 类比**：CoT=不查资料在草稿纸写解题步骤；ReAct=敏捷开发走一步看一步（写代码→跑测试→看结果→决定下一步，没有设计文档）；P&E=先出技术方案评审再排期开发，需求变了走变更（replan）。

**三者是组合而非互斥**：① ReAct 的 R（Reasoning）内置了 CoT——CoT 是 ReAct 推理质量的来源；② P&E 可以看作"大 CoT 规划 + ReAct 式执行"。**选型判据**：任务路径短且不可预知 → ReAct；任务长、步骤可预先拆解 → P&E。

**本项目落点**：finrag 用手写 ReAct（agent ⇄ tools 回环）——问答型任务路径短（检索→可能调工具→生成），**要不要调工具取决于检索结果，无法也无需预先拆解**。若做"生成完整研报"这种长程多阶段任务，才是 P&E 主场：先规划章节再逐节执行。

---

## Q9 · 规划与思考的区别？

**核心判据：产物是"决策"还是"计划"。** 思考回答"**现在这一步**该怎么做"，规划回答"**接下来 N 步**各做什么、什么顺序"。

| 维度 | 思考（Thinking/Reasoning） | 规划（Planning） |
|---|---|---|
| 时间指向 | 即时、单点 | 面向未来、全局 |
| 产物 | 一个决策（调哪个工具/怎么答） | 一份显式计划（子任务列表，可存储/审核/修改） |
| 发生位置 | 每次 LLM 调用内部（推理 token） | 行动之前的**独立阶段**，输出落成结构 |
| 出错代价 | 只错一步，下轮观察可纠 | 起点错→全盘偏，必须有 replan 兜底 |
| 技术代表 | CoT、o1/R1 思维链、ReAct 的 R | Plan-and-Solve、HuggingGPT、TODO list |

**关系：规划是思考的一种特定用途**——面向未来多步的思考。P&E 的 plan 阶段本质是一次强 CoT 思考，但要求产物落成显式结构。所以思考⇏规划（CoT 推数学题没有未来步骤要安排），规划⇒思考。

**Java 类比**：思考≈写代码时琢磨"这一行分支怎么处理"（局部、即时、不留档）；规划≈动手前写技术方案+排期表（全局拆解、可评审、变更走流程）。技术方案文档 vs 草稿纸演算的区别。

**身边的活例子**：opencode 的 todowrite 工具就是外置规划——干多步任务前先列清单逐项勾销（规划），每一步怎么改是思考。**P&E 把两者拆给不同角色（planner/executor），ReAct 只有思考没有规划**——这正是 Q8 说"ReAct 容易绕弯"的根源：没有规划兜底，每步的局部最优走偏了才发现。

**面试话术**："思考是单步推理解决当前决策，规划是多步拆解产出可 replan 的显式计划。ReAct 只有思考所以适合短路径，P&E 把规划独立成阶段所以适合长程——我的 finrag 是查询型短任务，ReAct 够用；要做长报告生成我会加 planner 节点，复用同一个 StateGraph 只是多一类节点和边。"

---

## Q10 · 工具写法规范——为什么 docstring 即接口文档？

**问题现场**：规范化 `tools.py` 时发现 `calculate_stock` 的 docstring 里写着"双保险设计：外层 pydantic 拦截非法参数，内层 try/except 兜底"——**这段架构说明会被 `@tool` 原样发给 LLM**：浪费 token、对模型选工具毫无帮助。

**原理**（详见 Q5 的 bind_tools 链路）：`@tool` 装饰器把函数 docstring 全文透传为请求体的 `description` 字段。docstring 对 Python 是注释，对 tool calling 是**发给模型的接口文档**。

**规范**（已沉淀在 tools.py 文件头第 4 条 + AGENTS.md）：

| 内容 | 写在哪 | 原因 |
|---|---|---|
| 使用场景（何时调用）+ 参数口径 + 返回内容 | 工具 docstring | 这是 LLM 选工具的唯一依据，必须面向"读者是模型" |
| 教学类比、实现细节（安全防护/双保险等） | `#` 代码注释 或 Input 类 docstring | `#` 运行时不存在；pydantic 类 docstring 不进 JSON Schema——两类都不会发给模型 |

**三条延伸结论**：

1. **docstring vs `@tool(description=...)` 二选一**：显式参数优先级更高（实验验证：会覆盖 docstring）。选 docstring 做单一信息源——IDE 悬浮/help()/文档工具与模型共用一份，永不漂移；`description="..."` 单行难写长文本。对照：`args_schema` 反而必须显式传——函数签名只有类型注解，装不下 `Field(gt=0)` 校验。口诀：**描述靠 docstring（一处维护），约束靠 args_schema（签名装不下）**。
2. **docstring 是运行时对象不是注释**：存在 `__obj.__doc__` 属性里（≈ `RetentionPolicy.RUNTIME` 的注解可反射读取），装饰器才能抓取透传；`#` 注释词法层面就被丢弃——这就是两类注释运行时可见性的本质差异。
3. **不写 docstring 语法上能跑、工程上等于盲飞**：模型只能看函数名猜用途，多工具必乱；错参数→错误文本→多烧一轮 ReAct 循环。

**Field description 统一格式**：含义 + 约束 + 示例（如"A股股票代码，6 位数字，如 600998"）。

**面试话术**："每个 tool 必写 docstring，且要写'使用场景'而不是复述函数名——因为它会被原样序列化进请求体的 description，等于给 LLM 看的接口文档，文档质量直接决定工具选择准确率。实现细节一律放代码注释，两类注释以'运行时是否发给模型'划界。"

关联 → Q5（bind_tools 四步链路）、[评估体系.md](评估体系.md) Q3（同样的"透传给模型"原理在 judge 输出契约上的复用）

---

## 关联知识链

workflow vs agent 判据（Q1）→ ReAct 循环节拍器（Q2）→ 循环内工具并发 gather（Q3）→ 并发 vs 并行的边界（Q4）→ function calling 四步链路（Q5）→ 工具进程外化与 MCP 接口设计（Q6）→ MCP vs REST 的标准化经济学（Q7）→ CoT/P&E/ReAct 三模式对比（Q8）→ 规划 vs 思考（Q9）→ 工具 docstring 即接口文档（Q10）。
交叉主题：检索路的打分与融合见 [RAG与检索.md](RAG与检索.md)；学习路径见 [学习路径.md](学习路径.md)；评估分层与 faithfulness 见 [评估体系.md](评估体系.md)。
