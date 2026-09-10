"""
agent_graph.py —— FinancialAssistant 的 LangGraph 状态图（核心）
=================================================================
职责：用 StateGraph 手写节点与条件边，编排
      retrieve(检索) -> agent(思考) <-> tools(执行) -> generate(收尾) 的 ReAct 循环。

对 Java 工程师的类比说明：
- StateGraph ≈ 工作流引擎（Activiti/Flowable）的状态机：
  · 节点(node)        ≈ 流程步骤 / ServiceTask
  · 条件边(conditional edge) ≈ 排他网关，按路由函数返回值决定下一跳
  · 状态(state)       ≈ 流程上下文变量
  区别：Java 工作流节点一般是"服务编排"，这里的节点是"LLM 推理 + 工具执行"，
  且 LangGraph 节点是可回环的（tools 回到 agent），本质是图而非线性流程。

- ReAct 循环（Reasoning + Acting）：
  思考 -> 调工具 -> 看结果 -> 再思考...，直到 LLM 认为信息足够、不再发起
  tool_calls，才进入 generate 收尾。这等价于 Java 中"while 循环"，
  只是循环条件由 LLM 的输出决定。

- checkpointer(MemorySaver) ≈ 会话快照持久化：
  每次运行结束保存状态快照，通过 thread_id 恢复历史，
  等价于把流程上下文序列化存档（生产可换成 SQLite/Redis 存储）。
"""
import asyncio
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolCall, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from . import config, tools
from .rag_retriever import FinancialRetriever
from .state import AgentState
from .trace import trace

logger = logging.getLogger("graph")

# 全局限量 LLM / 检索器实例（等价于 Java 的单例 Bean）
_llm: ChatOpenAI | None = None
_retriever: FinancialRetriever | None = None


def _get_llm() -> ChatOpenAI:
    """惰性单例：首次调用时创建，后续复用（等价于 Java double-checked singleton）。

    实现细节：先读入局部变量再判断/回写。
    PyCharm 对 global 变量的流分析不传播分支内赋值的收窄结果，
    直接 `return _llm` 会误报 ChatOpenAI | None；局部变量可被
    所有类型检查器正常收窄，且行为与原写法完全一致。
    """
    global _llm
    llm = _llm
    if llm is None:
        llm = config.get_llm()
        _llm = llm
    return llm


def get_retriever() -> FinancialRetriever:
    """获取全局检索器单例（若为 None 必先构造，实际永不返回 None）。"""
    global _retriever
    retriever = _retriever
    if retriever is None:
        retriever = FinancialRetriever()
        _retriever = retriever
    return retriever


# ---------------------------------------------------------------------------
# 节点 1：retrieve —— 根据用户问题检索 RAG 上下文
# ---------------------------------------------------------------------------
@trace("node.retrieve")
async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """从最近一条用户消息提取 query，检索并把结果写入 state.retrieval_docs。

    对应 Java：Controller 收到请求后先调用"检索服务"把上下文预加载进 ThreadLocal/上下文，
    再进入业务逻辑。这里用 state 承载，节点间无隐式耦合。
    """
    # messages 最后一条就是当前用户输入（多轮对话下自动取最新）
    last_message = state.messages[-1]
    query: str = str(last_message.content)

    chunks = await get_retriever().retrieve(query)
    # 转成带来源标记的上下文文本列表，供 agent 节点拼接进 System Prompt
    docs = [c.to_context_str() for c in chunks]

    return {"retrieval_query": query, "retrieval_docs": docs}


# ---------------------------------------------------------------------------
# 节点 2：agent —— LLM 思考（可发起 tool_calls）
# ---------------------------------------------------------------------------
@trace("node.agent")
async def agent_node(state: AgentState) -> dict[str, Any]:
    """调用 LLM 思考。

    关键点：
    1) 【手写 Prompt】：System Prompt 显式拼接 retrieval_docs 与工具清单，
       让 LLM 同时具备"RAG 上下文"与"可调用的工具"两种能力。
    2) bind_tools：把工具 Schema 注册给 LLM，模型输出 AIMessage.tool_calls。
    3) messages 通过 state 累积，实现多轮对话记忆。
    """
    llm = _get_llm()
    messages = state.messages

    # 从 state 读取检索上下文（retrieve 节点写入）
    retrieval_docs = state.retrieval_docs
    context_block = "\n\n".join(retrieval_docs) if retrieval_docs else "（本次没有检索到相关文档）"

    # 显式拼接 System Prompt —— 展示手写 Prompt 能力，不依赖任何封装
    system = SystemMessage(content=(
        "你是一位严谨的金融助手 FinancialAssistant，服务于股票、财务与行业分析咨询。\n"
        "回答规则：\n"
        "1. 优先依据下方【检索到的文档】作答，引用时标注文档来源；文档不足时明确说'资料未覆盖'，禁止编造。\n"
        "2. 需要实时数据（天气、行情、本地文件）时，调用可用工具获取，不要凭记忆猜测。\n"
        "3. 可以同时发起多个相互独立的工具调用以提升效率。\n"
        "4. 用中文简洁作答，涉及数字给出计算过程。\n\n"
        "【检索到的文档】\n"
        f"{context_block}"
    ))

    # bind_tools 返回一个带工具 Schema 的模型副本；invoke 前组合 system + 历史消息
    llm_with_tools = llm.bind_tools(tools.ALL_TOOLS)
    response: AIMessage = await llm_with_tools.ainvoke([system, *messages])

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 节点 3：tools —— 执行 LLM 发起的工具调用（并行）
# ---------------------------------------------------------------------------
@trace("node.tools")
async def tools_node(state: AgentState) -> dict[str, Any]:
    """解析 AIMessage.tool_calls 并【并行】执行，返回 ToolMessage 列表。

    并行原理：asyncio.gather 同时调度多个协程，对应 Java 的
    CompletableFuture.allOf() 并发调用多个下游服务。
    由于工具内部是 async（await 挂起而非阻塞线程），单事件循环即可并发。
    """
    # isinstance 收窄联合类型 AnyMessage -> AIMessage（同时防御非 AI 消息）
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        logger.warning("tools 节点最后一条消息不是 AIMessage，跳过工具执行")
        return {"messages": []}

    tool_calls = last_message.tool_calls or []

    if not tool_calls:
        return {"messages": []}

    logger.info("并行执行 %s 个工具调用: %s", len(tool_calls),
                [tc["name"] for tc in tool_calls])

    # gather(*coros)：把每个协程对象并发调度，全部完成才返回
    tool_messages: list[ToolMessage] = await asyncio.gather(
        *(_run_single_tool_call(tc) for tc in tool_calls)
    )
    return {"messages": tool_messages}


async def _run_single_tool_call(tool_call: ToolCall) -> ToolMessage:
    """执行单个工具调用，并做【错误降级】：任何异常都转成友好 ToolMessage。

    对应 Java 的 @RestControllerAdvice 全局异常处理 + CircuitBreaker fallback：
    不让工具异常炸掉整条链路，而是把失败原因作为消息回传给 LLM，
    让 LLM 决定如何向用户解释或换一种问法。
    """
    name = tool_call["name"]
    args = tool_call.get("args", {})
    call_id = tool_call["id"] or ""

    # 从工具清单中按名称找到目标工具
    target = next((t for t in tools.ALL_TOOLS if t.name == name), None)
    if target is None:
        return ToolMessage(content=f"未知工具：{name}", tool_call_id=call_id, name=name)

    try:
        # ainvoke 会先按 args_schema 校验参数再执行函数体。
        # 校验失败（如给 calculate_stock 传字母 price）会抛 ToolInputParsingException。
        content = await target.ainvoke(args)
        return ToolMessage(content=str(content), tool_call_id=call_id, name=name)
    except Exception as exc:  # noqa: BLE001 - 工具异常刻意兜底降级，不让链路中断
        # 友好降级：返回校验失败原因，而非让节点崩溃
        logger.warning("工具 %s 执行失败（已降级）: %s", name, exc)
        return ToolMessage(
            content=f"工具 {name} 调用失败：{exc}。"
                    f"请检查参数是否符合要求（数字参数必须是数值，路径须位于项目目录内）。",
            tool_call_id=call_id,
            name=name,
        )


# ---------------------------------------------------------------------------
# 节点 4：generate —— 生成最终答案
# ---------------------------------------------------------------------------
@trace("node.generate")
async def generate_node(state: AgentState) -> dict[str, Any]:
    """工具循环结束后，再让 LLM 基于全部上下文生成最终答案。

    为什么需要这个节点：agent 节点输出的可能是"带工具调用的中间态"，
    generate 用一次"收尾"调用生成面向用户的完整回答（无 tool_calls）。
    """
    llm = _get_llm()
    messages = state.messages

    system = SystemMessage(content=(
        "你已获得全部工具执行结果。请基于历史对话、检索文档与工具结果，"
        "给出最终答复：先给结论，再给关键依据；引用文档时标注来源；"
        "若工具调用失败或信息不足，如实说明。不要再调用任何工具。"
    ))
    response: AIMessage = await llm.ainvoke([system, *messages])

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 条件边：判断 agent 输出是否包含工具调用
# ---------------------------------------------------------------------------
def route_after_agent(state: AgentState) -> Literal["tools", "generate"]:
    """ReAct 循环的决策点（等价于排他网关）。

    看最近一条 AIMessage：
    - 有 tool_calls  -> 进入 tools 节点执行工具，之后回到 agent 继续思考
    - 没有 tool_calls -> 信息已足够，进入 generate 收尾
    """
    last_message = state.messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "generate"


# ---------------------------------------------------------------------------
# 图组装
# ---------------------------------------------------------------------------
def build_graph() -> CompiledStateGraph:
    """组装 StateGraph：注册节点、连边、挂 checkpointer。"""
    builder = StateGraph(AgentState)

    # 注册四个节点
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("generate", generate_node)

    # 入口：START -> retrieve -> agent
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "agent")

    # ReAct 循环核心：
    #   agent --(有tool_calls)--> tools --(loop)--> agent
    #   agent --(无tool_calls)--> generate --> END
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "generate": "generate"},  # 路由表：返回值 -> 目标节点
    )
    builder.add_edge("tools", "agent")   # 工具结果回到 agent 继续思考（回环边）
    builder.add_edge("generate", END)

    # MemorySaver：内存版检查点，按 thread_id 保存/恢复会话状态。
    # 生产可换 langgraph-checkpoint-sqlite / postgres 实现跨进程持久化。
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# 模块级单例：main.py 直接 import 使用
graph = build_graph()
