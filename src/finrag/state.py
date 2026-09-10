"""
state.py —— LangGraph 的状态定义
================================================================
职责：定义 Agent 在图中流转时"携带的数据结构"。

对 Java 工程师的类比说明：
- LangGraph 的 State 相当于"工作流上下文对象"（类似 Java 工作流引擎里的
  ProcessInstance Context / Activiti 的 variable），
  每个节点读写这个共享对象，节点之间通过它传递数据。
- State 支持三种定义方式：`TypedDict`、dataclass、Pydantic BaseModel。
  本项目选 Pydantic BaseModel，理由：
  1) 类型检查器友好：langgraph 的 StateT 泛型边界是协议联合
     （TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel），
     TypedDict 需要检查器特判运行时注入的 __required_keys__ 等属性，
     PyCharm/pyrefly/ty 均无法静态确认而误报；普通类继承 BaseModel
     则任何检查器都能验证。
  2) 运行时校验：节点输入/输出自动经过 pydantic 校验（对应 Java 的
     JSR-303 Bean Validation），非法数据在边界即被拦截。

reducer（归约器）原理：
- 普通字段（如 retrieval_docs）：节点返回值会【整体覆盖】旧值。
- 带 reducer 的字段（如 messages 用了 add_messages）：
  节点返回的不是整个列表，而是【增量】，LangGraph 把增量与旧值
  用 reducer 函数合并。add_messages 会对新消息做去重并追加。
  这等价于 Java 中"消息总线 append 语义"，天然适合多轮对话记忆。

为什么所有字段都有默认值：
- 图入口只传 {"messages": [...]},Pydantic 构造 AgentState 时
  其余字段必须可缺省,否则入口 validate 即抛 ValidationError。
"""
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Agent 图的全量状态。

    字段说明：
    - messages: 对话消息序列（含 user / ai / tool 三种消息）。
      使用 Annotated + add_messages 声明合并策略，
      否则每轮节点返回 messages 会互相覆盖，丢失上下文。
    - retrieval_query: 当前轮用户问题的原文，供检索与日志使用。
    - retrieval_docs: RAG 检索到的上下文块列表（每个元素是带 source 的文本）。
      注意：这里【故意不用】reducer，因为每个新问题都要重新检索、
      整体覆盖旧上下文，避免跨轮串味。
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    retrieval_query: str = ""
    retrieval_docs: list[str] = Field(default_factory=list)
