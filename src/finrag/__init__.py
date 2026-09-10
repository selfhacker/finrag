"""FinancialAssistant: RAG + Tool Calling 复合 Agent 包。"""

from .agent_graph import graph

# 显式声明公开 API（对应 Java 的模块导出），同时让 ruff 识别 re-export 不再报 F401
__all__ = ["graph"]
