"""
trace.py —— 关键节点耗时打点装饰器
=================================================================
职责：为 sync / async 函数统一打印执行耗时，用于事后分析 P99 延迟。

对 Java 工程师的类比说明：
- 装饰器（decorator）≈ Java 注解 + AOP 切面（如 Spring @Around）。
  但 Python 装饰器是"运行时的函数包装"，直接返回一个新函数，
  比 Java 的字节码织入更透明。
- 这里的耗时打点 ≈ 微服务的统一 Logging Filter / 链路追踪 Span。
  生产环境可替换为 OpenTelemetry 的 span，结构完全一致。
"""
import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("trace")


def trace(tag: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """打点装饰器工厂。

    用法：
        @trace("retrieve")          # 同步函数
        def retrieve(state): ...

        @trace("async_tool")        # 异步函数
        async def fetch_weather(city: str): ...

    实现细节：
    1) 外层函数 `decorator` 接收被装饰函数 fn；
    2) 用 functools.wraps 保留 fn 的名称/文档，便于调试与 IDE 提示；
    3) 若 fn 是协程函数（inspect 判断），返回 async 包装；否则返回 sync 包装。
       这是"装饰器同时兼容 async/sync"的标准写法（Java 无法直接用注解实现）。
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # functools.wraps 相当于 Java 的 @Override 保留元信息，
        # 否则装饰后 __name__ 会变成 wrapper
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _log(tag or fn.__name__, start)

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                _log(tag or fn.__name__, start)

        # inspect.iscoroutinefunction 判断是否为 async def 定义的函数
        # 对应 Java：判断方法是否返回 CompletableFuture 以决定是否异步包装
        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def _log(name: str, start: float) -> None:
    """统一日志格式：`[TRACE] <name> took 123.45 ms`。

    生产建议：改用 logging 的 json handler 输出，方便直接接入
    日志采集（如 Loki/ELK）做 P99 聚合分析。
    """
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("[TRACE] %s took %.2f ms", name, elapsed_ms)
