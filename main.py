"""
main.py —— FinancialAssistant 交互入口
=================================================================
功能：
1) 交互式 CLI：流式输出（graph.astream），多轮对话带记忆（thread_id 会话隔离）。
2) --demo 模式：演示"并发限流器"（asyncio.Semaphore），模拟多个用户并发提问，
   验证并发控制在 MAX_CONCURRENCY(=5) 以内。

对 Java 工程师的类比说明：
- asyncio.Semaphore(5)     ≈ Java 的 java.util.concurrent.Semaphore(5)
  `async with semaphore`   ≈ semaphore.acquire()/release() 的 try-with-resources 写法
  `asyncio.gather(*tasks)` ≈ CompletableFuture.allOf(...).join()（并发调度）
- 区别：Java 靠线程承载并发，Python asyncio 靠事件循环协程切换，
  单线程即可同时处理多个 IO 型请求，但 CPU 密集计算不受益。

运行：
    uv run python main.py            # 交互模式
    uv run python main.py --demo     # 并发压测演示
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage

from finrag import config
from finrag.agent_graph import graph

# 加载项目根 .env（锚定路径，避免从非项目根目录启动时加载不到密钥）
load_dotenv(Path(__file__).resolve().parent / ".env")

# 日志配置：输出 @trace 打点（P99 复盘依据）。
# 生产建议接入 json handler 统一采集到 ELK/Loki。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# 压低第三方 HTTP 库的 INFO 噪音，让 @trace 日志更醒目
for noisy in ("httpx", "httpcore", "urllib3", "openai", "chromadb.telemetry"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("cli")


async def stream_answer(thread_id: str, question: str) -> None:
    """以流式方式让 graph 处理一个问题，并实时打印 LLM 输出。

    graph.astream(stream_mode="messages")：
    - 逐 token 产出 (AIMessageChunk, metadata) 二元组，
      对应 Java 中"SSE / WebFlux 响应式流"的逐片下发。
    - 每轮 agent 思考、工具调用都会产出多个 AI 消息片段。
    """
    print(f"\n  You: {question}\n  AI: ", end="", flush=True)

    # 注意：graph 内部仍会完整执行 retrieve -> agent -> (tools) -> generate，
    # 只是输出层按 token 流式返回，而不是一次性吐整段（等效于流式 HTTP 响应）。
    async for chunk, _metadata in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},   # 会话标识，隔离多轮记忆
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        # 工具调用片段：content 为空、tool_call_chunks 有值
        if chunk.tool_call_chunks:
            print(f"\n  [调用工具: {chunk.tool_call_chunks[0].get('name', '?')}] ", end="", flush=True)
            continue
        text = chunk.content
        if isinstance(text, str) and text:
            print(text, end="", flush=True)

    print("\n" + "-" * 60)


async def interactive_loop() -> None:
    """交互式多轮对话：同一个 thread_id，LLM 通过 checkpoint 记忆上下文。"""
    thread_id = input("请输入会话 ID（默认 default，回车确认）: ").strip() or "default"
    print("输入 exit / quit 退出。")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        await stream_answer(thread_id, question)


async def concurrency_demo() -> None:
    """并发限流演示：一次性提交 N 个请求，验证并发被控制在 MAX_CONCURRENCY 内。

    实现：
    - 全局唯一 asyncio.Semaphore(MAX_CONCURRENCY)；
    - 每个请求先 `async with semaphore` 抢占令牌，抢不到就排队等待；
    - asyncio.gather 让所有请求协程并发调度（不是顺序执行）。
    """
    max_concurrency = config.MAX_CONCURRENCY
    total = max_concurrency * 2   # 8 个请求，超出信号量容量 -> 观察排队
    semaphore = asyncio.Semaphore(max_concurrency)
    # 活跃请求计数（模拟压测指标采集）；nonlocal 让闭包可改写外层 int
    active: int = 0

    async def worker(i: int, question: str) -> None:
        nonlocal active
        async with semaphore:
            active += 1
            logger.info("请求 #%s 获得令牌，当前活跃=%s/%s", i, active, max_concurrency)
            try:
                await stream_answer("demo-session", question)
            finally:
                active -= 1

    questions = [
        "华信科技 2024 年营业收入是多少？同比增长多少？",
        "华信科技 2025 年一季度 AI 业务收入是多少？",
        "储能行业 2025 年装机增长情况如何？",
        "如何用 ROE 判断企业盈利质量？",
        "华信科技 2024 年研发投入占比是多少？",
        "计算 600998 以 12.5 元、持有 2000 股的盈亏",
        "报告华信科技的分红计划",
        "什么是净现比？为什么重要？",
    ][:total]

    logger.info("并发压测演示：提交 %s 个请求，信号量上限 %s", total, max_concurrency)
    await asyncio.gather(*(worker(i, q) for i, q in enumerate(questions, start=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="FinancialAssistant CLI")
    parser.add_argument("--demo", action="store_true", help="运行并发限流演示")
    parser.parse_args()  # 当前固定跑 demo（调试状态），保留参数解析不报未使用
    asyncio.run(concurrency_demo())
    # if args.demo:
    #     asyncio.run(concurrency_demo())
    # else:
    #     # asyncio.run 创建/关闭事件循环，等价于 Java 主线程启动一个 ExecutorService
    #     asyncio.run(interactive_loop())


if __name__ == "__main__":
    sys.exit(main())
