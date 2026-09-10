"""
tools.py —— Agent 可调用的外部工具
=================================================================
职责：定义 fetch_weather / calculate_stock / read_local_doc 三个工具，
      供 LLM 通过 tool calling 调用。

对 Java 工程师的类比说明：
1) `@tool` 装饰器
   - 等价于 Java 中"把方法注册进工具注册表"（类似 Spring 把 @Bean
     注册进容器）。装饰器把函数元信息（名称、参数 Schema、描述）暴露给 LLM，
     LLM 据此生成结构化的 tool_calls。
   - 参数 Schema 用 pydantic 定义并绑定 args_schema，
     等价于 Java 的 `@JsonProperty` + JSR-303 校验注解，但天然序列化为 JSON Schema。

2) async def 异步函数
   - `async def` 定义一个协程函数，调用时返回一个"协程对象"，
     必须由事件循环 await 执行 —— 对应 Java 中返回 CompletableFuture 的方法。
   - 工具内部用 `await asyncio.sleep(...)` 模拟 IO 等待，
     此时事件循环会去调度其它协程，不会像 Java 阻塞线程那样浪费资源。
     对比：Java 是"线程级阻塞"，Python asyncio 是"协程级挂起"，
     单线程即可承载高并发 IO。

3) 错误降级策略
   - 工具调用失败【绝不抛异常】，而是返回友好错误文本。
     异常会被 langgraph 以 ERROR 状态中断整条链路，
     而返回错误文本只是让 LLM 知道"这个工具没成功"，可以继续思考或追问。

4) 工具写法规范（docstring 即接口文档）
   - @tool 会把函数 docstring 全文原样发给 LLM（description 字段），
     因此工具 docstring 必须面向"读者是模型"来写：
     首段 = 使用场景（何时调用），尾段 = 参数口径与返回内容。
   - 教学类比、实现细节（安全防护/双保险等）一律写在 # 代码注释或
     Input 类 docstring（pydantic 类 docstring 不会进 JSON Schema），
     绝不放进工具 docstring —— 避免浪费 token、干扰模型选工具。
"""
import asyncio
import hashlib
import logging
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

logger = logging.getLogger("tools")

# 项目根目录：本文件所在目录的上一级（src/finrag -> 项目根）
# 用于 read_local_doc 做路径穿越防护
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 允许被读取的最大文件字节数（1MB）
MAX_FILE_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# 1) fetch_weather —— 天气查询（模拟外部 HTTP API）
# ---------------------------------------------------------------------------
class WeatherInput(BaseModel):
    """天气查询的输入 Schema。

    对应 Java 中的请求 DTO。LLM 会按此结构生成参数 JSON。
    """
    city: str = Field(..., min_length=1, max_length=32, description="中文城市全名，不超过 32 字，如 北京、深圳")


@tool(args_schema=WeatherInput)
async def fetch_weather(city: str) -> str:
    """查询指定城市的当前天气。

    使用场景：用户询问某城市天气，或出行/着装建议需要天气数据时调用。
    city 传中文城市全名（如"北京"）。返回天气现象、气温与空气质量的文本描述；
    当前为演示环境，返回的是模拟数据。
    """
    # 模拟外部 API 的 IO 耗时（50~200ms）。
    # await 会把控制权交还给事件循环：并行调用多个工具时总耗时约等于最慢那个，
    # 而不是像 Java 同步代码那样线性累加（对应 Java 的 Completa bleFuture.allOf）。
    await asyncio.sleep(0.15)

    # 用城市名的确定性哈希生成稳定天气，保证"同城同结果"，便于演示与测试
    digest = hashlib.md5(city.encode("utf-8")).hexdigest()
    first = int(digest[0], 16)
    temperature = 12 + (first % 18)          # 12 ~ 29 度
    conditions = ["晴", "多云", "小雨", "阴天", "雷阵雨"]
    condition = conditions[first % len(conditions)]

    # 返回结构化文本而非 dict：ToolMessage 的 content 需要是字符串
    return f"{city} 天气：{condition}，气温 {temperature}℃，空气质量良。"


# ---------------------------------------------------------------------------
# 2) calculate_stock —— 股票持仓盈亏计算（严格校验 + 友好降级）
# ---------------------------------------------------------------------------
class StockInput(BaseModel):
    """股票计算输入 Schema。

    关键点：price / shares 的类型就是校验器。
    - pydantic v2 解析失败会抛出 ValidationError，被 tools 节点捕获并转为友好提示，
      绝不会让进程崩溃（对应 Java 的 MethodArgumentNotValidException + @ExceptionHandler）。
    - Field(gt=0) 增加"正数"约束，负价/负股数也会被拒绝。
    """
    ticker: str = Field(..., min_length=1, max_length=16, description="A股股票代码，6 位数字，如 600998、300750")
    price: float = Field(..., gt=0, description="当前股价，单位元，正数，如 12.50")
    shares: int = Field(..., gt=0, description="持股数量，正整数，如 1000")


@tool(args_schema=StockInput)
async def calculate_stock(ticker: str, price: float, shares: int) -> str:
    """根据股票代码、现价与持股数计算持仓市值与盈亏。

    使用场景：用户提供持股信息，要求计算市值或盈亏时调用。
    成本价为系统内置常量，仅支持 600998/300750/600519 三只，
    未收录的代码会明确提示可查范围。返回市值、成本、盈亏金额与百分比的文本结果。
    """
    # 双保险设计（教学点，不放进 docstring——那会原样发给 LLM）：
    # 1) 外层：pydantic 在进入函数前已拦截非法参数（如字母 price）；
    # 2) 内层：函数体再 try/except 一次，即使绕过 pydantic 也能兜底降级。
    try:
        # 预设每只股票的成本价（演示用），真实场景应查行情服务
        COST_PRICE_MAP = {"600998": 9.80, "300750": 22.10, "600519": 1680.0}

        cost = COST_PRICE_MAP.get(ticker.upper())
        if cost is None:
            return f"未收录股票 {ticker} 的成本价，无法计算盈亏。可查询：{list(COST_PRICE_MAP)}"

        # 注意：这里不再需要校验数字类型——pydantic 已保证 price/shares 是合法数值。
        # 仅做业务层校验（防呆）。
        if price <= 0 or shares <= 0:
            return "股价与持股数量必须为正数，请检查输入。"

        market_value = price * shares
        cost_value = cost * shares
        pnl = market_value - cost_value
        pnl_pct = (pnl / cost_value) * 100.0

        return (
            f"{ticker} 持仓 {shares} 股：现价 {price:.2f}，成本 {cost:.2f}；"
            f"市值 {market_value:.2f} 元，盈亏 {pnl:+.2f} 元（{pnl_pct:+.2f}%）。"
        )
    except Exception as exc:  # 最终兜底：任何异常都转成友好提示而非崩溃
        logger.exception("calculate_stock 内部异常")
        return f"股票计算失败（系统内部错误）：{exc}"


# ---------------------------------------------------------------------------
# 3) read_local_doc —— 本地文件读取（带安全防护）
# ---------------------------------------------------------------------------
class ReadDocInput(BaseModel):
    """本地文件读取输入 Schema。"""
    file_path: str = Field(
        ...,
        min_length=1,
        description="相对项目根目录的文件路径，目标为 data/ 下的语料文件，如 data/2024年年报摘要-华信科技.txt",
    )


@tool(args_schema=ReadDocInput)
async def read_local_doc(file_path: str) -> str:
    """读取项目内语料文档（如 data/ 下的公司财报 TXT）的文本内容。

    使用场景：需要查看语料原文细节、或为检索结果补充上下文时调用。
    file_path 传相对项目根目录的路径（如 data/xxx.txt）；
    仅允许读取项目目录内不超过 1MB 的文本文件，
    返回文件大小与正文前 3000 字符。
    """
    try:
        # --- 路径穿越防护（Security）---
        # Path.resolve() 会消除 ../ 和符号链接，得到绝对路径。
        # 必须校验解析后的绝对路径仍位于项目根目录内，
        # 否则攻击者可用 ../../etc/passwd 读取任意文件（对应 Java 的 Path.normalize + startsWith 校验）。
        target = (PROJECT_ROOT / file_path).resolve()
        if not target.is_relative_to(PROJECT_ROOT):
            return f"拒绝访问：{file_path} 超出允许的读取范围（项目根目录内）"

        if not target.exists():
            return f"文件不存在：{file_path}。可先检索 data/ 目录确认文件名。"

        # --- 文件大小防护（DoS 防御）---
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            return f"文件过大（{size} 字节），超过 1MB 上限，拒绝读取。"

        # 用 utf-8 读取，errors="replace" 避免个别乱码字符导致整体失败
        content = target.read_text(encoding="utf-8", errors="replace")
        # 截断到前 3000 字符，防止超大内容撑爆 LLM 上下文
        preview = content[:3000]
        return f"文件 {file_path}（{size} 字节）：\n{preview}"
    except Exception as exc:  # 兜底：读取失败转友好提示，不让链路中断
        logger.exception("read_local_doc 读取异常")
        return f"读取文件失败：{exc}"


# 工具清单：供 agent 节点一次性 bind 给 LLM
ALL_TOOLS: list[BaseTool] = [fetch_weather, calculate_stock, read_local_doc]
