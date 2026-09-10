"""
mcp_client.py —— 最小化 MCP Client（教学实现，零新依赖）
=================================================================
职责：演示 finrag 如何作为 MCP 客户端连接一个远程 MCP Server，
      完成「发现工具(tools/list) -> 调用工具(tools/call)」的完整链路，
      并把远程工具无缝转换成 LangChain 工具接入现有 bind_tools 主链路。

设计决策：为什么不装官方 `mcp` SDK？
    本项目刻意手写 ReAct 循环（不用 AgentExecutor/create_react_agent），
    同理这里也手写协议报文 —— MCP 的 wire format 就是 JSON-RPC 2.0
    （对应 Java 里理解 Dubbo 前先手动用 Socket 发一次 dubbo 协议报文，
    看清"框架魔法"底下只是字节流约定的组合）。看完本文件，
    再换 langchain-mcp-adapters / 官方 SDK 时一切尽在掌控。

-----------------------------------------------------------------
MCP 时序（Streamable HTTP 传输，2025-06-18 协议版本）：

    finrag(Python)                          业务系统(Java)
    ---------------                         ---------------
    POST initialize -----------------------> 握手：交换协议版本与能力
      <-- 200 {result, header: Mcp-Session-Id}
    POST notifications/initialized --------> (通知，无响应体)
    POST tools/list -----------------------> 返回工具目录 = "给 LLM 看的菜单"
      <-- 200 {tools:[{name, description, inputSchema}]}
    POST tools/call {name, arguments} -----> 执行 = 普通确定性 Java 方法
      <-- 200 {content:[{type:"text", text:"..."}]}

    注意：全程 LLM 只出现在 finrag 这一侧（决定调哪个工具、如何解读结果）；
    Java 服务端收到的永远是结构化参数，执行的是普通 CRUD 逻辑 ——
    "业务系统不集成 LLM"。

与 finrag 进程内链路（tools.py + agent_graph.py）的一一对照：

    进程内 @tool                 对应 MCP 世界
    -------------------------    --------------------------------
    ALL_TOOLS 工具清单        -> tools/list 返回的工具目录
    bind_tools(挂说明书)      -> 把工具目录放进 LLM 的请求上下文
    模型输出 tool_calls       -> 客户端据此构造 tools/call 请求
    tools_node gather 执行    -> MCP Server 收到请求执行 Service 方法
    ToolMessage 回填          -> content[0].text 结果回填上下文

-----------------------------------------------------------------
【Java 版 MCP Server 怎么写】—— 企业级 CRUD 系统加 MCP 能力的参考：

方案 B（嵌入应用内，Spring AI 最省事），在现有工程加依赖后：

    // build.gradle: implementation 'org.springframework.ai:spring-ai-starter-mcp-server-webmvc'
    @Service
    public class OrderMcpTools {

        // 每个 @Tool ≈ 一个面向 LLM 的 Controller 方法；
        // description 和 @ToolParam 的描述质量直接决定模型选工具的准确率。
        @Tool(name = "query_order",
              description = "根据订单号查询订单详情，含状态、金额与商品明细")
        public OrderDetail queryOrder(
                @ToolParam(description = "订单号，如 SO20260801") String orderId) {
            return orderService.findDetail(orderId);   // 直接复用现有 Service 层，零迁移
        }

        // 企业级要点（与主对话讨论一致）：
        // 1) 查询类优先暴露；写操作须叠加审批/二次确认机制
        // 2) 权限校验留在原 Service 层透传用户上下文，MCP 层只转发不自建权限
        // 3) 返回内容脱敏后再交给模型，防数据外泄与提示注入（prompt injection via data）
        // 4) 全量审计日志：谁在哪个会话调了什么工具、传了什么参数
    }

方案 A（独立适配服务）：新建轻量 Spring Boot 应用跑 MCP Server，
通过 HTTP/RPC 反调现有系统 —— 业务主站零改动，推荐用于遗留系统。

调试利器：官方 MCP Inspector（npx @modelcontextprotocol/inspector）可视化测握手与调用。
"""
import itertools
import json
import logging
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from finrag.config import MCP_SERVER_URL

logger = logging.getLogger("mcp_client")

# 说明：MCP_SERVER_URL 统一由 config.py 从 .env 读取（项目"单一配置中心"约定），
# 导入本模块即自动完成 .env 加载，无需在此重复 load_dotenv。
# 协议版本协商：客户端声明自己支持的最高版本，服务端返回实际采用的版本
PROTOCOL_VERSION = "2025-06-18"
# Streamable HTTP 规范要求：客户端必须同时接受 JSON 与 SSE 两种响应形态
ACCEPT_HEADERS = {"Accept": "application/json, text/event-stream"}
REQUEST_TIMEOUT = httpx.Timeout(30.0)


class McpToolDescriptor(BaseModel):
    """tools/list 返回的单个工具描述 —— 相当于 RPC 框架里的接口元数据。

    对应 Java 中 OpenAPI 解析后的 EndpointInfo：
    - inputSchema 就是该工具的"入参说明书"（JSON Schema 格式），
      后续 bind_tools 给 LLM 看的正是它的精简版。
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")

    model_config = {"populate_by_name": True}


class McpError(RuntimeError):
    """MCP 协议级错误（握手失败/服务端返回 error 对象等）。

    设计上只在【装配阶段】抛出（list_tools 时fail-fast，让配置错误尽早暴露，
    对应 Java 启动期的 BeanCreationException）；而【运行期调用】call_tool
    则遵循本工具模块的铁律：绝不抛异常，失败转为可读文本（见下）。
    """


def _parse_response_payload(resp: httpx.Response) -> dict[str, Any]:
    """解析 Streamable HTTP 的响应体：标准 JSON 或 SSE（text/event-stream）两种形态。

    SSE 形态下，JSON-RPC response 位于若干 `data:` 行中（教学版取最后一条）。
    对应 Java 用 BufferedReader 逐行读 HttpsURLConnection 流再拼 JSON。
    """
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        data_lines = [
            line[len("data:") :].strip() for line in resp.text.splitlines() if line.startswith("data:")
        ]
        if not data_lines:
            raise McpError("SSE 响应中未找到 data 行")
        return _loads_object(data_lines[-1])
    return _loads_object(resp.text)


def _loads_object(raw: str) -> dict[str, Any]:
    """json.loads + dict 类型收窄：JSON-RPC 报文必须是对象，否则视为协议错误。

    （静态检查器视角：json.loads 返回 Any，这里统一收窄为 dict[str, Any]，
     等价于 Java 的 ObjectMapper.readValue(raw, Map.class) + 强类型转换。）
    """
    value: Any = json.loads(raw)
    if not isinstance(value, dict):
        raise McpError(f"JSON-RPC 报文应为对象，实际为: {type(value).__name__}")
    return value


class McpClient:
    """最小 MCP 客户端：管理一个会话生命周期内的 initialize -> list_tools -> call_tool。

    对 Java 工程师的类比：
    - 一个 McpClient 实例 ≈ 一个带会话态的 Feign/S RestClient 客户端
      （initialize 建立 session 后，后续每笔请求都要带上 Mcp-Session-Id，
       类似登录后每笔请求带 Cookie/JWT）。
    - async with 连接管理 ≈ try-with-resources，实例不复用、生命周期短。
    """

    def __init__(self, base_url: str, *, timeout: httpx.Timeout | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout or REQUEST_TIMEOUT
        self._session_id: str | None = None  # 服务端握手后下发（HTTP 头返回，非 body）
        # JSON-RPC 要求 request id 单调递增；itertools.count 是无界计数器的惯用法
        self._ids = itertools.count(1)

    # -- 底层：单次 JSON-RPC 往返 -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """组装公共请求头：Accept（规范要求）+ 会话标识（握手后才有）。"""
        headers = dict(ACCEPT_HEADERS)
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        return headers

    async def _rpc(self, client: httpx.AsyncClient, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送一条 JSON-RPC 2.0 请求并解包 result；error 对象转成异常。

        request id 用递增整数 —— 让响应能和并发中的请求配对（本客户端串行调用，
        简化为顺序配对即可；生产异步多路复用时 id 配对是硬要求）。
        """
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": method,
            "params": params,
        }
        resp = await client.post(self._base_url, json=payload, headers=self._headers())
        resp.raise_for_status()

        session_id = resp.headers.get("mcp-session-id")
        if session_id:
            # 握手响应里首次拿到会话标识（HTTP 头携带，等价于 Set-Cookie 语义）
            self._session_id = session_id

        message = _parse_response_payload(resp)
        if "error" in message:
            err = message.get("error") or {}
            code = err.get("code", "?")
            text = err.get("message", "未知错误")
            raise McpError(f"MCP 服务端错误 [{code}]: {text}")
        return message.get("result") or {}

    async def _notify(self, client: httpx.AsyncClient, method: str) -> None:
        """发送 JSON-RPC 通知（无 id、无响应体，fire-and-forget，类似 MQ 消息）。"""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        await client.post(self._base_url, json=payload, headers=self._headers())

    # -- 对外：协议三步走 ---------------------------------------------------------

    async def list_tools(self) -> list[McpToolDescriptor]:
        """握手 + 拉取工具目录。【装配阶段】失败直接抛异常 fail-fast。"""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            handshake = await self._rpc(
                client,
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},  # 教学 demo 不需要任何服务端能力
                    "clientInfo": {"name": "finrag-mcp-demo", "version": "0.1.0"},
                },
            )
            server_name = (handshake.get("serverInfo") or {}).get("name", "?")
            agreed = handshake.get("protocolVersion", "?")
            logger.info("MCP 握手成功: server=%s 协议=%s", server_name, agreed)

            # 握手完成的通知（通知不等待响应，属单向告知）
            await self._notify(client, "notifications/initialized")

            result = await self._rpc(client, "tools/list", {})
            raw_tools = result.get("tools")
            if not isinstance(raw_tools, list):
                raise McpError(f"tools/list 返回格式异常: {type(raw_tools).__name__}")
            descriptors = [McpToolDescriptor.model_validate(t) for t in raw_tools]
            logger.info("发现 %s 个远程工具: %s", len(descriptors), [d.name for d in descriptors])
            return descriptors

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用远程工具并返回文本结果。【运行期】绝不抛异常，失败降级为错误说明文本。

        该策略与 tools.py/_run_single_tool_call 一致：未来把此方法接入 LangGraph
        的 tools 节点时，失败也只是让 LLM 看到"这步没成功"，链路不会中断。
        真正的工具结果在 content[0].text（type=text 为协议定义的主流形态）。
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # 远程调用前确保会话存在：demo 场景鼓励一次性 client 即用即弃
                await self._ensure_session(client)
                result = await self._rpc(client, "tools/call", {"name": name, "arguments": arguments})

                # isError=True 表示业务层执行失败（如订单不存在），同样降级为文本
                if result.get("isError"):
                    return f"工具 {name} 执行失败：{_extract_text(result)}"
                return _extract_text(result)
        except McpError as exc:
            logger.warning("MCP 调用 %s 协议级失败: %s", name, exc)
            return f"工具 {name} 调用失败：{exc}"
        except httpx.HTTPError as exc:  # 网络层错误（超时/连接拒绝/5xx）
            logger.warning("MCP 调用 %s 网络失败: %s", name, exc)
            return f"工具 {name} 调用失败（网络异常）：{exc}。请检查 MCP_SERVER_URL 是否可达。"
        except Exception as exc:  # noqa: BLE001 —— 刻意兜底：工具失败绝不炸链路（见类注释铁律）
            logger.warning("MCP 调用 %s 意外失败: %s", name, exc)
            return f"工具 {name} 调用失败：{exc}"

    async def _ensure_session(self, client: httpx.AsyncClient) -> None:
        """call_tool 若被单独使用（未先 list_tools），此处补一次握手。"""
        if self._session_id is None:
            await self._rpc(
                client,
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "finrag-mcp-demo", "version": "0.1.0"},
                },
            )
            await self._notify(client, "notifications/initialized")


def _extract_text(result: dict[str, Any]) -> str:
    """从 tools/call 的 result 中抽取文本内容；无文本时回退到原始 JSON 序列化。"""
    content = result.get("content")
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# JSON Schema -> LangChain 工具的桥接：让远程工具直接接入现有 bind_tools 主链路
# ---------------------------------------------------------------------------

# JSON Schema 基本类型 -> Python 注解的最小映射表。
# 生产级实现还应处理 enum/format/nested object 等，此处保持教学极简。
_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
}


def _schema_to_pydantic(desc: McpToolDescriptor) -> type[BaseModel] | None:
    """把工具的 inputSchema 动态翻译成 pydantic 模型，充当 args_schema。

    pydantic.create_model 动态建模 ≈ Java 在运行期用反射 + ASM 字节码生成 DTO 类。
    这样 LLM 看到的参数约束与远程接口完全同源 —— "说明书即契约"，
    也是 StructuredTool 校验入参的依据（传错类型在本地就被拦下，不必等远端报错）。
    """
    props = desc.input_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return None

    required: set[str] = set(desc.input_schema.get("required") or [])
    fields: dict[str, Any] = {}
    for field_name, raw_spec in props.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        annotation = _JSON_TYPE_MAP.get(str(spec.get("type", "string")), str)
        description = str(spec.get("description", ""))
        default = ... if field_name in required else None  # ... 表示必填
        fields[field_name] = (annotation, Field(default, description=description))

    model_name = f"{desc.name.title().replace('_', '')}Input"
    return create_model(model_name, **fields)


async def to_langchain_tools(base_url: str) -> list[BaseTool]:
    """拉取远程工具目录并包装成 LangChain 工具 —— 一键接入 agent_graph.bind_tools 主链路。

    转换后的工具对本项目的 LangGraph 图完全透明：retrieve→agent⇄tools 流程
    无需任何改动就能编排"进程外"工具，这正是 MCP 标准化的价值所在。
    （生产替换方案：langchain-mcp-adapters 的 load_mcp_tools 一行等效实现。）
    """
    client = McpClient(base_url)
    descriptors = await client.list_tools()

    tools: list[BaseTool] = []
    for desc in descriptors:
        args_model = _schema_to_pydantic(desc)

        def _make_invoker(mcp: McpClient, tool_name: str):  # 工厂闭包：签名由 StructuredTool 在运行期推断
            """闭包捕获各自的 client/tool 名，避免循环变量 Late-Binding 陷阱
            （对应 Java lambda 捕获局部变量的 final 语义）。"""

            async def _invoke(**kwargs: Any) -> str:
                return await mcp.call_tool(tool_name, kwargs)

            return _invoke

        invoker = _make_invoker(McpClient(base_url), desc.name)
        # args_schema 缺失时退化为无参工具（inputSchema 为空的服务端实现仍可用）
        tools.append(
            StructuredTool.from_function(coroutine=invoker, name=desc.name, description=desc.description, args_schema=args_model)
            if args_model
            else StructuredTool.from_function(coroutine=invoker, name=desc.name, description=desc.description)
        )
    return tools


# ---------------------------------------------------------------------------
# 手动冒烟入口：uv run python src/finrag/mcp_client.py
# 前置：.env 配置 MCP_SERVER_URL（指向 Java 版 MCP Server，见文件头注释）
# ---------------------------------------------------------------------------
async def demo() -> None:
    """打印工具目录并对第一个工具发一次空参试调，验证端到端连通性。"""
    if not MCP_SERVER_URL:
        print("未配置 MCP_SERVER_URL（.env 中设置，参考 .env.example），demo 跳过。")
        return

    client = McpClient(MCP_SERVER_URL)
    descriptors = await client.list_tools()
    if not descriptors:
        print("服务端未暴露任何工具。")
        return

    first = descriptors[0]
    print(f"正在试调第一个工具: {first.name}({list(first.input_schema.get('properties', {}).keys())})")
    result = await client.call_tool(first.name, {})
    print(f"调用结果:\n{result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo())
