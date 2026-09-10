"""
config.py —— 全局配置中心
=================================================================
职责：统一从 .env 读取 LLM / Embedding / RAG / 并发等全部运行参数，
      并提供工厂函数 get_llm() / get_embeddings()。

对 Java 工程师的类比说明：
- 这里的模块级函数相当于 Java 的 `@Configuration` + `@Bean`，
  即"一个集中装配依赖的工厂类"。
- 通过环境变量切换 Provider，等价于 Spring 的
  `@Value("${...}")` + profile 机制。

切换 Provider 的扩展方式：
  1) LLM 换成 OpenAI 官方： 只需把 LLM_BASE_URL 改为 https://api.openai.com/v1，
     并填入 OpenAI 的 key；若想换回本地 Ollama，可改用 langchain-ollama 的
     OllamaLLM(model="qwen2.5:7b")，接口签名兼容。
  2) Embedding 换成 OpenAI：同理改 EMBEDDING_PROVIDER 与对应 base_url/key。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# 项目根目录：src/finrag/config.py -> 项目根（与 tools.py 的 PROJECT_ROOT 同一约定）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 加载项目根目录下的 .env（锚定项目根路径，避免依赖运行时 CWD 而加载不到密钥）
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str | None, default: str) -> str:
    """把相对路径解析为基于项目根的绝对路径，避免依赖运行时 CWD。

    原因：若程序从非项目根目录启动（如 IDE 工作目录未设对），`./data`、`./chroma_db`
    会解析到错误位置——chmod_db 会在错误位置建出空库、data 目录找不到而报错。
    统一锚定到项目根后，无论从哪启动都能命中正确的数据目录。
    """
    raw = (value or default).strip()
    p = Path(raw)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def _get_bool(name: str, default: bool = False) -> bool:
    """把环境变量字符串解析为布尔值，缺失时返回默认值。

    注：Python 中没有 Java 的 `@Value(required=false)`，这里用手写解析函数兜底。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """把环境变量字符串解析为 int，缺失或非法时返回默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# RAG 参数
# ---------------------------------------------------------------------------
DATA_DIR: str = _resolve_path(os.getenv("DATA_DIR"), "./data")
CHROMA_PERSIST_DIR: str = _resolve_path(os.getenv("CHROMA_PERSIST_DIR"), "./chroma_db")
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)  # 分块大小（字符）
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 50)  # 相邻块重叠字符数
RETRIEVAL_TOP_K: int = _get_int("RETRIEVAL_TOP_K", 4)  # 检索返回的候选块数
ENABLE_RE_RANK: bool = _get_bool("ENABLE_RE_RANK", False)  # 小语料实测负优化，默认关闭（见 README 第六节）

# ---------------------------------------------------------------------------
# 并发控制
# ---------------------------------------------------------------------------
# 全局并发信号量上限，对应 Java 中 `new Semaphore(5)`，见 main.py
MAX_CONCURRENCY: int = _get_int("MAX_CONCURRENCY", 5)

# ---------------------------------------------------------------------------
# MCP Client（可选）：远程 MCP Server 端点
# ---------------------------------------------------------------------------
# 未配置（空串）则 mcp_client 的 demo 自动跳过；Java 版服务端写法参见
# src/finrag/mcp_client.py 文件头注释。
MCP_SERVER_URL: str = (os.getenv("MCP_SERVER_URL") or "").strip()

# ---------------------------------------------------------------------------
# LLM 配置（默认 DeepSeek）
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
# 评估判官（LLM-as-Judge）专用模型，默认非思考版：
# 裁判要便宜、快、确定性输出——思考模型慢且贵，且不支持强制 tool_choice，
# 会导致 with_structured_output 的 function_calling 方式 400。
LLM_JUDGE_MODEL: str = os.getenv("LLM_JUDGE_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# Embedding 配置（默认阿里云百炼 DashScope，text-embedding-v3）
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "dashscope")
DASHSCOPE_BASE_URL: str = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_EMBEDDING_MODEL: str = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")


def _require_key(key: str, name: str) -> str:
    """密钥缺失时给出清晰错误，避免运行时出现难以排查的 401。

    对应 Java 中的 `Assert.notNull(key, "...")` 前置校验。
    """
    if not key or key.startswith("sk-your-"):
        raise RuntimeError(
            f"缺少环境变量 {name}，请在项目根目录创建 .env 并填写真实密钥"
            f"（参考 .env.example）。当前值: {key!r}"
        )
    return key


def get_llm() -> ChatOpenAI:
    """创建 LLM 实例。

    底层是 langchain-openai 的 ChatOpenAI，对 DeepSeek 来说：
    - base_url 指向 DeepSeek 的 OpenAI 兼容端点
    - 支持 function calling / tool calling / streaming

    注意：本项目中 LLM 需要支持 tool calling（ReAct 循环依赖它）。
    若换成不含工具调用能力的模型，tools 节点将永远不会被触发。
    """
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=SecretStr(_require_key(LLM_API_KEY, "LLM_API_KEY")),
        model=LLM_MODEL,  # 新版 langchain-openai 推荐写法，等价于旧的 model_name
        temperature=LLM_TEMPERATURE,
        max_retries=2,  # 内置请求级重试（网络抖动时自动重试）
        timeout=60,  # 请求超时（秒），超时抛异常由上层重试/熔断处理
    )


def get_judge_llm() -> ChatOpenAI:
    """创建评估判官 LLM 实例（LLM-as-Judge 专用，见 eval/faithfulness.py）。

    与业务 LLM（get_llm）解耦的原因——裁判选型三原则：便宜、快、确定性：
    - 温度固定 0（可复现、可当回归门禁）；
    - 默认非思考模型（LLM_JUDGE_MODEL）：思考模型的思维链对判官无用反增成本延迟，
      且不支持强制 tool_choice，会使 with_structured_output 的
      function_calling 方式直接 400。
    注意不要用 model_copy(update={"model": ...}) 从 get_llm() 派生：
    langchain-openai 的 model 字段带 alias（model_name），update 键不命中，
    会静默沿用业务模型（已踩坑验证）。
    """
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=SecretStr(_require_key(LLM_API_KEY, "LLM_API_KEY")),
        model=LLM_JUDGE_MODEL,
        temperature=0.0,
        max_retries=2,
        timeout=60,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """创建 Embedding 模型实例（阿里云百炼 text-embedding-v3）。

    OpenAI 兼容接口，因此复用 langchain-openai 的 OpenAIEmbeddings：
    - base_url 指向 DashScope 的 compatible-mode 端点
    - 输入向量维度固定为 1024（text-embedding-v3 默认）
    """
    return OpenAIEmbeddings(
        base_url=DASHSCOPE_BASE_URL,
        api_key=SecretStr(_require_key(DASHSCOPE_API_KEY, "DASHSCOPE_API_KEY")),
        model=DASHSCOPE_EMBEDDING_MODEL,
        # text-embedding-v3 用 `dimensions` 显式控制向量维度，保证索引可复用
        dimensions=1024,
        # 关键：禁用 langchain-openai 默认的 tiktoken 长度安全处理。
        # 默认开启时它会把文本 tokenize 成 ID 数组（input=[[86461, ...]]）发给服务端，
        # OpenAI 官方接受，但 DashScope 的兼容模式只认原始字符串，会报
        # 400 "contents is neither str nor list of str"。关闭后直接发送原文。
        check_embedding_ctx_length=False,
    )
