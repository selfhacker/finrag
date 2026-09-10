"""
eval/faithfulness.py —— 生成质量评估：LLM-as-Judge 之 faithfulness（忠实度）
================================================================
定位：检索评估（compare.py，确定性指标）之上的 L2 层——评估"生成的答案
有没有脱离召回上下文胡编"（幻觉检测）。

核心思想（为什么只评 faithfulness、不评"答得好不好"）：
- 把答案拆成一条条"原子断言"，逐条问 judge：这句话能否从上下文推出来？
  → 这是事实核查，接近客观判定，可当回归门禁（类比单元测试的 assert）。
- 而"答案通不通顺、详不详细"是主观偏好（类比代码评审打星），
  judge 换个模型分数就漂，且答案写得越顺、编得越像真的，主观分越高，
  恰恰会掩盖幻觉 —— 所以不进指标体系。

打分公式：faithfulness = 被支撑的断言数 / 断言总数
  1.0 = 全部有出处；0.5 = 一半是编的；0 = 全是编的。

Judge 的两条铁律（写进系统提示词）：
1. 只准依据【上下文】判定，禁止用模型自己的知识补证 —— 否则"编得合理"
   的内容会被误判为 supported，指标失效；
2. 数字/日期必须精确匹配 —— 幻觉最常见的形态就是"数值漂移"。

工程要点：
- judge 温度固定 0（裁判要确定性，可复现）；
- 结构化输出走 with_structured_output(pydantic)（与 tools.py 的
  args_schema 一脉相承：用 pydantic 当"输出 DTO + 校验器"）；
- judge 调用失败【绝不抛异常】—— 与 tools.py 同款降级哲学：
  返回 score=-1 的 error 结果，让上层决定跳过或重跑，不炸评估脚本。

对 Java 工程师的类比：
- 原子断言核查 ≈ 单元测试逐条 assert；
- FaithfulnessAudit（judge 的结构化输出）≈ 接口反序列化 DTO；
- score=-1 的 error 降级 ≈ 依赖服务挂了返回兜底响应而不是抛异常炸链路。

用法：
    uv run python eval/faithfulness.py          # 跑 3 个内置样例（3 次 judge 调用，成本可忽略）
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import cast

# 允许直接 import finrag（本脚本位于 eval/ 目录，运行时注入项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from finrag.config import get_judge_llm

logger = logging.getLogger("eval.faithfulness")


# ---------------------------------------------------------------------------
# 输出 DTO：judge 的结构化输出 + 对外结果
# ---------------------------------------------------------------------------
class AssertionVerdict(BaseModel):
    """单条原子断言的核查结论（judge 输出 DTO 的元素）。

    对应 Java 里的一个校验结果条目：{断言内容, 是否通过, 失败原因}。
    """

    statement: str = Field(description="从答案中拆出的原子断言，一句话只表达一个事实")
    supported: bool = Field(description="该断言是否被上下文直接支撑（true=有出处）")
    reason: str = Field(description="一句话判定理由：指出上下文中的依据，或说明缺失什么")


class FaithfulnessAudit(BaseModel):
    """judge 的完整结构化输出（with_structured_output 的目标模型）。"""

    verdicts: list[AssertionVerdict] = Field(description="答案拆解出的全部断言及逐条核查结论")


class FaithfulnessResult(BaseModel):
    """对外暴露的评估结果。

    score = supported / total，取值 [0, 1]；
    judge 调用失败时 score=-1 且 error 非空（降级结果，绝不向上抛异常）。
    """

    score: float = Field(description="忠实度得分：被支撑断言占比；judge 失败时为 -1")
    total: int = Field(description="拆解出的原子断言总数")
    supported: int = Field(description="被上下文直接支撑的断言数")
    verdicts: list[AssertionVerdict] = Field(default_factory=list, description="逐条核查明细")
    error: str | None = Field(default=None, description="judge 失败原因（成功时为 None）")


# ---------------------------------------------------------------------------
# Judge 提示词：两条铁律写死在系统提示里
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """你是严格的事实核查员。规则：
1. 只准依据【上下文】判定断言真伪，严禁使用你自己的知识补证——上下文没有依据就是 not supported；
2. 数字、日期、金额必须精确匹配，数值不同即 not supported；
3. 把答案拆成最少的原子断言（每条只表达一个事实，不要合并）；
4. 对每条断言输出：statement（断言原文）、supported（布尔）、reason（一句话理由）。"""

JUDGE_USER_TEMPLATE = """【上下文】
{context}

【答案】
{answer}

请把答案拆成原子断言并逐条核查。"""

# 复用全局 LLM 工厂，但用 bind 覆盖温度：裁判必须确定性（可复现、可回归）
# bind ≈ Java 里从原型 Bean 派生一个定制配置实例，不影响原 get_llm() 的默认值


async def evaluate_faithfulness(
    question: str,
    context: str,
    answer: str,
    judge_llm: Runnable | None = None,
) -> FaithfulnessResult:
    """评估单条答案的忠实度。

    question 仅辅助 judge 理解语境；faithfulness 的判定只看 context 与 answer
    ——答案是否忠于上下文，与"问题问得好不好"无关。
    judge_llm 可注入替代实现（测试友好），默认 DeepSeek + 结构化输出。
    """
    if judge_llm is None:
        # judge 模型与业务模型解耦（get_judge_llm，默认 deepseek-chat 非思考版）：
        # 裁判选型三原则——便宜、快、确定性。思考模型（如 deepseek-reasoner /
        # deepseek-v4-flash）思维链对判官无用反增成本延迟，且不支持强制 tool_choice，
        # 会让 with_structured_output 的 function_calling 方式直接 400。
        # method="function_calling" 是跨 Provider 兼容的关键：
        # 新版 langchain-openai 默认走 OpenAI 专有的 response_format(json_schema)，
        # DeepSeek 不支持（400: This response_format type is unavailable now）；
        # 改走 function calling（tools 参数）承载 JSON Schema——本项目 ReAct
        # 已验证 DeepSeek 的 tool calling 可用，同一条能力复用。
        judge_llm = get_judge_llm().with_structured_output(
            FaithfulnessAudit, method="function_calling"
        )

    messages = [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("user", JUDGE_USER_TEMPLATE.format(context=context, answer=answer)),
    ]
    try:
        # cast 而非标注：with_structured_output 运行时确实实例化 FaithfulnessAudit，
        # 但 Runnable.ainvoke 的静态返回类型追不到该信息，需显式收窄（PyCharm/pyrefly 双认）
        audit = cast(FaithfulnessAudit, await judge_llm.ainvoke(messages))
    except Exception as exc:  # 兜底降级：评估器自身故障不炸脚本（同工具铁律）
        logger.exception("faithfulness judge 调用失败")
        return FaithfulnessResult(score=-1.0, total=0, supported=0, error=str(exc))

    total = len(audit.verdicts)
    supported = sum(1 for v in audit.verdicts if v.supported)
    # total=0 属于 judge 异常输出（答案再差也至少该拆出一条），按 0 分防御
    score = (supported / total) if total > 0 else 0.0
    return FaithfulnessResult(score=score, total=total, supported=supported, verdicts=audit.verdicts)


def render_result(title: str, r: FaithfulnessResult) -> str:
    """把结果渲染成可读文本（demo 与将来批量评估共用）。"""
    if r.error:
        return f"[{title}] judge 失败（降级）：{r.error}"
    lines = [f"[{title}] faithfulness = {r.supported}/{r.total} = {r.score:.2f}"]
    mark = {True: "✓ 支撑", False: "✗ 无出处"}
    for v in r.verdicts:
        lines.append(f"    {mark[v.supported]} | {v.statement}（{v.reason}）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 内置 demo：三个样例覆盖 忠实 / 半幻觉 / 全编造
# 上下文故意只包含两条事实，让"编得像不像"与"有没有出处"形成反差
# ---------------------------------------------------------------------------
CONTEXT = "2024年，华信科技实现营业收入12.5亿元，同比增长15%；全年研发投入1.2亿元，占营收比重9.6%。"

DEMO_CASES: list[dict[str, str]] = [
    {
        "title": "样例A：忠实答案（预期≈1.0）",
        "context": CONTEXT,
        "answer": "华信科技2024年实现营业收入12.5亿元，同比增长15%。",
    },
    {
        "title": "样例B：半幻觉（预期≈0.5，句子通顺但后半段无出处）",
        "context": CONTEXT,
        "answer": "华信科技2024年营收12.5亿元，同比增长15%，经营状况十分良好，预计明年营收将突破20亿元。",
    },
    {
        "title": "样例C：完全编造（预期=0，数值全部漂移）",
        "context": CONTEXT,
        "answer": "华信科技2024年营收30亿元，同比下降8%，净利润率高达25%。",
    },
]


async def demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("=" * 72)
    print("faithfulness 评估 demo：judge = DeepSeek（温度 0，结构化输出）")
    print(f"固定上下文：{CONTEXT}")
    print("=" * 72)
    for case in DEMO_CASES:
        r = await evaluate_faithfulness(question="", context=case["context"], answer=case["answer"])
        print(render_result(case["title"], r))
        print("-" * 72)


if __name__ == "__main__":
    asyncio.run(demo())
