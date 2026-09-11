"""
eval/compare.py —— 三路检索策略对比实验
=================================================================
对比对象：
    策略一  dense           纯 Dense 向量检索（基线）
    策略二  hybrid          Dense + BM25 混合检索 + RRF 融合
    策略三  hybrid+rerank   混合检索 + RRF + LLM 精排

双指标输出（见 metrics.py）：
    [A] 文档级命中：Top-k 内是否命中期望来源文档
    [B] 答案级命中：期望答案片段是否真实出现在召回 chunk 文本中
        —— 更严格，能暴露"召回对了文档、但 chunk 里没答案"的假阳性。

用法：
    uv run python eval/compare.py                 # 全量 golden set
    uv run python eval/compare.py --sample 8      # 抽样快速跑
    uv run python eval/compare.py --skip-rerank   # 不跑 LLM 重排（省时省钱）

注意：hybrid+rerank 每条 golden 会触发一次 LLM 打分调用（DeepSeek），
全量 35 条预计耗时数分钟。
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

# 允许直接 import eval.metrics（本脚本位于 eval/ 目录）
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
# pyrefly: ignore[missing-import] 原因：metrics 模块通过运行时 sys.path 注入，
# 静态解析（按 src/ 布局）无法发现，属运行时导入技巧，非真实缺失。
from metrics import (  # noqa: E402
    Chunk,
    answer_hit_at_k,
    answer_mrr_at_k,
    evaluate_batch,
    hit_at_k,
    mrr_at_k,
)

from finrag.rag_retriever import FinancialRetriever  # noqa: E402

GOLDEN_SET_FILE = EVAL_DIR / "golden_set.yaml"

logger = logging.getLogger("eval")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("httpx", "httpcore", "urllib3", "openai", "chromadb.telemetry", "rag", "trace"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def load_golden_set() -> list[dict]:
    if not GOLDEN_SET_FILE.exists():
        raise FileNotFoundError(f"找不到 golden set：{GOLDEN_SET_FILE}")
    with GOLDEN_SET_FILE.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("golden set 格式错误：应为非空 YAML 列表")
    return data


async def retrieve_strategy(retriever: FinancialRetriever, strategy: str,
                            query: str, k: int) -> list[Chunk]:
    """按策略检索，返回 (source, text) 块列表（供文档级与答案级两套判定）。"""
    if strategy == "dense":
        chunks = await retriever.retrieve_dense(query, k=k)
    elif strategy == "hybrid":
        chunks = await retriever.retrieve(query, k=k, rerank=False)
    else:  # hybrid+rerank
        chunks = await retriever.retrieve(query, k=k, rerank=True)
    return [(c.source, c.text) for c in chunks]


def pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def fmt_row(name: str, metrics: dict, k_limit: int, ans: bool) -> str:
    """打印一行指标；ans=False 取文档级，ans=True 取答案级。"""
    prefix = "ans_" if ans else ""
    hits = "  |  ".join(pct(metrics[f"{prefix}hit@{k}"]) for k in (1, 3, 5))
    return f"  {name:<18} |  {hits}  |  {metrics[f'{prefix}mrr@{k_limit}']:.3f}"


def print_metric_table(title: str, results: dict[str, list[list[Chunk]]],
                       strategies: list[str], goldens: list[dict], k: int,
                       ans: bool) -> None:
    """打印一张指标表（文档级或答案级）。"""
    kind = "答案级命中：期望答案片段是否出现在召回 chunk 文本中" \
        if ans else "文档级命中：Top-k 内是否命中期望来源文档"
    print(f"  [{title}] {kind}")
    print(f"  {'策略':<18} |  Hit@1  |  Hit@3  |  Hit@5  |  MRR@{k}")
    print("  " + "-" * 58)
    for strategy in strategies:
        m = evaluate_batch(results[strategy], goldens, k_limit=k)
        print(fmt_row(strategy, m, k, ans=ans))
    print()


async def run_comparison(goldens: list[dict], k: int, skip_rerank: bool) -> None:
    retriever = FinancialRetriever()
    retriever.ensure_index()
    print(f"\n  Golden set 共 {len(goldens)} 条，检索深度 Top-{k}\n")

    strategies = ["dense", "hybrid"] + ([] if skip_rerank else ["hybrid+rerank"])
    # 用空列表占位"未完成"，避免 Optional 导致后续类型收窄困难（PyCharm/pyrefly 兼容）
    results: dict[str, list[list[Chunk]]] = {
        s: [[] for _ in goldens] for s in strategies
    }

    # 并发执行各条 golden（rerank 的 LLM 调用是真实异步 IO，并发能大幅缩短总耗时）。
    # Semaphore 限流：防止并发打爆上游 API（对应 main.py 的并发控制思路）。
    semaphore = asyncio.Semaphore(4)

    async def process(i: int, g: dict) -> None:
        async with semaphore:
            for strategy in strategies:
                results[strategy][i] = await retrieve_strategy(
                    retriever, strategy, g["query"], k
                )
        logger.info("[eval] %s/%s %s 完成", i + 1, len(goldens), g["query"][:24])

    await asyncio.gather(*(process(i, g) for i, g in enumerate(goldens)))
    assert all(r for s in strategies for r in results[s]), "部分检索结果缺失"

    # ---- 主表：文档级 + 答案级两套指标 ----
    print("=" * 60)
    print(f"  三路检索策略对比（golden set = {len(goldens)} 条, Top-{k}）")
    print("=" * 60)
    print_metric_table("A 文档级", results, strategies, goldens, k, ans=False)
    print_metric_table("B 答案级", results, strategies, goldens, k, ans=True)

    # ---- 分组明细：按用例类型（direct/semantic/fuzzy），取答案级 Hit@3 ----
    type_order = ["direct", "semantic", "fuzzy"]
    print("  按用例类型【答案级】Hit@3 分组明细：")
    header = "  " + f"{'类型':<14} | " + " | ".join(f"{s:<14}" for s in strategies)
    print(header)
    print("  " + "-" * 58)
    for t in type_order:
        # 关键：按原索引收集该类型对应的检索结果，避免与全量结果错位
        idx = [i for i, g in enumerate(goldens) if g.get("type", "direct") == t]
        if not idx:
            continue
        row = [f"{t} (n={len(idx)})"]
        for strategy in strategies:
            preds_t = [results[strategy][i] for i in idx]
            goldens_t = [goldens[i] for i in idx]
            m = evaluate_batch(preds_t, goldens_t, ks=(3,), k_limit=k)
            row.append(f"{pct(m['ans_hit@3']):<14}")
        print("  " + " | ".join(row))

    # ---- 逐条明细：重点暴露"文档命中但答案未命中"的假阳性 ----
    best = strategies[-1]
    print(f"\n  逐条结果（best = {best}，doc=文档级✓，ans=答案级✓，均按 Top-3 判定）:")
    false_positives: list[str] = []
    for i, g in enumerate(goldens):
        chunks = results[best][i]
        sources = [s for s, _t in chunks]
        expected = g["expected_sources"]
        answer = g.get("answer")

        d3 = hit_at_k(sources, expected, 3)
        a3 = answer_hit_at_k(chunks, answer, 3)
        dm = mrr_at_k(sources, expected, k)
        am = answer_mrr_at_k(chunks, answer, k)
        flag = "✓✓" if (d3 and a3) else "✓✗" if d3 else "✗✗"
        if d3 and not a3:
            false_positives.append(g["query"][:26])
        print(f"  [{flag}] doc_mrr={dm:.3f} ans_mrr={am:.3f}  {g['type']:<8} {g['query'][:30]}")

    if false_positives:
        print(f"\n  ⚠ {len(false_positives)} 条『文档命中但答案未命中』"
              f"（文档被召回，但 Top-{k} 里没有含答案的 chunk，需查分块/召回）:")
        for q in false_positives:
            print(f"    - {q}")


def main() -> None:
    parser = argparse.ArgumentParser(description="三路检索策略对比实验")
    parser.add_argument("--sample", type=int, default=0,
                        help="只跑前 N 条 golden（0 表示全量）")
    parser.add_argument("--k", type=int, default=5, help="检索深度 / 评估截断")
    parser.add_argument("--skip-rerank", action="store_true",
                        help="跳过 LLM 重排策略（节省时间与费用）")
    args = parser.parse_args()

    goldens = load_golden_set()
    if args.sample:
        goldens = goldens[: args.sample]
        print(f"\n  [提示] --sample={args.sample}，仅评估前 {len(goldens)} 条")

    asyncio.run(run_comparison(goldens, k=args.k, skip_rerank=args.skip_rerank))


if __name__ == "__main__":
    main()
