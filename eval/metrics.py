"""
eval/metrics.py —— 检索质量量化指标（Hit@k / MRR@k）
=================================================================
用途：把 golden set 上的一次检索结果转化为可对比的数值，
     作为"优化依据"与"回归门禁"。面试讲解时可直接引用这些数字。

两套命中口径（避免"文档级命中但上下文里没有答案"的假阳性）：
- 文档级（hit@k / mrr@k）：Top-k 是否命中"期望来源文档"。
  衡量"能否把答案所在的文档召回"。
- 答案级（ans_hit@k / ans_mrr@k）：期望答案片段是否真实出现在
  召回 chunk 的文本中。衡量"喂给 LLM 的上下文里，答案是否真的出现"。
  后者更严格：命中了文档，但命中的 chunk 不含答案，仍算失败。
"""
from pathlib import Path

# 检索结果块的形态：(source 短文件名, chunk 文本)
Chunk = tuple[str, str]


def normalize_source(source: str) -> str:
    """把检索返回的 source 归一化为短文件名。

    检索器返回的 source 可能是完整路径（如 .../data/2024年年报摘要-华信科技.txt），
    而 golden set 里存的是短文件名，统一按 basename 比较。
    """
    return Path(source).name


def _norm(text: str) -> str:
    """归一化文本：去掉所有空白字符（换行/空格）。

    分块时 RecursiveCharacterTextSplitter 保留原文换行，同一句话可能被断成
    "每季度至少召开\n一次会议"，直接子串匹配会失配；去掉空白后即可命中。
    对"96.8 亿元"这类带空格数字无副作用。
    """
    return "".join(text.split())


def hit_at_k(pred_sources: list[str], expected_sources: list[str], k: int) -> int:
    """文档级 Hit@k：Top-k 内是否命中任一期望来源文档（返回 0/1）。"""
    top = {normalize_source(s) for s in pred_sources[:k]}
    return int(any(Path(e).name in top for e in expected_sources))


def mrr_at_k(pred_sources: list[str], expected_sources: list[str], k: int) -> float:
    """文档级 MRR@k：期望来源文档在 Top-k 内首次出现的排名倒数。"""
    expected = {Path(e).name for e in expected_sources}
    for rank, source in enumerate(pred_sources[:k], start=1):
        if normalize_source(source) in expected:
            return 1.0 / rank
    return 0.0


def answer_hit_at_k(pred_chunks: list[Chunk], answer: str | None, k: int) -> int:
    """答案级 Hit@k：期望答案片段是否出现在某个召回 chunk 的文本中。"""
    if not answer:
        return 0
    target = _norm(answer)
    for _source, text in pred_chunks[:k]:
        if target in _norm(text):
            return 1
    return 0


def answer_mrr_at_k(pred_chunks: list[Chunk], answer: str | None, k: int) -> float:
    """答案级 MRR@k：含答案片段的 chunk 首次出现的排名倒数。"""
    if not answer:
        return 0.0
    target = _norm(answer)
    for rank, (_source, text) in enumerate(pred_chunks[:k], start=1):
        if target in _norm(text):
            return 1.0 / rank
    return 0.0


def evaluate_batch(preds: list[list[Chunk]], goldens: list[dict],
                   ks: tuple[int, ...] = (1, 3, 5),
                   k_limit: int = 5) -> dict[str, float]:
    """批量计算一组策略的两套指标。

    参数：
        preds:   每条 query 的检索结果块列表（已按相关性降序，元素为 (source, text)）
        goldens: golden set 条目列表（须含 expected_sources 与 answer 字段）
        ks:      要报告的 Hit@k 集合
        k_limit: 评估用的排序截断深度（检索时按此值召回）
    返回：{'hit@1', 'hit@3', ..., 'mrr@5'（文档级）,
          'ans_hit@1', 'ans_hit@3', ..., 'ans_mrr@5'（答案级）}
    """
    total = len(goldens)
    if total == 0:
        return {}

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = sum(
            hit_at_k([s for s, _t in p], g.get("expected_sources", []), k)
            for p, g in zip(preds, goldens)
        ) / total
        metrics[f"ans_hit@{k}"] = sum(
            answer_hit_at_k(p, g.get("answer"), k)
            for p, g in zip(preds, goldens)
        ) / total

    metrics[f"mrr@{k_limit}"] = sum(
        mrr_at_k([s for s, _t in p], g.get("expected_sources", []), k_limit)
        for p, g in zip(preds, goldens)
    ) / total
    metrics[f"ans_mrr@{k_limit}"] = sum(
        answer_mrr_at_k(p, g.get("answer"), k_limit)
        for p, g in zip(preds, goldens)
    ) / total
    return metrics
