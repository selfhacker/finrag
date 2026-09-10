#!/usr/bin/env bash
# 三路检索策略对比实验（参数原样透传给 eval/compare.py）
# 用法：
#   ./scripts/eval.sh                      # 全量 golden set（较慢）
#   ./scripts/eval.sh --sample 8           # 抽样快速跑
#   ./scripts/eval.sh --skip-rerank        # 不跑 LLM 重排（省时省钱）
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python eval/compare.py "$@"
