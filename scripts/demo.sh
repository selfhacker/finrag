#!/usr/bin/env bash
# 并发限流演示（8 个并发请求，信号量上限 5）
# 用法：./scripts/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python main.py --demo
