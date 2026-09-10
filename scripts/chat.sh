#!/usr/bin/env bash
# 交互式多轮对话（流式输出）
# 用法：./scripts/chat.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python main.py
