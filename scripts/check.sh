#!/usr/bin/env bash
# 质量门禁：代码风格（ruff）+ 类型检查（pyrefly）
# 用法：./scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> ruff check（代码风格）"
uv run ruff check main.py src eval

echo "==> pyrefly check（类型检查）"
uv run pyrefly check

echo "==> 全部通过 ✓"
