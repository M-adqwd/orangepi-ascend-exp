#!/usr/bin/env bash
# 启动 Orange Pi 交互式 SSH 桥接（前台运行，Ctrl+C 退出）
PY="$(command -v python3 || command -v python)"
cd "$(dirname "$0")" || exit 1

"$PY" opi_bridge.py
