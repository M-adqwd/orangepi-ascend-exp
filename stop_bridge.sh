#!/usr/bin/env bash
# 停止 Orange Pi 交互式 SSH 桥接
cd "$(dirname "$0")" || exit 1
if [ -f opi_bridge.pid ]; then
  PID=$(cat opi_bridge.pid)
  kill "$PID" 2>/dev/null && echo "已停止桥接 PID=$PID" || echo "进程 $PID 已不存在"
  rm -f opi_bridge.pid
else
  echo "没有 pid 文件，桥接未运行"
fi
