#!/bin/bash
# 停止实时检测服务
PIDS=$(pgrep -f '[c]am_stream_server')
if [ -z "$PIDS" ]; then
    echo "服务未在运行"
    exit 0
fi
kill $PIDS 2>/dev/null
sleep 1
LEFT=$(pgrep -f '[c]am_stream_server')
if [ -n "$LEFT" ]; then
    kill -9 $LEFT 2>/dev/null
    echo "已强制停止"
else
    echo "已停止 (PID $(echo $PIDS | tr '\n' ' '))"
fi
