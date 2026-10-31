#!/bin/bash
# 启动 USB 摄像头 + YOLOv8n 实时人员检测（MJPEG 网页直播）
# 用法: bash start_stream.sh [端口]   默认 8000
PORT="${1:-8000}"
DIR="$HOME/exp/exp3"
cd "$DIR" || { echo "目录不存在: $DIR"; exit 1; }

# ---- 环境变量：不设这几项，python3 会找不到 pyACL / ais_bench ----
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export PATH=/usr/local/miniconda3/bin:$HOME/exp/bin:$PATH
export TE_PARALLEL_COMPILER=1 MAX_COMPILE_CORE_NUMBER=1

# ---- 停掉旧实例（[c] 技巧：避免 pkill 匹配到自己这条命令） ----
OLD=$(pgrep -f '[c]am_stream_server')
if [ -n "$OLD" ]; then
    kill $OLD 2>/dev/null
    sleep 1
    echo "已停止旧实例: $(echo $OLD | tr '\n' ' ')"
fi

# ---- 检查模型 ----
if [ ! -f yolov8n.om ]; then
    echo "缺少模型文件 $DIR/yolov8n.om"
    exit 1
fi

# ---- 后台启动，脱离当前终端 ----
rm -f stream.log
setsid nohup python3 cam_stream_server.py --om yolov8n.om --port "$PORT" \
    >> stream.log 2>&1 < /dev/null &
disown

sleep 2
PID=$(pgrep -f '[c]am_stream_server' | head -1)
if [ -z "$PID" ]; then
    echo "启动失败，日志如下："
    cat stream.log
    exit 1
fi

echo "已启动 (PID $PID)，端口 $PORT"
# 优先显示局域网地址（hostname -I 第一个常是 docker 网桥 172.17.0.1）
IP=$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.' | head -1)
[ -z "$IP" ] && IP=$(hostname -I | awk '{print $1}')
echo "浏览器访问: http://$IP:$PORT"
echo "日志: $DIR/stream.log"
