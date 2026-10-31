"""Orange Pi 交互式 shell 客户端。

用法:
  opi.py ls -la                 执行一条命令（自动回传退出码）
  opi.py "cd /opt && ls"        支持引号包裹的复合命令
  opi.py --raw python3          进入交互式程序（如 python / top / vim），不做包装
  opi.py --peek                 只取回当前残留输出，不发送任何命令
  opi.py --timeout 120 make     自定义超时秒数
"""
import argparse
import json
import os
import socket
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE, "opi_config.json")

with open(CFG_PATH, encoding="utf-8") as f:
    CFG = json.load(f)

ap = argparse.ArgumentParser(add_help=True)
ap.add_argument("cmd", nargs="*", help="要执行的命令")
ap.add_argument("--raw", action="store_true", help="原样发送，不包装退出码")
ap.add_argument("--peek", action="store_true", help="只取回输出，不发送命令")
ap.add_argument("--join", action="store_true", help="结束标记挂同一行，适合 ATC 等长时间、不读 stdin 的命令")
ap.add_argument("--timeout", type=float, default=float(CFG.get("default_timeout", 20)))
a = ap.parse_args()

payload = {
    "cmd": " ".join(a.cmd),
    "raw": a.raw,
    "peek": a.peek,
    "join": a.join,
    "timeout": a.timeout,
}

port = int(CFG.get("bridge_port", 7831))
try:
    s = socket.create_connection(("127.0.0.1", port), timeout=a.timeout + 20)
except OSError:
    sys.exit(f"连不上桥接服务 127.0.0.1:{port}。先运行 start_bridge.sh 启动它。")

s.sendall(json.dumps(payload).encode("utf-8"))
s.shutdown(socket.SHUT_WR)

buf = b""
try:
    while True:
        d = s.recv(65536)
        if not d:
            break
        buf += d
except (ConnectionResetError, ConnectionAbortedError):
    if not buf:
        buf = "[连接被重置] 桥接端未返回内容，命令可能仍在后台执行；用 --peek 查看".encode("utf-8")
try:
    s.close()
except OSError:
    pass

sys.stdout.write(buf.decode("utf-8", "replace"))
if not buf.endswith(b"\n"):
    sys.stdout.write("\n")
