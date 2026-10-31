"""常驻交互式 SSH 桥接服务。

在本地开一个 TCP 端口，所有命令都转发到 Orange Pi 上**同一个**持久 shell 会话里执行。
因此 cd、export、后台进程、虚拟环境激活等状态会一直保留。

协议: 客户端连入 -> 发一个 JSON {"cmd": "...", "raw": bool, "timeout": 秒, "peek": bool}
      -> 半关闭写端 -> 服务端执行并回传输出 -> 关闭连接。
"""
import json
import os
import re
import socket
import socketserver
import sys
import threading
import time
import uuid

import paramiko

BASE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE, "opi_config.json")
LOG_PATH = os.path.join(BASE, "opi_bridge.log")
PID_PATH = os.path.join(BASE, "opi_bridge.pid")

ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[()][0-9A-Za-z]"
    r"|\x1b[=>]"
)


def log(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " " + str(msg) + "\n")
    except Exception:
        pass


def load_cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


def clean(text):
    text = text.replace("\r\n", "\n")
    text = ANSI_RE.sub("", text)
    return text


class Shell:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        self.chan = None
        self.connect()

    def connect(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        host = self.cfg["host"]
        port = int(self.cfg.get("port", 22))
        user = self.cfg["user"]
        log(f"连接 {user}@{host}:{port} ...")
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kw = dict(
            hostname=host, port=port, username=user,
            timeout=15, allow_agent=False, look_for_keys=False,
            banner_timeout=15, auth_timeout=15,
        )
        if self.cfg.get("key_file"):
            kw["key_filename"] = self.cfg["key_file"]
        else:
            kw["password"] = self.cfg.get("password", "")
        c.connect(**kw)
        c.get_transport().set_keepalive(30)
        self.client = c
        ch = c.invoke_shell(
            term=self.cfg.get("term", "xterm-256color"),
            width=int(self.cfg.get("width", 200)),
            height=int(self.cfg.get("height", 60)),
        )
        ch.settimeout(0.2)
        self.chan = ch
        self.drain(2.5)
        self.send_raw("export PS1=''\n")
        self.drain(0.8)
        self.send_raw("stty -echo\n")
        self.drain(0.8)
        log("交互式 shell 就绪")

    def alive(self):
        return self.client is not None and self.chan is not None and not self.chan.closed

    def send_raw(self, s):
        self.chan.send(s.encode("utf-8"))

    def drain(self, seconds):
        end = time.time() + seconds
        buf = b""
        while time.time() < end:
            if self.chan is None or self.chan.closed:
                break
            try:
                if self.chan.recv_ready():
                    d = self.chan.recv(65536)
                    if not d:
                        break
                    buf += d
                else:
                    time.sleep(0.05)
            except socket.timeout:
                continue
            except Exception:
                break
        return buf.decode("utf-8", "replace")


CFG = load_cfg()
SHELL = Shell(CFG)
LOCK = threading.Lock()


def run_command(cmd, raw, timeout, join=False):
    global SHELL
    if not SHELL.alive():
        log("会话已断开，重连中...")
        SHELL.connect()

    leftover = SHELL.drain(0.4)

    if raw:
        SHELL.send_raw(cmd if cmd.endswith("\n") else cmd + "\n")
        return clean(SHELL.drain(timeout)).strip("\n")

    if not cmd.strip():
        return clean(leftover).strip("\n")

    uid = uuid.uuid4().hex[:8]
    marker = f"__OPI_{uid}__"
    done_re = re.compile(re.escape(marker) + r":(-?\d+)")
    quiet = 0.6
    start = time.time()

    if join:
        SHELL.send_raw(f"{cmd}; echo {marker}:$?\n")
        marker_sent = True
        marker_deadline = start + timeout
    else:
        SHELL.send_raw(cmd + "\n")
        marker_sent = False
        marker_deadline = None

    buf = ""
    last_out = time.time()
    sudo_sends = 0

    while True:
        now = time.time()
        if marker_sent and now > marker_deadline:
            break
        if not marker_sent and now - start > timeout:
            break

        chunk = SHELL.drain(0.25)
        if chunk:
            buf += chunk
            last_out = time.time()

        if done_re.search(buf):
            break

        prompts = 0 if join else buf.lower().count("[sudo] password for")
        if (
            CFG.get("auto_sudo")
            and CFG.get("password")
            and prompts > sudo_sends
            and sudo_sends < 3
        ):
            SHELL.send_raw(CFG["password"] + "\n")
            sudo_sends += 1
            last_out = time.time()
            continue

        if not marker_sent and not join and time.time() - last_out > quiet:
            SHELL.send_raw(f"echo {marker}:$?\n")
            marker_sent = True
            marker_deadline = time.time() + 30

    buf = clean(buf)
    m = done_re.search(buf)
    if not m:
        partial = clean(leftover + buf).strip("\n")
        tail = f"\n\n[超时 {timeout}s，命令可能仍在运行或正在等待输入。用 `opi.py --peek` 继续取输出]"
        return (partial + tail) if partial else tail.strip("\n")

    code = m.group(1)
    body = buf[: m.start()]
    if cmd and body.startswith(cmd):
        body = body[len(cmd):]
    body = body.strip("\n")
    if leftover.strip():
        body = clean(leftover).strip("\n") + "\n" + body
    return f"{body}\n[退出码 {code}]" if body else f"[退出码 {code}，无输出]"


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw_bytes = self.rfile.read()
            try:
                req = json.loads(raw_bytes.decode("utf-8"))
            except Exception:
                req = {"cmd": raw_bytes.decode("utf-8", "replace")}
            cmd = req.get("cmd", "")
            raw = bool(req.get("raw", False))
            peek = bool(req.get("peek", False))
            join = bool(req.get("join", False))
            timeout = float(req.get("timeout", CFG.get("default_timeout", 20)))
            if not LOCK.acquire(timeout=max(10.0, timeout)):
                self.wfile.write(
                    "[桥接忙] 上一条命令仍占用会话，请稍后重试".encode("utf-8")
                )
                return
            try:
                if peek:
                    out = clean(SHELL.drain(timeout)).strip("\n")
                else:
                    out = run_command(cmd, raw, timeout, join)
            finally:
                LOCK.release()
            self.wfile.write(out.encode("utf-8", "replace"))
        except Exception as e:
            log(f"处理出错: {type(e).__name__}: {e}")
            try:
                self.wfile.write(f"[桥接错误] {type(e).__name__}: {e}".encode("utf-8"))
            except Exception:
                pass


if __name__ == "__main__":
    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))
    port = int(CFG.get("bridge_port", 7831))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    log(f"桥接服务监听 127.0.0.1:{port}")
    print(f"bridge listening on 127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
