"""在 Orange Pi 上执行单条命令（独立 SSH 通道，无持久 shell 状态，最稳）。

用法:
  opi_exec.py "<shell 命令>"
自动用 bash -lc 包裹，返回 stdout/stderr/退出码。
长时间任务请自行加 nohup + 日志，见 run_bg.sh 用法。
"""
import json
import os
import shlex
import sys

import paramiko

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "opi_config.json"), encoding="utf-8") as f:
    CFG = json.load(f)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(
        hostname=CFG["host"], port=int(CFG.get("port", 22)),
        username=CFG["user"], timeout=20, allow_agent=False, look_for_keys=False,
    )
    if CFG.get("key_file"):
        kw["key_filename"] = CFG["key_file"]
    else:
        kw["password"] = CFG.get("password", "")
    c.connect(**kw)
    return c


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = " ".join(sys.argv[1:])
    cann = "source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1; "
    wrap = (
        f"{cann}"
        "export PATH=/usr/local/miniconda3/bin:$HOME/exp/bin:$PATH; "
        "export TE_PARALLEL_COMPILER=1 MAX_COMPILE_CORE_NUMBER=1; "
        f"{cmd}"
    )
    wrapped = f"bash -c {shlex.quote(wrap)}"
    c = connect()
    try:
        _in, out, err = c.exec_command(wrapped, get_pty=False)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        rc = out.channel.recv_exit_status()
    finally:
        c.close()
    if o.strip():
        print(o.rstrip())
    if e.strip():
        print("--- stderr ---")
        print(e.rstrip())
    print(f"[exit {rc}]")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
