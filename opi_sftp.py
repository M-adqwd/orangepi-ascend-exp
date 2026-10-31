"""通过 SFTP 在本地与 Orange Pi 之间传文件。

用法:
  opi_sftp.py push <本地路径> <远程路径>    文件或目录（目录会递归）
  opi_sftp.py pull <远程路径> <本地路径>
"""
import json
import os
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


def mkdir_p(sftp, path):
    is_abs = path.startswith("/")
    cur = "/" if is_abs else ""
    for part in [p for p in path.split("/") if p]:
        cur = cur.rstrip("/") + "/" + part if cur else part
        try:
            sftp.mkdir(cur)
        except IOError:
            pass


def push(sftp, local, remote):
    if os.path.isfile(local):
        parent = remote.rsplit("/", 1)[0] if "/" in remote else "."
        if parent:
            mkdir_p(sftp, parent)
        sftp.put(local, remote)
        print(f"↑ {local} -> {remote}")
        return 1
    mkdir_p(sftp, remote)
    n = 0
    for root, _dirs, files in os.walk(local):
        rel = os.path.relpath(root, local).replace("\\", "/")
        rdir = remote if rel == "." else f"{remote}/{rel}"
        mkdir_p(sftp, rdir)
        for fn in files:
            lp = os.path.join(root, fn)
            sftp.put(lp, f"{rdir}/{fn}")
            n += 1
            if n % 200 == 0:
                print(f"  已传 {n} 个文件...", flush=True)
    print(f"↑ 目录 {local} -> {remote}（{n} 个文件）")
    return n


def pull(sftp, remote, local):
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    sftp.get(remote, local)
    print(f"↓ {remote} -> {local}")
    return 1


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    action, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    c = connect()
    sftp = c.open_sftp()
    try:
        if action == "push":
            push(sftp, a, b)
        elif action == "pull":
            pull(sftp, a, b)
        else:
            sys.exit(__doc__)
    finally:
        sftp.close()
        c.close()


if __name__ == "__main__":
    main()
