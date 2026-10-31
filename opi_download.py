"""把板端实验产物递归下载到本地，并生成清单。

用法:
  opi_download.py <远程目录> <本地目录> [...]
  opi_download.py --all                     # 下载 ~/exp 下三个实验
"""
import hashlib
import json
import os
import posixpath
import sys
import time

import paramiko

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "opi_config.json"), encoding="utf-8") as f:
    CFG = json.load(f)

SKIP_DIRS = {"__pycache__", ".git", "frames"}
SKIP_EXT = {".pyc", ".log"}


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


def walk(sftp, root):
    out = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sftp.listdir_attr(d)
        except IOError:
            continue
        for e in entries:
            p = posixpath.join(d, e.filename)
            if e.st_mode & 0o040000:
                if e.filename in SKIP_DIRS:
                    continue
                stack.append(p)
            else:
                if os.path.splitext(e.filename)[1].lower() in SKIP_EXT:
                    continue
                out.append((p, e.st_size))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--all":
        pairs = [
            ("/home/HwHiAiUser/exp/exp1", "/opt/orangepi/downloads/exp1"),
            ("/home/HwHiAiUser/exp/exp2", "/opt/orangepi/downloads/exp2"),
            ("/home/HwHiAiUser/exp/exp3", "/opt/orangepi/downloads/exp3"),
        ]
    else:
        if len(args) % 2:
            sys.exit(__doc__)
        pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]

    c = connect()
    sftp = c.open_sftp()
    manifest = []
    t0 = time.time()
    total_bytes = 0
    total_files = 0

    try:
        for remote_root, local_root in pairs:
            files = walk(sftp, remote_root)
            print(f"\n=== {remote_root}  共 {len(files)} 个文件 ===", flush=True)
            for p, size in files:
                rel = p[len(remote_root):].lstrip("/")
                lp = os.path.join(local_root, rel.replace("/", os.sep))
                os.makedirs(os.path.dirname(lp) or ".", exist_ok=True)
                if os.path.exists(lp) and os.path.getsize(lp) == size:
                    pass
                else:
                    sftp.get(p, lp)
                total_bytes += size
                total_files += 1
                manifest.append({
                    "remote": p, "local": lp.replace("\\", "/"),
                    "bytes": size, "sha256": sha256(lp),
                })
                if total_files % 25 == 0:
                    print(f"  {total_files} 个文件 / {total_bytes/1048576:.1f} MB", flush=True)
    finally:
        sftp.close()
        c.close()

    mf = os.path.join(BASE, "downloads", "MANIFEST.json")
    os.makedirs(os.path.dirname(mf), exist_ok=True)
    with open(mf, "w", encoding="utf-8") as f:
        json.dump({
            "host": CFG["host"],
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": total_files,
            "bytes": total_bytes,
            "elapsed_s": round(time.time() - t0, 1),
            "items": manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n完成：{total_files} 个文件 / {total_bytes/1048576:.1f} MB / "
          f"{time.time()-t0:.1f}s")
    print(f"清单：{mf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
