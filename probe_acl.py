"""探测 pyACL 与 ais_bench 可用的批量推理入口。"""
import os

import acl

print("== acl 模块级 ==")
print([x for x in dir(acl) if "data_buffer" in x or "MALLOC" in x or "MEMCPY" in x])
print("== acl.rt malloc/memcpy ==")
print([x for x in dir(acl.rt) if "malloc" in x.lower() or "memcpy" in x.lower()])

print("\n== ais_bench 包结构 ==")
import ais_bench

base = os.path.dirname(ais_bench.__file__)
print("base:", base)
for root, dirs, files in os.walk(os.path.join(base, "infer")):
    depth = root[len(base):].count(os.sep)
    if depth > 2:
        continue
    print(" ", root[len(base):] or "/", "->", sorted(files)[:12])

print("\n== ais_bench.infer 顶层 ==")
try:
    import ais_bench.infer as inf

    print([x for x in dir(inf) if not x.startswith("_")][:40])
except Exception as e:
    print("ERR", e)

print("\n== session 模块 ==")
for mod in ("ais_bench.infer.session", "ais_bench.infer.interface", "ais_bench.infer.infer_api"):
    try:
        m = __import__(mod, fromlist=["x"])
        names = [x for x in dir(m) if not x.startswith("_") and x[0].isupper()]
        print(mod, "->", names[:15])
    except Exception as e:
        print(mod, "ERR", type(e).__name__, e)
