"""解包 E:/03实验 下的实验素材。

exp1 / exp3 全量解包；exp2_full.zip 跳过 data/chest_xray/train（约 5000 张，板端不需要）。
"""
import os
import zipfile

SRC = "E:/03实验"
OUT = "E:/03实验/extracted"

SKIP_PREFIXES = ("data/chest_xray/train",)


def extract(name, subdir, skip=()):
    zpath = os.path.join(SRC, name)
    dest = os.path.join(OUT, subdir)
    os.makedirs(dest, exist_ok=True)
    got, skipped = 0, 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            n = info.filename
            if any(n.startswith(p) for p in skip):
                skipped += 1
                continue
            z.extract(info, dest)
            got += 1
    print(f"{name} -> {dest}: 解出 {got} 项，跳过 {skipped} 项")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    extract("exp1.zip", "exp1")
    extract("exp3.zip", "exp3")
    extract("exp2_full.zip", "exp2", SKIP_PREFIXES)
    print("完成")
