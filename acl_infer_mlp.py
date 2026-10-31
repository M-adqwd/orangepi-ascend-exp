"""实验一：昇腾 OM 批量推理（单次加载模型，逐条 execute）。

原版 infer_mlp_ascend.py 对每条样本起一次 ais_bench 子进程（56962 次），
启动开销主导，约 16 小时且测不到真实延迟。本脚本改用 InferSession 批量循环。

用法:
  python3 acl_infer_mlp.py --om mlp_fraud.om --split . --limit 200
  python3 acl_infer_mlp.py --om mlp_fraud.om --split . --all --threshold 0.99
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--om", required=True)
    p.add_argument("--split", default=".")
    p.add_argument("--limit", type=int, default=200, help="只跑前 N 条（含校验）")
    p.add_argument("--all", action="store_true", help="跑完整测试集")
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--out", default=None)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def roc_auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y, s):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(y.sum(), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(precision, recall))


def main():
    args = parse_args()
    from ais_bench.infer.interface import InferSession

    split = Path(args.split)
    test = np.load(split / "split_test.npz")
    X = test["X"].astype(np.float32)
    y = test["y"].astype(np.int64)
    n = len(X) if args.all else min(args.limit, len(X))
    print(f"样本数 {n} / 全部 {len(X)}  正例 {int(y[:n].sum())}")

    t_load0 = time.perf_counter()
    session = InferSession(args.device, args.om)
    t_load = (time.perf_counter() - t_load0) * 1000
    print(f"模型加载: {t_load:.1f} ms")

    # 预热
    for _ in range(3):
        session.infer([X[0:1]])

    logits = np.empty(n, dtype=np.float32)
    lat = np.empty(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        a = time.perf_counter()
        out = session.infer([X[i:i + 1]])[0]
        lat[i] = (time.perf_counter() - a) * 1000
        logits[i] = float(np.asarray(out).reshape(-1)[0])
    total = time.perf_counter() - t0

    prob = sigmoid(logits)
    pred = (prob >= args.threshold).astype(np.int64)
    yi = y[:n]
    tp = int(((pred == 1) & (yi == 1)).sum())
    fp = int(((pred == 1) & (yi == 0)).sum())
    fn = int(((pred == 0) & (yi == 1)).sum())
    tn = int(((pred == 0) & (yi == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    lat_sorted = np.sort(lat)
    res = {
        "om": str(args.om),
        "samples": n,
        "threshold": args.threshold,
        "test_auprc": auprc(yi, prob),
        "test_roc_auc": roc_auc(yi, prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tp + tn) / n,
        "latency_mean_ms": float(lat.mean()),
        "latency_p50_ms": float(lat_sorted[len(lat) // 2]),
        "latency_p99_ms": float(lat_sorted[int(len(lat) * 0.99)]),
        "throughput_per_s": float(n / total),
        "wall_seconds": float(total),
        "model_load_ms": t_load,
        "timing_note": "单条同步 execute 的墙钟时间，含 Python/运行时封装开销，非纯 NPU 核心耗时",
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved ->", args.out)

    try:
        session.free_resource()
    except Exception as e:
        print("free_resource 跳过:", type(e).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
