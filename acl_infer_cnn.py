"""实验二：昇腾 OM 肺炎 X 光分类推理。

用法:
  python3 acl_infer_cnn.py --om cnn_pneumonia_efficientnet_b0.om --image <path>
  python3 acl_infer_cnn.py --om ... --test-dir data/chest_xray/test --out metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import preprocess_cnn  # noqa: E402

CLASSES = ["NORMAL", "PNEUMONIA"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--om", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", default=None)
    p.add_argument("--threshold", type=float, default=0.5, help="判定为 PNEUMONIA 的概率阈值")
    p.add_argument("--scores-out", default=None, help="把逐图 score/label 存成 npz，便于扫阈值")
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def roc_auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), len(y) - pos.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y, s):
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(yy.sum(), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(precision, recall))


def main():
    args = parse_args()
    from ais_bench.infer.interface import InferSession

    session = InferSession(args.device, args.om)

    if args.image:
        x = preprocess_cnn(cv2_imread(args.image), size=args.size)[None, ...]
        for _ in range(3):
            session.infer([x])
        t0 = time.perf_counter()
        out = np.asarray(session.infer([x])[0]).reshape(-1)
        dt = (time.perf_counter() - t0) * 1000
        prob = softmax(out)
        k = int(prob.argmax())
        print(f"image={args.image}")
        print(f"logits={out.round(4).tolist()}  prob={prob.round(4).tolist()}")
        print(f"pred={CLASSES[k]}  conf={prob[k]:.4f}  latency={dt:.2f}ms")
    elif args.test_dir:
        root = Path(args.test_dir)
        files, labels = [], []
        for idx, name in enumerate(CLASSES):
            for f in sorted((root / name).glob("*")):
                if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    files.append(f)
                    labels.append(idx)
        y = np.array(labels)
        print(f"测试集 {len(files)} 张  正例 {int(y.sum())}")

        for _ in range(3):
            session.infer([preprocess_cnn(cv2_imread(files[0]), size=args.size)[None, ...]])

        scores, lat = np.empty(len(files)), np.empty(len(files))
        preds = np.empty(len(files), dtype=np.int64)
        for i, f in enumerate(files):
            x = preprocess_cnn(cv2_imread(f), size=args.size)[None, ...]
            a = time.perf_counter()
            out = np.asarray(session.infer([x])[0]).reshape(-1)
            lat[i] = (time.perf_counter() - a) * 1000
            p = softmax(out)
            scores[i] = p[1]
            preds[i] = int(p[1] >= args.threshold)

        if args.scores_out:
            np.savez(args.scores_out, y=y, scores=scores)
            print("scores saved ->", args.scores_out)

        acc = float((preds == y).mean())
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        pre = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        res = {
            "om": args.om,
            "images": len(files),
            "threshold": args.threshold,
            "accuracy": acc,
            "auprc": auprc(y, scores),
            "roc_auc": roc_auc(y, scores),
            "precision": pre,
            "recall": rec,
            "f1": 2 * pre * rec / max(pre + rec, 1e-9),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "latency_mean_ms": float(lat.mean()),
            "latency_p50_ms": float(np.median(lat)),
            "latency_p99_ms": float(np.sort(lat)[int(len(lat) * 0.99)]),
            "timing_note": "单张同步 execute 墙钟时间，含预处理后的运行时开销",
        }
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print("saved ->", args.out)
    else:
        sys.exit("需要 --image 或 --test-dir")

    try:
        session.free_resource()
    except Exception:
        pass
    return 0


def cv2_imread(path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"读不到图像: {path}")
    return img


if __name__ == "__main__":
    sys.exit(main())
