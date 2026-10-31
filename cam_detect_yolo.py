"""实验三：USB 摄像头 YOLOv8n 实时人员检测（昇腾 OM）。

使用 InferSession（已在本板验证）+ 项目自带 letterbox/decode。
规避 infer_yolov8_ascend.py 中 pyACL 常量名与本板版本不兼容的问题。

用法:
  python3 cam_detect_yolo.py --om yolov8n.om --device /dev/video2 --save ./frames --max-frames 120
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.preprocess import letterbox_yolov8, preprocess_yolov8  # noqa: E402
from utils.yolov8_decode import decode_yolov8  # noqa: E402


def open_camera(device, width, height):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"打不开摄像头 {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=True)
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save", default="./frames")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    from ais_bench.infer.interface import InferSession

    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.device, args.width, args.height)
    session = InferSession(0, args.om)

    n = 0
    infer_ms = []
    e2e_ms = []
    saved = 0
    t_start = time.perf_counter()

    while n < args.max_frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("读帧失败，退出")
            break
        n += 1

        canvas, scale, (left, top) = letterbox_yolov8(frame, args.size)
        x = preprocess_yolov8(canvas)[None, ...]

        ti = time.perf_counter()
        out = np.asarray(session.infer([x])[0])
        dt = (time.perf_counter() - ti) * 1000

        boxes, scores, _cls = decode_yolov8(
            out, conf_thres=args.conf, iou_thres=args.iou, person_only=True
        )

        if n > args.warmup:
            infer_ms.append(dt)
            e2e_ms.append((time.perf_counter() - t0) * 1000)

        h0, w0 = frame.shape[:2]
        for b, s in zip(boxes, scores):
            x1 = max(0, int((b[0] - left) / scale))
            y1 = max(0, int((b[1] - top) / scale))
            x2 = min(w0 - 1, int((b[2] - left) / scale))
            y2 = min(h0 - 1, int((b[3] - top) / scale))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"person {s:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"persons={len(boxes)}  {dt:.1f}ms", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if n % args.save_every == 0:
            p = out_dir / f"frame_{n:04d}.jpg"
            cv2.imwrite(str(p), frame)
            saved += 1
            if n % (args.save_every * 4) == 0:
                print(f"  [{n}/{args.max_frames}] persons={len(boxes)} "
                      f"infer={dt:.1f}ms -> {p.name}", flush=True)

    wall = time.perf_counter() - t_start
    cap.release()

    print()
    print(f"处理帧数 {n}  保存 {saved} 张  总耗时 {wall:.1f}s")
    if infer_ms:
        a = np.array(infer_ms)
        b = np.array(e2e_ms)
        print(f"纯推理: mean={a.mean():.1f}ms p50={np.median(a):.1f}ms "
              f"p99={np.sort(a)[int(len(a)*0.99)]:.1f}ms => {1000/a.mean():.1f} FPS")
        print(f"端到端: mean={b.mean():.1f}ms => {1000/b.mean():.1f} FPS")
    try:
        session.free_resource()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
