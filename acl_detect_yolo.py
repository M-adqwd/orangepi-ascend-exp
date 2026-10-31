"""实验三：昇腾 OM YOLOv8n 静态图人员检测 + 延迟基准。

摄像头环节需 USB UVC 设备，本板当前无 USB 摄像头（/dev/video* 为 MIPI multiplanar），
故先做静态图验证，确认 OM 与后处理链路正确。

用法:
  python3 acl_detect_yolo.py --om yolov8n.om --image bus.jpg --out out.jpg
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="detected.jpg")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--bench", type=int, default=50)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    from ais_bench.infer.interface import InferSession

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"读不到图像: {args.image}")
    h0, w0 = img.shape[:2]

    canvas, scale, (left, top) = letterbox_yolov8(img, 640)
    x = preprocess_yolov8(canvas)[None, ...]

    session = InferSession(args.device, args.om)
    for _ in range(3):
        session.infer([x])

    t0 = time.perf_counter()
    out = np.asarray(session.infer([x])[0])
    lat = (time.perf_counter() - t0) * 1000
    print(f"输出形状 {out.shape}")

    boxes, scores, cls = decode_yolov8(out, conf_thres=args.conf, iou_thres=args.iou, person_only=True)
    print(f"检出 {len(boxes)} 个人")

    for b, s in zip(boxes, scores):
        x1 = (b[0] - left) / scale
        y1 = (b[1] - top) / scale
        x2 = (b[2] - left) / scale
        y2 = (b[3] - top) / scale
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w0 - 1, int(x2)), min(h0 - 1, int(y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"person {s:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        print(f"  box=({x1},{y1},{x2},{y2}) score={s:.3f}")

    cv2.imwrite(args.out, img)
    print(f"单帧延迟 {lat:.2f} ms  -> {args.out}")

    if args.bench:
        for _ in range(3):
            session.infer([x])
        ts = []
        for _ in range(args.bench):
            a = time.perf_counter()
            session.infer([x])
            ts.append((time.perf_counter() - a) * 1000)
        ts = np.array(ts)
        print(f"基准 {args.bench} 次: mean={ts.mean():.2f}ms p50={np.median(ts):.2f}ms "
              f"p99={np.sort(ts)[int(len(ts)*0.99)]:.2f}ms  => {1000/ts.mean():.1f} FPS")

    try:
        session.free_resource()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
