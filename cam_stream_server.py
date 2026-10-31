"""实验三：USB 摄像头 YOLOv8n 实时人员检测 —— MJPEG 网页直播。

板端跑一个 HTTP 服务，浏览器打开 http://<板子IP>:8000 即可看到带框的实时画面。
检测 + 绘图在单一后台线程完成，HTTP 线程只负责把最新 JPEG 推给客户端，
避免多线程同时操作 NPU 会话。

用法:
  python3 cam_stream_server.py --om yolov8n.om --port 8000          # 自动找 USB 摄像头
  python3 cam_stream_server.py --om yolov8n.om --device /dev/video0  # 手动指定
  python3 cam_stream_server.py --list                                # 只看设备清单
  # 浏览器: http://192.168.8.199:8000
  # 单帧快照: http://192.168.8.199:8000/snapshot

注意: 板子重启后 USB 摄像头的 /dev/videoN 编号可能变化（本次就从 video2 变成了 video0），
      所以默认走自动探测，不要硬编码节点号。
"""
import argparse
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.preprocess import letterbox_yolov8, preprocess_yolov8  # noqa: E402
from utils.yolov8_decode import decode_yolov8  # noqa: E402

STATE = {
    "jpeg": None,
    "seq": 0,
    "persons": 0,
    "infer_ms": 0.0,
    "fps": 0.0,
    "error": None,
}
LOCK = threading.Lock()

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orange Pi AI Pro · 实时人员检测</title>
<style>
 body{margin:0;background:#111;color:#eee;font:14px/1.5 -apple-system,"Segoe UI",sans-serif;
      display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}
 header{padding:12px 0;font-weight:500;letter-spacing:.3px}
 .wrap{position:relative;max-width:96vw}
 img{display:block;max-width:96vw;max-height:78vh;border-radius:8px;background:#000}
 .tip{margin-top:10px;color:#888;font-size:12px}
</style></head>
<body>
<header>Orange Pi AI Pro · Ascend 310B4 · YOLOv8n 人员检测</header>
<div class="wrap"><img src="/stream" alt="实时检测画面"></div>
<div class="tip">实时 MJPEG 流 · 单帧快照 /snapshot</div>
</body></html>
"""


def list_cameras():
    """返回 [(设备名, [节点...]), ...]，解析 v4l2-ctl --list-devices 的输出。"""
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    blocks, cur = [], None
    for line in out.splitlines():
        if not line.strip():
            cur = None
            continue
        if not line.startswith(("\t", " ")):          # 设备名行
            cur = (line.strip().rstrip(":"), [])
            blocks.append(cur)
        elif cur is not None:                          # 节点行
            node = line.strip()
            if node.startswith("/dev/video"):
                cur[1].append(node)
    return blocks


def pick_camera(explicit=None):
    """挑出 USB 摄像头的 video 节点。显式指定时直接返回。"""
    if explicit and explicit != "auto":
        return explicit, f"手动指定 {explicit}"
    blocks = list_cameras()
    usb = [(n, nodes) for n, nodes in blocks if "usb" in n.lower() and nodes]
    if usb:
        name, nodes = usb[0]
        return nodes[0], f"USB 摄像头「{name}」→ {nodes[0]}"
    # 没有 usb 关键字时，退回排除板载 platform 设备的其他项
    other = [(n, nodes) for n, nodes in blocks
             if "platform:" not in n.lower() and nodes]
    if other:
        name, nodes = other[0]
        return nodes[0], f"非板载设备「{name}」→ {nodes[0]}"
    return None, "未找到可用的 /dev/video 节点"


def worker(args, device):
    from ais_bench.infer.interface import InferSession

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        with LOCK:
            STATE["error"] = f"打不开摄像头 {device}"
        print(STATE["error"], flush=True)
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    session = InferSession(0, args.om)
    print(f"摄像头 {device} 已打开，模型 {args.om} 已加载", flush=True)

    n = 0
    t_prev = time.perf_counter()
    fps_ema = 0.0
    fail = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail += 1
            if fail > 30:
                with LOCK:
                    STATE["error"] = "连续读帧失败，摄像头可能已断开"
                print(STATE["error"], flush=True)
                break
            time.sleep(0.05)
            continue
        fail = 0
        n += 1

        canvas, scale, (left, top) = letterbox_yolov8(frame, args.size)
        x = preprocess_yolov8(canvas)[None, ...]

        ti = time.perf_counter()
        out = np.asarray(session.infer([x])[0])
        dt = (time.perf_counter() - ti) * 1000

        boxes, scores, _cls = decode_yolov8(
            out, conf_thres=args.conf, iou_thres=args.iou, person_only=True
        )

        h0, w0 = frame.shape[:2]
        for b, s in zip(boxes, scores):
            x1 = max(0, int((b[0] - left) / scale))
            y1 = max(0, int((b[1] - top) / scale))
            x2 = min(w0 - 1, int((b[2] - left) / scale))
            y2 = min(h0 - 1, int((b[3] - top) / scale))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"person {s:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = inst if fps_ema == 0 else fps_ema * 0.9 + inst * 0.1

        cv2.putText(frame, f"persons={len(boxes)}  infer={dt:.1f}ms  {fps_ema:.1f}fps",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            continue
        with LOCK:
            STATE["jpeg"] = buf.tobytes()
            STATE["seq"] += 1
            STATE["persons"] = len(boxes)
            STATE["infer_ms"] = dt
            STATE["fps"] = fps_ema

    cap.release()
    try:
        session.free_resource()
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.do_stream()
        elif self.path.startswith("/snapshot"):
            self.do_snapshot()
        elif self.path.startswith("/stats"):
            self.do_stats()
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_snapshot(self):
        with LOCK:
            jpg = STATE["jpeg"]
        if not jpg:
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpg)

    def do_stats(self):
        import json

        with LOCK:
            data = {k: STATE[k] for k in ("seq", "persons", "infer_ms", "fps", "error")}
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                with LOCK:
                    seq, jpg = STATE["seq"], STATE["jpeg"]
                if jpg is None or seq == last:
                    time.sleep(0.005)
                    continue
                last = seq
                head = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(head)
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--om", required=False)
    ap.add_argument("--device", default="auto",
                    help="摄像头节点，默认 auto = 自动找 USB 摄像头")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--list", action="store_true", help="只列出摄像头设备后退出")
    args = ap.parse_args()

    if args.list:
        for name, nodes in list_cameras():
            tag = "  [USB]" if "usb" in name.lower() else "  [板载]"
            print(f"{name}{tag}\n    " + ", ".join(nodes))
        dev, why = pick_camera(args.device)
        print(f"\n自动选择: {why}")
        return 0

    if not args.om:
        ap.error("需要 --om 指定模型文件（或改用 --list 只看设备）")

    device, why = pick_camera(args.device)
    if not device:
        print(why, flush=True)
        for name, nodes in list_cameras():
            print(f"  可用: {name} -> {nodes}", flush=True)
        return 1
    print(f"[摄像头] {why}", flush=True)

    threading.Thread(target=worker, args=(args, device), daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"服务已启动: http://<板子IP>:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
