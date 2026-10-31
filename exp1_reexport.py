"""在板端从 mlp_fraud.pt 重新导出 ONNX（权重内联，不产生外部 .data 文件）。

原包里的 mlp_fraud.onnx 带外部权重引用（mlp_fraud.onnx.data），ATC 无法解析，故重导。
"""
import sys

import torch

# aarch64 上 oneDNN 的 matmul 原语描述符创建会失败，导出阶段关掉 MKLDNN
torch.backends.mkldnn.enabled = False

sys.path.insert(0, "/home/HwHiAiUser/exp/exp1")
from train_mlp import FraudMLP  # noqa: E402

CKPT = "/home/HwHiAiUser/exp/exp1/mlp_fraud.pt"
OUT = "/home/HwHiAiUser/exp/exp1/mlp_fraud_inline.onnx"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print("ckpt keys:", sorted(ckpt.keys()))

model = FraudMLP(
    in_features=ckpt["in_features"],
    hidden1=ckpt["hidden1"],
    hidden2=ckpt["hidden2"],
    dropout=ckpt["dropout"],
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print("params =", n_params)

dummy = torch.zeros(1, ckpt["in_features"], dtype=torch.float32)
torch.onnx.export(
    model,
    dummy,
    OUT,
    input_names=["input"],
    output_names=["logits"],
    opset_version=12,
    dynamic_axes=None,
    do_constant_folding=True,
)
print("exported ->", OUT)

import os

print("size =", os.path.getsize(OUT), "bytes")
