"""按 .pt 权重手工构建 mlp_fraud 的 ONNX 计算图（权重内联）。

原包 mlp_fraud.onnx 引用外部权重 mlp_fraud.onnx.data（未随包提供），ATC 无法加载。
torch.onnx.export 在本板 aarch64 上触发 oneDNN matmul 描述符错误，故直接用 onnx 构图。

结构（eval 模式，Dropout 失效）：
  MatMul/Add -> Relu -> MatMul/Add -> Relu -> MatMul/Add -> logits[1,1]
"""
import sys

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, "/home/HwHiAiUser/exp/exp1")
from train_mlp import FraudMLP  # noqa: F401,E402

CKPT = "/home/HwHiAiUser/exp/exp1/mlp_fraud.pt"
OUT = "/home/HwHiAiUser/exp/exp1/mlp_fraud_inline.onnx"
SAMPLE = "/home/HwHiAiUser/exp/exp1/calib_first_sample.npy"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
in_f = int(ckpt["in_features"])
h1 = int(ckpt["hidden1"])
h2 = int(ckpt["hidden2"])
print(f"in={in_f} h1={h1} h2={h2}")


def arr(name):
    return sd[name].detach().cpu().numpy().astype(np.float32)


W1, b1 = arr("fc1.weight").T, arr("fc1.bias")
W2, b2 = arr("fc2.weight").T, arr("fc2.bias")
W3, b3 = arr("fc3.weight").T, arr("fc3.bias")
print("shapes:", W1.shape, b1.shape, W2.shape, b2.shape, W3.shape, b3.shape)

inits = [
    numpy_helper.from_array(a, n)
    for n, a in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2), ("W3", W3), ("b3", b3)]
]

nodes = [
    helper.make_node("MatMul", ["input", "W1"], ["mm1"]),
    helper.make_node("Add", ["mm1", "b1"], ["add1"]),
    helper.make_node("Relu", ["add1"], ["relu1"]),
    helper.make_node("MatMul", ["relu1", "W2"], ["mm2"]),
    helper.make_node("Add", ["mm2", "b2"], ["add2"]),
    helper.make_node("Relu", ["add2"], ["relu2"]),
    helper.make_node("MatMul", ["relu2", "W3"], ["mm3"]),
    helper.make_node("Add", ["mm3", "b3"], ["logits"]),
]

graph = helper.make_graph(
    nodes,
    "fraud_mlp",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, in_f])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1])],
    inits,
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 12)], producer_name="handmade"
)
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("saved ->", OUT)

x = np.load(SAMPLE).astype(np.float32)
if x.ndim == 1:
    x = x.reshape(1, -1)

onnx_out = ReferenceEvaluator(model).run(None, {"input": x})[0]

ref = x @ W1 + b1
ref = np.maximum(ref, 0)
ref = ref @ W2 + b2
ref = np.maximum(ref, 0)
ref = ref @ W3 + b3

print("onnx  logit =", onnx_out.ravel())
print("numpy logit =", ref.ravel())
print("max abs diff = %.3e" % float(np.abs(onnx_out - ref).max()))
