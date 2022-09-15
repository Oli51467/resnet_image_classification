import onnx
import torch
from onnxsim import simplify

INPUT_DICT = './weight/best-go.pth'
OUT_ONNX = './weight/best.onnx'

x = torch.randn(1, 3, 224, 224)
input_names = ["input"]
out_names = ["output"]

model = torch.load(INPUT_DICT, map_location=torch.device('cpu'))
model.eval()

torch.onnx._export(model, x, OUT_ONNX, export_params=True, training=False, input_names=input_names,
                   output_names=out_names)
onnx_model = onnx.load('./weight/best.onnx')
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, './weight/best-sim.onnx')
