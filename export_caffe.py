# Some standard imports
import io
import numpy as np
from torch import nn
import torch.onnx
import onnx
import onnx_caffe2.backend
from ChexNetPyTorch import ChexNetPyTorch
def export_model():
    the_model_class = ChexNetPyTorch()
    garbage, model_inputs = the_model_class.preprocessing("text.jpg")
    the_model = the_model_class.model
    torch_out = torch.onnx._export(the_model,             # model being run
                               model_inputs,                       # model input (or a tuple for multiple inputs)
                               "chexnet.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file
def test_model_export():
    # TODO refactor messy shit
    model = onnx.load("chexnet.onnx")
    the_model_class = ChexNetPyTorch()
    garbage, model_inputs = the_model_class.preprocessing("text.jpg")
    prepared_backend = onnx_caffe2.backend.prepare(model)
    W = {model.graph.input[0].name: model_inputs.numpy()}
    c2_out = prepared_backend.run(W)[0]
    return c2_out

print(test_model_export())