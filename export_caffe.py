# Some standard imports
import io
import numpy as np
from torch import nn
import torch.onnx
import onnx
import onnx_caffe2.backend
from ChexNetPyTorch import ChexNetPyTorch
import json 
def export_model():
    the_model_class = ChexNetPyTorch()
    garbage, model_inputs = the_model_class.preprocessing("test_examples/test1.jpg")
    the_model = the_model_class.model
    the_model.train(False)
    torch_out = torch.onnx._export(the_model,             # model being run
                               model_inputs,                       # model input (or a tuple for multiple inputs)
                               "model_weights/chexnet-py.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file
    return torch_out
def test_model_export():
    # TODO refactor messy shit
    model = onnx.load("model_weights/chexnet-py.onnx")
    the_model_class = ChexNetPyTorch()
    garbage, model_inputs = the_model_class.preprocessing("test_examples/test1.jpg")
    prepared_backend = onnx_caffe2.backend.prepare(model)
    W = {model.graph.input[0].name: model_inputs.numpy()}
    c2_out = prepared_backend.run(W)[0]
    return c2_out
export_model()
torch_out = export_model()
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), test_model_export(), decimal=3)
#print(torch_out.data.cpu().numpy())