from onnx_tf.backend import prepare
import onnx 
def convert_to_tf(onnx_model_path, export_path, device):
    chex_model = onnx.load("model_weights/chexnet.onnx")
    chex_model_tf = prepare(chex_model, device='cpu')
    chex_model_tf.export_graph('model_weights/chest.pb')
convert_to_tf("", "", "") 