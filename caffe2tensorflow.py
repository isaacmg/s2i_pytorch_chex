from onnx_tf.backend import prepare
import onnx 
import tensorflow as tf
import os
import sys
from tensorflow.python.platform import gfile
import numpy as np
from scipy.misc import imread
import glob
def convert_to_tf(onnx_model_path, export_path, device):
    chex_model = onnx.load("model_weights/chexnet.onnx")
    chex_model_tf = prepare(chex_model, device='cpu')
    chex_model_tf.export_graph('model_weights/chest.pb')
def test_tensorflow():
    with gfile.FastGFile("model_weights/chest.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Session() as sess:
            sess.graph.as_default()
            tf.import_graph_def(graph_def)

#convert_to_tf("", "", "") 
#test_tensorflow()