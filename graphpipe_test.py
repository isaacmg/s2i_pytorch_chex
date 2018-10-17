from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests
from model_agnostic.models.graph_pipe import GraphPipeRemote
import torch
import torchvision.transforms as move

from graphpipe import remote
def other_slu():
    data = np.array(Image.open("text.jpg"))
    data = data.reshape([1] + list(data.shape))
    data = np.rollaxis(data, 3, 1).astype(np.float32)  # channels first
    print(data.shape)

def preprocessing(image_path):
        """ """
        image = Image.open(image_path)
        normalize = move.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        trans = move.Compose([
                                        move.Resize(256),
                                        move.TenCrop(224),
                                        move.Lambda
                                        (lambda crops: torch.stack([move.ToTensor()(crop) for crop in crops])),
                                         move.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])
        image = trans(image)
        
        n_crops, c, h, w = image.size()
        
        return n_crops, torch.autograd.Variable(image.float(), volatile=True)

def get_image(img_path):
    junk, img = preprocessing(img_path)
    return img.numpy()



data = get_image("text.jpg")
#pred = remote.execute("http://127.0.0.1:9000", data)
pred = remote.execute("http://127.0.0.1:9000", data[:1])
print(pred)
print("Expected 504 (Coffee mug), got: %s", np.argmax(pred, axis=1))
#[inputs:0 sequence_length:0 slots:0 slot_weights:0 intent:0]
