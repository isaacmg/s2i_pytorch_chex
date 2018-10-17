from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests
from model_agnostic.models.graph_pipe import GraphPipeRemote
import torch
import torchvision.transforms as move

from graphpipe import remote
class ChexNetDeploy(GraphPipeRemote):
    def preprocessing(self, image_path:str):
            """ 
            """
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
            
            return n_crops, torch.autograd.Variable(image.float(), volatile=True).numpy()

    def process_result(self, n):
        if len(self.result>1):
            print("data below")
            #print(self.result)
            result = self.result.mean(axis=0)
            
        else: 
            result = self.result[0]
        class_name = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        #print('Predicted: ', ' '.join('%5s' % class_name[predicted[j]] for j in range(1)))
        result_dict = {}

        for i in range(0, len(class_name)-1):
            result_dict[class_name[i]] = result[i]
        return result_dict


model = ChexNetDeploy("http://127.0.0.1:9000")
n_crops, preprocessed_image = model.preprocessing("text.jpg")
model.predict(preprocessed_image)
final_result = model.process_result(n_crops)
print(final_result)
#pred = remote.execute("http://127.0.0.1:9000", data)
#pred = remote.execute("http://127.0.0.1:9000", data[:1])
#print(pred)
#print("Expected 504 (Coffee mug), got: %s", np.argmax(pred, axis=1))
#[inputs:0 sequence_length:0 slots:0 slot_weights:0 intent:0]
