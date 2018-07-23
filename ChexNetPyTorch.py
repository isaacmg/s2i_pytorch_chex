from agnostic_model.models.dense_ne import DenseNet121
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as move
from PIL import Image
from model_agnostic.models.pytorch_serve import PytorchModel
from densenet.dense_ne import DenseNet121

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ChexNetPyTorch(PytorchModel):
    def __init__(self, weight_path="model_new2.pth.tar", load_type="full"):
        DenseNet121(14)
        
from agnostic_model.models.pytorch_serve import PytorchModel

class ChexNetPyTorch(PytorchModel):
    def __init__(self, weight_path="state_dic.pth.tar", load_type=""):
        super(ChexNetPyTorch, self).__init__(weight_path, load_type)
        #torch.save(self.model.state_dict(), "state_dic.pth.tar")
        
    def create_model(self):
        return DenseNet121(14)
        
    def preprocessing(self, image_path):
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
        
    
    def predict(self, non_formatted_data):
        """Overide for compatibility with Seldon Core S2I"""
        n_crops, formatted_data = self.preprocessing(non_formatted_data)
        result = self.model(formatted_data)
        return self.process_result(n_crops, result)
    
    def process_result(self, n,  result):
        result = result.view(1, n, -1).mean(1)
        print(result.data[0])
        ir , predicted = torch.max(result, 1)
        class_name = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        print('Predicted: ', ' '.join('%5s' % class_name[predicted[j]] for j in range(1)))
        result_dict = {}

        for i in range(0, len(class_name)-1):
            result_dict[class_name[i]] = result.tolist()[0][i]
        return result_dict
        



model = ChexNetPyTorch()
print(model.predict("text.jpg"))
