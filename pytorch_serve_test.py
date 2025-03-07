import sys 
sys.path.append("..")
from model_agnostic.agnostic_model import ModelAgnostic
import torch
from collections import OrderedDict
class PytorchModel(ModelAgnostic):
    def __init__(self, weight_path, load_type=None):
        self.torch = __import__('torch')
        super().__init__(weight_path, "PyTorch")
        if load_type is "full":
            self.model = torch.load(weight_path)
            
        else:
            self.model = self.create_model()

            checkpoint = torch.load(weight_path, map_location= lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            if torch.cuda.device_count() < 1 and "state_dict" in checkpoint:
                for k, v in checkpoint['state_dict'].items() :
                    k = k[7:] # remove `module.`

                
                self.model.load_state_dict(new_state_dict)
            # Here we are assuming that raw state_dict has already been transformed 
            elif "state_dict" not in checkpoint:
                self.model.load_state_dict(checkpoint)
            else:
                self.model = torch.nn.DataParallel(self.model)
                self.model.load_state_dict(checkpoint['state_dict'])

    def create_model(self):
        pass 

    def predict(self, formatted_data):
        self.model(formatted_data)
    
    def preprocessing(self, items):
        pass
