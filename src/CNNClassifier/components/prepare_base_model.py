import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import models 

from src.CNNClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel():
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config 
    
    def get_base_model(self):
        self.model = models.vgg16(pretrained=True)
        self.save_model(self.model, self.config.base_model_path)

    def prepare_full_model(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.config.params_classes)

        self.save_model(self.model, self.config.updated_base_model_path)
        return self.model

    @staticmethod
    def save_model(model, path):
        torch.save(model, path)