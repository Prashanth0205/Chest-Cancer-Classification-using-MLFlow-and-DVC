import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import dagshub
import mlflow
import mlflow.pytorch
from pathlib import Path
from urllib.parse import urlparse

from src.CNNClassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config 

    def get_trained_model(self):
        self.device = self.config.params_device
        self.model = torch.load(self.config.path_of_model)
        self.model = self.model.to(self.device)

    def test_generator(self):
        test_data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_dataset = datasets.ImageFolder(root=Path(f'{self.config.training_data}/test'), transform=test_data_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    def test(self):
        criterion = nn.CrossEntropyLoss() 
        total_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)  
                _, preds = torch.max(outputs, 1)

                total_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

        self.loss = total_loss / len(self.test_loader.dataset)
        self.accuracy = running_corrects/ len(self.test_loader.dataset)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.loss, "accuracy": self.accuracy}

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.loss, "accuracy": self.accuracy})

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name='VGG16_PytorchModel')
            else:
                mlflow.pytorch.log_model(self.model, "model")
