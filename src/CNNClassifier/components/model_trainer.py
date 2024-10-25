import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm 

from src.CNNClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config 

    def get_base_model(self):
        self.device = self.config.params_device
        self.model = torch.load(self.config.updated_base_model_path)
        self.model = self.model.to(self.device)

    def train_valid_genertor(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }   

        train_dataset = datasets.ImageFolder(root=Path(f'{self.config.training_data}/train'), transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(root=Path(f'{self.config.training_data}/test'), transform=data_transforms['test'])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.params_batch_size, shuffle=False)
    
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)

        best_acc = 0.0  

        for epoch in tqdm(range(self.config.params_epochs), desc="Training: "):
            for phase in ['train', 'val']:
                self.model.train() if phase == 'train' else self.model.eval()  
                data_loader = self.train_loader if phase == 'train' else self.val_loader

                running_loss, running_corrects = 0.0, 0

                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_corrects.double() / len(data_loader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model, self.config.trained_model_path)
        