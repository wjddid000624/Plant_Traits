import torch
import torch.nn as nn
import torchvision.models as models

class EfficientPlant(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_v2_s(pretrained=True)
        self.numeric = nn.Sequential(nn.Linear(27, 100),
                                     nn.BatchNorm1d(100),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(100, 200),
                                     nn.BatchNorm1d(200),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(200, 200),
                                     nn.BatchNorm1d(200),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(200, 100))
        self.output = nn.Sequential(nn.Linear(600, 300),
                                    nn.BatchNorm1d(300),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(300, 100),
                                    nn.BatchNorm1d(100),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(100, 6))
        self.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(1280, 500, bias=True))
        setattr(self.model, 'classifier', self.classifier)
                                        

    def forward(self, image, data):
        image = self.model(image)
        data = self.numeric(data)
        sum = torch.cat([image, data],dim=1)
        return self.output(sum)