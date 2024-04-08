import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Model1(nn.Module):
    ### Resnet pretrained model 이용
    def __init__(self, output_dim=6, freeze_backbone=False):
        super(Model1, self).__init__()
        resnet = models.resnet50(pretrained=True)
        backbone = nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-2])])) # drop last layer which is classifier

        ## freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        ## avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ## customized regression layers
        self.regressor = nn.Sequential(
            nn.Linear(2075, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

    def forward(self, img, infos):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        concat = torch.cat((feat_1d, infos), 1)
        logit = self.regressor(concat)

        return logit
