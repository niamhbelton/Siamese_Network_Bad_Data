import torch
import torch.nn as nn
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x1):
        x1 = torch.squeeze(x1, dim=0)
        features = self.pretrained_model.features(x1)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features1 = torch.max(pooled_features, 0, keepdim=True)[0]
        x1 = self.sigmoid(flattened_features1)
        return x1
    
