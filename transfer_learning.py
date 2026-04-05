import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torchsummary import summary 


class SkinDiseaseModelResNet(nn.Module):
  def __init__(self, num_classes=22):
    super().__init__()
    self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1024),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(in_features=1024, out_features=num_classes)
    )
    
  def forward(self, x):
    return self.model(x)