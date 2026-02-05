from torchvision.models import efficientnet_b2, mobilenet_v3_small, resnet34
from torchvision import models
import torch.nn as nn


def create_resnet34_model(num_classes: int) -> nn.Module:
    model = resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_efficientnet_b2_model(num_classes: int) -> nn.Module:
    model = efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def create_mobilenet_v3_small_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model