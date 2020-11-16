import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, num_classes, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        features = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(features, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet169(nn.Module):
    
    def __init__(self, num_classes, isTrained):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        
        features = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(features, num_classes), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, num_classes, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        features = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(features, num_classes), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x


class ResNet50(nn.Module):
    
    def __init__ (self, num_classes, isTrained):
        
        super(ResNet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        features = self.resnet50.fc.in_features
        
        self.resnet50.fc = nn.Sequential(nn.Linear(features, num_classes), nn.Sigmoid())
        
    def forward (self, x):
        x = self.resnet50(x)
        return x


class ResNet101(nn.Module):
    
    def __init__ (self, num_classes, isTrained):
        
        super(ResNet101, self).__init__()
    
        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        features = self.resnet101.fc.in_features
        
        self.resnet101.fc = nn.Sequential(nn.Linear(features, num_classes), nn.Sigmoid())
        
    def forward (self, x):
        x = self.resnet101(x)
        return x
