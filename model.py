import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained = True).features)[:26]
        self.features = nn.ModuleList(features).eval()
        
    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 18, 25}:
                results.append(x)
        return results

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu = nn.Relu()

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.add(x, x_in)
        return x


class Encoding_network(nn.Module):
    def __init__(self):
        super(Encoding_network, self).__init__()
        self.conv

class Texture_model(nn.Module):
    def __init__(self):
        super(Texture_model, self).__init__()


    def forward(self, x):
