import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        pass

class BottleneckBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        pass

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_weights()

    def init_weights(self):
        for name, module in self.named_modules():
            pass # init modules

    def forward(self, input_tensor):
        pass