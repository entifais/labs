import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Resnet building block

    y = F(x, {Wi}) + x.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.ModuleList([
            self.make_block(in_channels, out_channels),
            self.make_block(out_channels, out_channels)
        ])
        
    def make_block(self, in_channels, out_channels):
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        
        return net 

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        latent = input_tensor

        for func in self.net:
            latent = func(latent) + latent
        
        return latent

class BottleneckBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.ModuleList([
            self.make_block(in_channels, out_channels),
            self.make_block(out_channels, out_channels)
        ])
        projection = nn.ModuleList([
            self.make_projections(in_channels, out_channels),
            self.make_projections(out_channels, out_channels)
        ])
        net=nn.Sequential(
        )
    def make_projections(in_channels, out_channels):
        if in_channels==out_channels:
            projection=nn.Identity()
        else:
            projection=nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        latent = input_tensor

        for func in self.net:
            latent = func(latent) + latent
        
        return latent


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_weights()

    def init_weights(self):
        for name, module in self.named_modules():
            pass # init modules

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass