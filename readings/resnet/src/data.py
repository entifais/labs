from typing import Optional

import pytorch_lightning as pl
from torch.utils import data
from torchvision.datasets import CIFAR10
from torchvision import transforms as T

# CIFAR_MEAN = 
# CIFAR_STD = 

class CifarDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> data.DataLoader:
        pass

    def val_dataloader(self) -> data.DataLoader:
        pass