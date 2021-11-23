from typing import Tuple
import pytorch_lightning as pl
import torch

from src.model import Resnet

Batch = Tuple[torch.Tensor, torch.Tensor]

class ResnetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: Batch, _) -> torch.Tensor:
        pass

    def validation_step(self, batch: Batch, _) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        pass