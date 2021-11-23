from typing import Callable, Optional

from pathlib import Path
import requests
import tarfile

import pytorch_lightning as pl
from torch.utils import data
from torchvision.datasets import DatasetFolder
from torchvision import transforms as T
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

# IMAGEWOOF_MEAN = 
# IMAGEWOOF_STD = 

_IMAGEWOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz"
_CLASS_MAPPING = {
    "shih-tzu": 0,
    "rhodesian_ridgeback": 1,
    "beagle": 2,
    "english_foxhound": 3,
    "border_terrier": 4,
    "australian_terrier": 5,
    "golden_retriever": 6,
    "old_english_sheepdog": 7,
    "samoyed": 8,
    "dingo": 9,
}

def _download_url(url: str, data_dir: Path) -> Path:
    res = requests.get(url, stream=True)
    total_size_bytes = int(res.headers.get("content-length", 0))
    block_size = 1024 # 1 Kilobyte
    progress_bar = tqdm(total=total_size_bytes, unit="iB", unit_scale=True)
    tar_file = data_dir / "imagewoof.tgz"

    with open(tar_file, "wb") as buf:
        for data in res.iter_content(block_size):
            progress_bar.update(len(data))
            buf.write(data)
    
    progress_bar.close()

    assert total_size_bytes == 0 or progress_bar.n == total_size_bytes, RuntimeError

    return tar_file

def _untar_file(fname: Path) -> Path:
    target_folder = fname.parent / "imagewoof2"
    tmp_folder = fname.parent / "tmp"

    with tarfile.open(fname) as file:
        file.extractall(tmp_folder)

    (tmp_folder / "imagewoof2").rename(target_folder)
    tmp_folder.rmdir()

    return target_folder

def int2str(class_idx: int) -> str:
    return list(_CLASS_MAPPING.keys())[class_idx]

def str2int(class_name: str) -> int:
    return _CLASS_MAPPING[class_name]

class ImagewoofDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_root_dir: str = ".data",
        batch_size: int = 32,
        num_workers: int = 2, # colab default
        pin_memory: bool = True,
        train_image_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        val_image_transform: Optional[Callable] = None,
        val_target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.data_root_dir = Path(data_root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_image_transform = train_image_transform
        self.train_target_transform = train_target_transform
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform

    def prepare_data(self):
        if not self.data_root_dir.exists():
            self.data_root_dir.mkdir()
            tar_file = _download_url(_IMAGEWOOF_URL, self.data_root_dir)
            self.data_root_dir = _untar_file(tar_file)
        else:
            self.data_root_dir = self.data_root_dir / "imagewoof2" 

    def setup(self, _):
        self._train_dataset = ImageFolder(
            self.data_root_dir / "train",
            transform=self.train_image_transform,
            target_transform=self.train_target_transform,
        )
        self._val_dataset = ImageFolder(
            self.data_root_dir / "val",
            transform=self.val_image_transform,
            target_transform=self.val_target_transform,
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self._train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self._train_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )