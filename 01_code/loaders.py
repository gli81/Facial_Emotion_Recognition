# -*- coding: utf-8 -*-

"""
functions to get different loaders
"""

from torchvision import transforms
from dataset import FacialImageData
from torch.utils.data import DataLoader
from custom_transforms import ImgMask

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 100
MASK_DICT = {
    "full": [],
    "upper": [1, 2],
    "lower": [3, 4],
    "left": [2, 3],
    "right": [1, 4],
    "diagonal": [2, 4],
    "transdiagonal": [1, 3]
}

def get_loader(
    mask: "str"="full",
    train: "bool"=True,
):
    
    if mask not in MASK_DICT:
        raise ValueError("Invalid mask parameter")
    dir_ = "./data/test"
    if train:
        dir_ = "./data/train"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            ImgMask(MASK_DICT[mask]),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ]
    )
    set_ = FacialImageData(
        directory=dir_,
        transform=transform
    )
    batch_size = TRAIN_BATCH_SIZE if train else VAL_BATCH_SIZE
    loader_ = DataLoader(
        set_,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2 ## just to fit my device
    )
    return loader_




