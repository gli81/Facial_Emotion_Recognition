# -*- coding: utf-8 -*-

import torch
import os
from PIL import Image

## define dataset
class FacialImageData(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all the images and labels
        for label, subdir in enumerate(sorted(os.listdir(directory))):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                for img_name in os.listdir(subdir_path):
                    if not os.path.isdir(os.path.join(subdir_path, img_name)):
                        self.images.append(os.path.join(subdir_path, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label