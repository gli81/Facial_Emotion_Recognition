# -*- coding: utf-8 -*-

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FacialImageData
from custom_transforms import ImgMask
from torchvision.transforms.functional import to_pil_image


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet50 = models.resnet50()
    num_classes = 7 ## how many classes in fer?
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
    pretrained = torch.load("./saved_model/resnet50_on_FER.pth")
    resnet50.load_state_dict(pretrained["state_dict"])
    resnet50.to(device)
    mask_upper_sample = transforms.Compose(
        [
            transforms.ToTensor(),
            ImgMask([1, 2])
        ]
    )
    upper_val_set = FacialImageData(
        directory="./data" + "/test",
        transform=mask_upper_sample
    )
    upper_mask_val_loader = DataLoader(
        upper_val_set,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )
    for idx, (inputs, target) in enumerate(upper_mask_val_loader):
        image = to_pil_image(inputs[0])
        image.save("test.png")
        break

if __name__ == "__main__":
    main()
