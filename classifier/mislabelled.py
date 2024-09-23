# -*- coding: utf-8 -*-

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FacialImageData
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn
import os
import shutil


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet50 = models.resnet50()
    num_classes = 7
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
    pretrained = torch.load("./saved_model/resnet50_on_FER.pth")
    resnet50.load_state_dict(pretrained["state_dict"])
    resnet50.to(device)
    resnet50.eval()
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ]
    )
    val_set = FacialImageData(
        directory="./data" + "/test",
        transform=val_transform
    )
    val_loader = DataLoader(
        val_set,
        batch_size=100,
        shuffle=True,
        num_workers=2
    )
    loss_func = nn.CrossEntropyLoss().to(device)
    means = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(-1, 1, 1)
    stds = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(-1, 1, 1)
    for idx, (inputs, target) in enumerate(val_loader):
        # copy inputs to device
        inputs = inputs.to(device)
        target = target.to(device)
        # compute the output and loss
        out = resnet50(inputs)
        loss = loss_func(out, target)
        ## get predicted results
        _, predicted = torch.max(out, 1)
        # print(predicted)
        # print(target)
        mislabelled = (predicted != target)
        print(mislabelled)
        mislabelled_indcies = [i for i, mis in enumerate(mislabelled) if mis]
        print(mislabelled_indcies)
        if os.path.exists("./mislabelled"):
            shutil.rmtree("./mislabelled")
        os.makedirs("./mislabelled")
        for index in mislabelled_indcies:
            img = inputs[index]
            ## restore the image by denormalizing
            denormalized = img * stds + means
            denormalized = torch.clamp(denormalized, 0, 1)
            # print(denormalized)
            # print(denormalized.shape)
            # print(predicted[idx].item())
            # print(target[idx].item())
            to_save_img = to_pil_image(denormalized)
            to_save_img.save(f"./mislabelled/{index}_{predicted[index].item()}_{target[index].item()}.png")
            # break
        # val_loss += loss.detach().cpu()
        # total_examples += target.shape[0]
        # correct_examples += correct.item()
        # state = {'state_dict': resnet50.state_dict(),
        #         'epoch': i,
        #         'lr': current_learning_rate}
        # torch.save(state, os.path.join(CHECKPOINT_FOLDER, 'resnet50_on_FER.pth'))
        break

if __name__ == "__main__":
    main()
