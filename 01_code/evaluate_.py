# -*- coding: utf-8 -*-

from typing import Dict
import torch
import torchvision.models as models
import torch.nn as nn
import loaders

## TODO load a base/finetuned model
## TODO get corresponding validation loader
## validate on that loader, record the predicted and the target value

def evaluate_and_get_metrics(
    model_: "str",
    evaluate_on: "str",
    device: "str"
) -> "Dict":
    ## load model based on parameter model_
    if model_ not in ["baseline", "upper", "lower", "left", "right"]:
        raise ValueError("Invalid model")
    model = models.resnet50()
    num_classes=7
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    pretrained = torch.load("./saved_model/" + model_ + ".pth")
    model.load_state_dict(pretrained["state_dict"])
    model.to(device)
    model.eval()
    ## get loader based on the designated type of image
    loader = loaders.get_loader(
        mask=evaluate_on,
        train=False,
        shuffle=True
    )
    loss_func = nn.CrossEntropyLoss().to(device)
    ## evaluate on designated type of image
    with torch.no_grad():
        val_loss = 0
        target_examples = []
        predicted_examples = []
        total_examples_ct = 0
        correct_examples_ct = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            # copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute the output and loss
            out = model(inputs)
            loss = loss_func(out, targets)
            # count the number of correctly predicted samples
            # in the current batch
            _, predicted = torch.max(out, 1)
            correct = predicted.eq(targets).sum()
            val_loss += loss.detach().cpu()
            total_examples_ct += targets.shape[0]
            correct_examples_ct += correct.item()
            predicted_examples.extend(predicted.cpu().tolist())
            target_examples.extend(targets.cpu().tolist())
    avg_loss = val_loss / len(loader)
    avg_acc = correct_examples_ct / total_examples_ct
    print(
        "Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc)
    )


