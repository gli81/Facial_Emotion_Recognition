# -*- coding: utf-8 -*-

from typing import Dict
import torch
import torchvision.models as models
import torch.nn as nn
import loaders
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import seaborn as sns

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
    if evaluate_on not in ["baseline", "upper", "lower", "left", "right"]:
        raise ValueError("Invalid evaluation")
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
    ## evaluate on designated type of image
    with torch.no_grad():
        total_examples_ct = 0
        correct_examples_ct = 0
        ## record predict_proba, predicted, true
        target_all = np.empty((0))
        predicted_all = np.empty((0))
        predicted_proba_all = np.empty((0, num_classes))
        for batch_idx, (inputs, targets) in enumerate(loader):
            # copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute the output
            out = model(inputs)
            predicted_proba_all = np.concatenate(
                (predicted_proba_all, softmax(out.cpu().numpy())), axis=0
            )
            # count the number of correctly predicted samples
            # in the current batch
            _, predicted = torch.max(out, 1)
            correct = predicted.eq(targets).sum()
            total_examples_ct += targets.shape[0]
            correct_examples_ct += correct.item()
            predicted_all = np.concatenate(
                (predicted_all, predicted.cpu().numpy()), axis=0
            )
            target_all = np.concatenate(
                (target_all, targets.cpu().numpy()), axis=0
            )
    avg_acc = correct_examples_ct / total_examples_ct
    ## create a one hot encoding version, for metrics calculation
    target_ohe_all = np.eye(num_classes)[target_all.astype(int)]
    predicted_ohe_all = np.eye(num_classes)[predicted_all.astype(int)]
    ans = {}
    ## overall accuracy, save in ans, and return
    ans["overall_accuracy"] = avg_acc
    ## class-wise accuracy, save in ans, and return
    cm = confusion_matrix(target_all, predicted_all)
    classwise_acc = np.diag(cm) / np.sum(cm, axis=1)
    class_labels = [0, 1, 2, 3, 4, 5, 6]
    ans["classwise_accuracy"] = {
        label: acc for label, acc in zip(class_labels, classwise_acc)
    }
    print(ans["classwise_accuracy"])
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    if not os.path.exists(f"./results/{model_}"):
        os.makedirs(f"./results/{model_}")
    ## ensure replace
    if os.path.exists(f"./results/{model_}/{evaluate_on}"):
        shutil.rmtree(f"./results/{model_}/{evaluate_on}")
    os.makedirs(f"./results/{model_}/{evaluate_on}")
    ## roc plot for each class, save plot
    ans["classwise_fpr"], ans["classwise_tpr"], ans["classwise_auc"] = \
        plot_class_wise_roc(
            target_ohe_all, predicted_proba_all,
            f"./results/{model_}/{evaluate_on}",
            True
        )
    ## macro-roc curve, which treat each class equally, save plot
    ans["macro_fpr"], ans["macro_fpr"], ans["macro_auc"] = plot_macro_roc(
        ans["classwise_fpr"], ans["classwise_tpr"],
        f"./results/{model_}/{evaluate_on}"
    )
    ## micro-roc curve, which treat each sample equally, save plot
    ans["micro_fpr"], ans["micro_tpr"], ans["micro_auc"] = plot_micro_roc(
        target_ohe_all, predicted_proba_all,
        f"./results/{model_}/{evaluate_on}"
    )
    ## weighted-average-roc curve, which considers imbalance between classes
    ans["weighted_average_fpr"], ans["weighted_average_tpr"], ans["weighted_average_auc"] = plot_wa_roc(
        ans["classwise_fpr"], ans["classwise_tpr"], target_ohe_all,
        f"./results/{model_}/{evaluate_on}"
    )
    ## macro-F1 score, treat each class equally, return value
    ans["macro_f1"] = f1_score(target_all, predicted_all, average="macro")
    ## micro-F1 score, treat each sample equally, return value
    ans["micro_f1"] = f1_score(target_all, predicted_all, average="micro")
    ## TODO weighted-average F1 score, consider imbalance, return value
    ans["weighted_average_f1"] = f1_score(target_all, predicted_all, average="weighted")
    ## create confusion matrix, save plot
    get_confusion_matrix(
        target_all, predicted_all,
        f"./results/{model_}/{evaluate_on}"
    )
    return ans


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def plot_class_wise_roc(
        target_ohe, predict_proba, path, save=False
    ):
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure()
    for i in range(target_ohe.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(target_ohe[:, i], predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i],
            label=f"Class {i + 1} ROC curve (AUC={roc_auc[i]:.4f})"
        )
    plt.legend()
    ## save plot
    if save:
        plt.savefig(f"{path}/classwise_roc.png")
    plt.show()
    return fpr, tpr, roc_auc

def plot_macro_roc(fpr, tpr, path):
    all_fpr = np.unique(
        np.concatenate([fpr[i] for i in range(len(fpr))])
    )
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(fpr)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Average it and compute AUC
    mean_tpr /= len(fpr)
    # Finally, compute the AUC
    macro_auc = auc(all_fpr, mean_tpr)
    plt.figure()
    plt.plot(
        all_fpr, mean_tpr,
        label=f"Macro-average ROC curve (area = {macro_auc:.4f})",
    )
    plt.legend()
    plt.savefig(f"{path}/macro_average_roc.png")
    plt.show()
    return all_fpr, mean_tpr, macro_auc

def plot_micro_roc(target_ohe, predict_proba, path):
    fpr, tpr, _  = roc_curve(target_ohe.ravel(), predict_proba.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Micro average ROC (AUC: {roc_auc:.4f})")
    plt.legend()
    plt.savefig(f"{path}/micro_average_roc.png")
    plt.show()
    return fpr, tpr, roc_auc

def plot_wa_roc(fpr, tpr, target_ohe, path):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(fpr))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(fpr)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) * (sum(target_ohe[:, i]) / target_ohe.shape[0])
    weighted_roc_auc = auc(all_fpr, mean_tpr)
    plt.figure()
    plt.plot(all_fpr, mean_tpr, label=f"Weighted average ROC (AUC: {weighted_roc_auc:.4f})")
    plt.legend()
    plt.savefig(f"{path}/weighted_average_roc.png")
    plt.show()
    return all_fpr, mean_tpr, weighted_roc_auc

def get_confusion_matrix(y_true, y_pred, path):
    plt.figure()
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt='d',
        cmap="Blues"
    )
    plt.savefig(f"{path}/confusion_matrix.png")
    plt.show()
