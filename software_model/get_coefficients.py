#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt

from main_backprop import get_dataset

model_fname = sys.argv[1]
if torch.cuda.is_available():
    default_device = "cuda"
else:
    default_device = "cpu"
model = torch.load(model_fname).to(default_device)
if len(sys.argv) > 3:
    model.bias[0] += int(sys.argv[3])

_, test_set = get_dataset()

test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=256)


tp = 0
tn = 0
fp = 0
fn = 0
positive_skews = []
negative_skews = []
device = next(model.parameters()).device
for features, labels in test_loader:
    features, labels = features.to(device), labels.to(device)
    outputs = model(features)

    _, predicted = torch.max(outputs.data, 1)
    tp += ((predicted == 1) * (labels == 1)).sum().item()
    tn += ((predicted == 0) * (labels == 0)).sum().item()
    fp += ((predicted == 1) * (labels == 0)).sum().item()
    fn += ((predicted == 0) * (labels == 1)).sum().item()

    skews = outputs.data[:,1] - outputs.data[:,0]
    for i in range(features.shape[0]):
        if labels[i] == 1:
            positive_skews.append(skews[i].item())
        else:
            negative_skews.append(skews[i].item())
positive_skews = np.array(positive_skews)
negative_skews = np.array(negative_skews)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2*(precision*recall)/(precision+recall)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")

print(f"G-measure: {(precision*recall)**0.5}")

mcc = ((tp*tn)-(fp*fn))/(((tp+fn)*(tn+fp)*(tn+fn)*(tp+fp))**0.5)
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

print(f"FPR: {fp/(tn+fp)}")
print(f"FNR: {fn/(tp+fn)}")

min_skew = int(np.floor(min(positive_skews.min(), negative_skews.min())))
max_skew = int(np.ceil(max(positive_skews.max(), negative_skews.max())))
skew_tpr = []
skew_fpr = []
for s in range(-max_skew-1, -min_skew+1):
    skew_tp = ((positive_skews + s) >= 0).sum()
    skew_tn = ((negative_skews + s) < 0).sum()
    skew_fp = len(negative_skews) - skew_tn
    skew_fn = len(positive_skews) - skew_tp
    skew_tpr.append(skew_tp / (skew_tp + skew_fn))
    skew_fpr.append(skew_fp / (skew_fp + skew_tn))
auroc = 0.0
for i in range(len(skew_tpr)-1):
    delta_fpr = skew_fpr[i+1] - skew_fpr[i]
    mean_tpr = (skew_tpr[i] + skew_tpr[i+1]) / 2
    auroc += delta_fpr * mean_tpr
print(f"Area under ROC: {auroc}")
end_idx = skew_tpr.index(1.0) + 1
plt.plot(skew_fpr[:end_idx], skew_fpr[:end_idx], color="#ff0000", linestyle="dashed")
plt.plot(skew_fpr[:end_idx], skew_tpr[:end_idx])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig(os.path.splitext(model_fname)[0] + "_ROC.png", bbox_inches="tight")

print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn)}")

total_model_size = model.unit_entries * model.mask.sum()
print(f"Total model size (bits): {total_model_size}")

