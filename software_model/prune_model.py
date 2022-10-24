#!/usr/bin/env python3

import torch
import torch.nn as nn

import os
import sys
import numpy as np

from backprop_wisard import BackpropWiSARD
from main_backprop import get_dataset, train_model, run_inference

class BiasModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return x + self.bias

def run_bias_inference(model, dset_loader):
    total = 0
    correct = 0
    device = next(model.parameters()).device
    for features, labels in dset_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return total, correct

# Train pre-specified model
def train_bias_model(model, train_loader, val_loader, num_epochs=8, learning_rate=1e-3, device="cuda"):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        train_total = 0
        train_correct = 0
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
          
            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_total, val_correct = run_bias_inference(model, val_loader)
        print(f"At end of epoch {epoch}: "\
                f"Train set: Correct: {train_correct}/{train_total} ({round(((100*train_correct)/train_total).item(), 3)}%); "\
                f"Validation set: Correct: {val_correct}/{val_total} ({round(((100*val_correct)/val_total).item(), 3)}%)")
    
    model.eval()
    model = model.to("cpu")
    return model

def process_dset(model, dset_loader, device="cuda"):
    results = None
    labels = torch.empty(len(dset_loader.dataset), dtype=torch.long)
    model = model.to(device)
    model.eval()
    idx = 0
    for features, l in dset_loader:
        features = features.to(device)
        outputs = model(features)
        if idx == 0:
            results = torch.empty(len(dset_loader.dataset), *outputs.shape[1:])
        results[idx:idx+features.shape[0]] = outputs.detach().cpu()
        labels[idx:idx+features.shape[0]] = l.cpu()
        idx += features.shape[0]

    return torch.utils.data.TensorDataset(results, labels)

def compute_new_model_size(model, ratio):
    eff_filters_per_discrim = model.filters_per_discrim - int(ratio * model.filters_per_discrim)
    total_filters = eff_filters_per_discrim * model.num_classes
    total_model_size = total_filters * model.unit_entries
    model_size_k = round(total_model_size / (2**13), 2)
    print(f"Target model size: {model_size_k} KiB")

def prune_model(model, ratio=0.5, device="cuda"):
    batch_size = 32
    
    compute_new_model_size(model, ratio)

    train_dataset, test_dataset = get_dataset()

    torch.manual_seed(0) # For reproducability
    split_idx = int(len(train_dataset) * 0.9)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [split_idx, len(train_dataset)-split_idx])
    test_set = test_dataset
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=256)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=256)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=256)

    model = model.to(device)
    model.eval()
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Before pruning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")

    num_classes = model.num_classes

    print("Compute scores")
    scores = torch.zeros((num_classes, model.filters_per_discrim))
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        responses = model.get_filter_responses(features)
        for sample_idx in range(features.shape[0]):
            label = labels[sample_idx]
            #resprnses[sample_idx][label] *= -1
            responses[sample_idx][label] *= -(num_classes-1)
        scores -= responses.sum(axis=0).cpu().detach()
    scores = scores.amax(axis=0)

    print("Zero filters")
    prune_count = int(ratio * model.filters_per_discrim)
    prune_idxs = (-scores).topk(prune_count).indices
    # Shrink the model
    keep_idxs = torch.LongTensor([i for i in range(model.filters_per_discrim) if i not in prune_idxs])
    input_order = model.input_order
    new_input_order = torch.cat([input_order[model.unit_inputs*i:model.unit_inputs*(i+1)] for i in keep_idxs])
    new_data = model.data[:,keep_idxs].clone()
    if hasattr(model, "bleach_weights"):
        new_bleach_weights = model.bleach_weights[:,keep_idxs]
    with torch.no_grad():
        model.input_order.resize_(len(new_input_order))
        # PyTorch refuses to resize tensors which require grad - so we use a hack
        model.data.requires_grad = False
        model.data.resize_(num_classes, len(keep_idxs), model.unit_entries)
        print(num_classes, len(keep_idxs), model.unit_entries)
        model.data.requires_grad = True
        model.mask.resize_(num_classes, len(keep_idxs))
        if hasattr(model, "bleach_weights"):
            model.bleach_weights.resize_(num_classes, len(keep_idxs), model.unit_entries)
        model.input_order.data = new_input_order.to(device)
        model.data = nn.Parameter(new_data.to(device))
        model.mask = nn.Parameter(torch.ones((num_classes, len(keep_idxs))).to(device), requires_grad=False)
        model.filters_per_discrim = len(keep_idxs)
        if hasattr(model, "bleach_weights"):
            model.bleach_weights = nn.Parameter(new_bleach_weights, requires_grad=False)

    test_total, test_correct = run_inference(model, test_loader)
    print(f"After pruning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")
    
    print("Learn biases")
    model_train_dset = process_dset(model, train_loader, device="cuda")
    model_val_dset = process_dset(model, val_loader, device="cuda")
    model_train_loader = torch.utils.data.DataLoader(dataset=model_train_dset, batch_size=batch_size)
    model_val_loader = torch.utils.data.DataLoader(dataset=model_val_dset, batch_size=batch_size)
    bias_model = BiasModel(num_classes)
    bias_model = train_bias_model(bias_model, model_train_loader, model_val_loader)
    model.bias.data = bias_model.bias.data.detach().to(device)
  
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Before fine-tuning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")

    print("Fine-tune model")
    model = train_model(model, train_loader, val_loader, num_epochs=10)
    
    test_total, test_correct = run_inference(model, test_loader)
    print(f"After fine-tuning: Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")
    return model

if __name__ == "__main__":
    model_fname, ratio = sys.argv[1], float(sys.argv[2])
    ratio = min(max(ratio, 0.0), 1.0)
    out_fname = os.path.splitext(model_fname)[0] + f"_pruned_{str(ratio).replace('.', '_')}.pt"
    print(f"Out filename is {out_fname}")
    model = torch.load(model_fname)
    pruned_model = prune_model(model, ratio=ratio)
    torch.save(pruned_model, out_fname)
        
