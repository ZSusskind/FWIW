#!/usr/bin/env python3

import torch
import torch.nn as nn

import os
import sys
import signal
import numpy as np
import pandas as pd

import unswnb15
from backprop_wisard import BackpropWiSARD

# Run inference using dataset (validation or test set)
def run_inference(model, dset_loader):
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

Abort_Training = False
def sigint_handler(signum, frame):
    global Abort_Training
    if not Abort_Training:
        print("Will abort training at end of epoch")
        Abort_Training = True
    else:
        sys.exit("Quitting immediately on second SIGINT")

# Train pre-specified model
if torch.cuda.is_available():
    default_device = "cuda"
else:
    print("WARNING: CUDA is not available; expect longer training times")
    default_device = "cpu"
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device=default_device):
    global Abort_Training
    Abort_Training = False
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

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
            model.clamp()

        model.eval()
        val_total, val_correct = run_inference(model, val_loader)
        print(f"At end of epoch {epoch}: "\
                f"Train set: Correct: {train_correct}/{train_total} ({round(((100*train_correct)/train_total).item(), 3)}%); "\
                f"Validation set: Correct: {val_correct}/{val_total} ({round(((100*val_correct)/val_total).item(), 3)}%)")
        
        if Abort_Training:
            break
    
    model.eval()
    signal.signal(signal.SIGINT, old_handler)
    
    model = model.to("cpu")
    return model

def compute_model_size(num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes, bits_per_input):
    filters_per_discriminator = int(np.ceil((num_inputs * bits_per_input) / unit_inputs))
    total_filters = filters_per_discriminator * num_classes
    total_model_size = total_filters * unit_entries
    model_size_k = round(total_model_size / (2**13), 2)
    print(f"Total model size: {model_size_k} KiB")

def get_dataset():
    if (not os.path.exists("dataset/train.csv")) or (not os.path.exists("dataset/test.csv")):
        unswnb15.download_dset()
        unswnb15.unpack_dset()
        unswnb15.preprocess_dset()
    train_df = pd.read_csv("dataset/train.csv")
    train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)
    test_df = pd.read_csv("dataset/test.csv")
    train_features = torch.FloatTensor(train_df.drop(["Label"], axis=1).values)
    train_classes = torch.LongTensor(train_df["Label"].values)
    test_features = torch.FloatTensor(test_df.drop(["Label"], axis=1).values)
    test_classes = torch.LongTensor(test_df["Label"].values)
    train_dataset = torch.utils.data.TensorDataset(train_features, train_classes)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_classes)
    
    return train_dataset, test_dataset

def create_model(unit_inputs, unit_entries, unit_hashes, bits_per_input,
        model_fname, num_epochs=100, learning_rate=1e-3, num_workers=4):
    batch_size = 32

    train_dataset, test_dataset = get_dataset()
    num_inputs = train_dataset.tensors[0].shape[1]
    num_classes = (train_dataset.tensors[1].amax() + 1).item()

    print(f"Num inputs/classes: {num_inputs}/{num_classes}")
    print(unit_inputs, unit_entries, unit_hashes, bits_per_input)
    print(f"Batch size: {batch_size}")
    compute_model_size(num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes, bits_per_input)
    
    torch.manual_seed(0) # For reproducability
    split_idx = int(len(train_dataset) * 0.9)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [split_idx, len(train_dataset)-split_idx])
    test_set = test_dataset

    train_data = torch.stack(tuple(t[0].flatten() for t in train_set))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
   
    model = BackpropWiSARD(num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes, bits_per_input, train_data)

    model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    
    test_total, test_correct = run_inference(model, test_loader)
    print(f"Test set: Correct: {test_correct}/{test_total} ({round(((100*test_correct)/test_total).item(), 3)}%)")
    torch.save(model, model_fname)

    return model

if __name__ == "__main__":
    model_fname = "unswnb15.pt"
    if len(sys.argv) > 1:
        model_fname = sys.argv[1]
    
    model = create_model(10, 64, 1, 4,
            model_fname, num_epochs=10, learning_rate=4e-3)

