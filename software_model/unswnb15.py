#!/usr/bin/env python3

import os
import requests
import hashlib
import zipfile
import math
import numpy as np
import pandas as pd

dataset_url = "https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files"
dataset_sha256 = "abe2cd88012fea2a897e996ecbdfd4ee2ab99badf4584cf3db4a2a9929c4cb35"
dataset_dir = "./dataset"
dataset_fname = os.path.join(dataset_dir, "UNSW-NB15.zip")

def download_dset():
    os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.exists(dataset_fname):
        download = True
    else:
        with open(dataset_fname, "rb") as f:
            file_sha256 = hashlib.sha256(f.read()).hexdigest()
        if file_sha256 != dataset_sha256:
            print("Dataset zip file exists but sha256 does not match; will re-download")
            download = True
        else:
            print("Dataset zip exists and checksum matches; skipping download")
            download = False
    if download:
        print("Downloading UNSW-NB15 dataset")
        with open(dataset_fname, "wb") as f:
            f.write(requests.get(dataset_url, allow_redirects=True).content)
        with open(dataset_fname, "rb") as f:
            file_sha256 = hashlib.sha256(f.read()).hexdigest()
        assert(file_sha256 == dataset_sha256)

def unpack_dset():
    print("Unpacking dataset")
    assert(os.path.exists(dataset_fname))
    with zipfile.ZipFile(dataset_fname, "r") as z:
        z.extractall(dataset_dir)

def preprocess_dset():
    print("Preprocessing dataset")
    label_fname = os.path.join(dataset_dir, "UNSW-NB15 - CSV Files", "NUSW-NB15_features.csv")
    inp_fnames = [os.path.join(dataset_dir, "UNSW-NB15 - CSV Files", f"UNSW-NB15_{c}.csv") for c in range(1, 4+1)]

    df_labels = pd.read_csv(label_fname, encoding="cp1252")

    inp_dfs = []
    for fname in inp_fnames:
        df = pd.read_csv(fname, names=df_labels["Name"])
        df = df[df_labels.loc[df_labels["Type "] != "nominal"]["Name"]] # Drop nominal (text) columns
        inp_dfs.append(df)

    df_inp = pd.concat(inp_dfs, ignore_index=True)
    df_inp = df_inp.fillna(0)
    df_inp = df_inp.replace(r"^\s*$", 0, regex=True)
    df_inp = df_inp.apply(lambda x: (int(x) if x.isnumeric() else (int(x, 16) if x[0:2] == "0x" else float("NaN"))) if (type(x) == str) else x)
    df_inp = df_inp.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    df_inp = df_inp.dropna()
    df_inp = df_inp.drop_duplicates()

    df_test = df_inp.sample(frac=0.1, random_state=0)
    df_train = df_inp.drop(df_test.index)

    df_0 = df_train[df_train["Label"] == 0]
    df_1 = df_train.drop(df_0.index)

    df_0 = df_0.iloc[np.concatenate((np.arange(len(df_0)), np.random.choice(np.arange(len(df_0)), max(len(df_1)-len(df_0), 0))))]
    df_1 = df_1.iloc[np.concatenate((np.arange(len(df_1)), np.random.choice(np.arange(len(df_1)), max(len(df_0)-len(df_1), 0))))]
    df_train = pd.concat([df_0, df_1], ignore_index=True)

    df_train.to_csv(os.path.join(dataset_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)


if __name__ == "__main__":
    #download_dset()
    #unpack_dset()
    preprocess_dset()

