from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "ellipticco/elliptic-data-set/versions/1"

DATA_SUBDIR = "elliptic_bitcoin_dataset"
FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

LOCAL_FEATURES = [f"local_{i}" for i in range(1, 94)]
AGG_FEATURES = [f"agg_{i}" for i in range(1, 73)]
FEATURES = LOCAL_FEATURES + AGG_FEATURES

LABELS = {"1": "illicit", "2": "licit", "unknown": "unknown"}
TARGET = {"illicit": 1, "licit": 0}


def _data_dir():
    return Path(kagglehub.dataset_download(DATASET)) / DATA_SUBDIR


def load_features():
    features = pd.read_csv(_data_dir() / FEATURES_FILE, header=None)
    features.columns = ["txId", "time_step"] + FEATURES
    return features


def load_txs():
    features = load_features()
    classes = pd.read_csv(_data_dir() / CLASSES_FILE, dtype={"class": str})
    txs = features.merge(classes, on="txId", how="left", validate="one_to_one")
    txs["label"] = txs["class"].map(LABELS)
    txs["y"] = txs["label"].map(TARGET)
    return txs.drop(columns="class")


def load_edges():
    return pd.read_csv(_data_dir() / EDGES_FILE)
