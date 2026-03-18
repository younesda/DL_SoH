"""
train_lstm.py — Phase 3 : Entraînement LSTM (approche inter-cycle)
Battery SoH LSTM Project
Auteur : Younes Hachami

Input shape : (5, 7)  <- 5 cycles x 7 features agrégées par cycle
Dataset     : ~1 300 samples (vs 23 000 en intra-cycle)

Usage :
  python train_lstm.py           # Run #3  — LSTM(64) baseline inter-cycle
  python train_lstm.py --run 4   # Run #4  — LSTM(128) larger
  python train_lstm.py --run 5   # Run #5  — stacked LSTM
  python train_lstm.py --run 6   # Run #6  — BiLSTM
  python train_lstm.py --run 7   # Run #7  — BiLSTM window=10 (11feat v4b)
  python train_lstm.py --run 8   # Run #8  — BiLSTM(128) window=10 (11feat v4b)
  python train_lstm.py --run 9   # Run #9  — BiLSTM(64) window=10 (12feat v5)
  python train_lstm.py --run 10  # Run #10 — 2xBiLSTM(64) stacked + weight_decay + CosineAnnealing (11feat w=10)
  python train_lstm.py --run 11  # Run #11 — BiLSTM(64) + Attention + HuberLoss + weight_decay (11feat w=10)
  python train_lstm.py --run 12  # Run #12 — BiLSTM(64) + Attention + HuberLoss + cycle_number_norm (12feat v6)
  python train_lstm.py --run 13  # Run #13 — BiLSTM(64) + Attention + HuberLoss + soh_prev (12feat v7)
  python train_lstm.py --ensemble  # Ensemble pondéré des meilleurs runs (filtre R²>0.75)
"""

import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parents[2]
_DATA    = _ROOT / "data" / "processed" / "final"
_EXP     = _ROOT / "experiments"
META_JSON  = str(_DATA / "metadata_v4b.json")
OUT_DIR    = _ROOT

WINDOW_SIZE = 5    # cycles
N_FEATURES  = 7    # features agrégées par cycle (v2)
FEATURE_NAMES = ["mean_V", "std_V", "min_V",
                 "mean_T", "std_T", "mean_I", "slope_SoC"]

# Fichiers de données par run (run 5+ utilisent les features enrichies v3)
# Tuple: (x_train_file, x_test_file, n_features, y_suffix)
# y_suffix=None  -> y_train.npy / y_test.npy
# y_suffix="v4b" -> y_train_v4b.npy / y_test_v4b.npy
RUN_DATA = {
    3: ("X_train.npy",    "X_test.npy",    7,  None),
    4: ("X_train.npy",    "X_test.npy",    7,  None),
    5: ("X_train_v3.npy", "X_test_v3.npy", 11, None),  # BiLSTM + 11 features
    6: ("X_train_v3.npy", "X_test_v3.npy", 11, None),
    7: ("X_train_v4b.npy","X_test_v4b.npy",11, "v4b"), # BiLSTM window=10
    8: ("X_train_v4b.npy","X_test_v4b.npy",11, "v4b"), # BiLSTM(128) window=10
    9: ("X_train_v5.npy", "X_test_v5.npy", 12, "v4b"), # BiLSTM(64)  window=10, 12feat
   10: ("X_train_v4b.npy","X_test_v4b.npy",11, "v4b"), # 2xBiLSTM(64) stacked window=10
   11: ("X_train_v4b.npy","X_test_v4b.npy",11, "v4b"), # BiLSTM(64)+Attention window=10
   12: ("X_train_v6.npy", "X_test_v6.npy", 12, "v6"),  # BiLSTM(64)+Attention+cycle_norm window=10
   13: ("X_train_v7.npy", "X_test_v7.npy", 12, "v7"),  # BiLSTM(64)+Attention+soh_prev window=10
   14: ("X_train_v8.npy", "X_test_v8.npy", 12, "v8"),  # BiLSTM(64)+Attention+soh_delta window=10
   15: ("X_train_v7.npy", "X_test_v7.npy", 12, "v7"),  # BiLSTM(64)+Attention+soh_prev, bat-holdout val
}

# Window size par run (défaut 5 pour runs 3-6)
RUN_WINDOW = {
    7: 10,
    8: 10,
    9: 10,
   10: 10,
   11: 10,
   12: 10,
   13: 10,
   14: 10,
   15: 10,
}

# Hyperparamètres adaptés au petit dataset inter-cycle
BATCH_SIZE   = 32    # réduit (dataset ~1 300 samples)
MAX_EPOCHS   = 150   # augmenté (moins d'itérations/epoch)
LR           = 1e-3
PATIENCE_ES  = 20    # early stopping
PATIENCE_LR  = 8     # ReduceLROnPlateau
LR_FACTOR    = 0.5
MIN_LR       = 1e-5
VAL_RATIO    = 0.20  # 20% du train (augmenté vs 15%)

# Baseline inter-cycle (Ridge sur les nouveaux tenseurs)
BASELINE = {"MAE": 3.507, "RMSE": 4.368, "R2": 0.707}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
class LSTMv3(nn.Module):
    """
    Run #3 — LSTM(64) -> Dropout(0.2) -> Dense(32) -> Dense(1)
    Input : (batch, 5 cycles, 7 features)
    """
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(N_FEATURES, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(64, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class LSTMv4(nn.Module):
    """
    Run #4 — BiLSTM(64) -> Dropout(0.3) -> Dense(64) -> Dense(32) -> Dense(1)
    merge_mode='concat' => hidden=64*2=128 après BiLSTM.
    """
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(N_FEATURES, 64, batch_first=True,
                               bidirectional=True)   # output: (B, T, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(128, 64)
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(64, 32)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])    # dernier pas : (B, 128)
        out = self.relu1(self.fc1(out))      # (B, 64)
        out = self.relu2(self.fc2(out))      # (B, 32)
        return self.fc3(out).squeeze(-1)     # (B,)


class LSTMv5(nn.Module):
    """
    Run #5 — BiLSTM(64) sur 11 features (v3 enrichi)
    Même archi que Run #4 mais input_size=11.
    """
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, batch_first=True,
                               bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(128, 64)
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(64, 32)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        return self.fc3(out).squeeze(-1)


class LSTMv6(nn.Module):
    """
    Run #6 — BiLSTM(64, 11feat) -> Dropout(0.2) -> Dense(32) -> Dense(1)
    Version légère si v5 overfitte.
    """
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, batch_first=True,
                               bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(128, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class LSTMv7(nn.Module):
    """
    Run #7 — BiLSTM(64, 11feat, window=10) -> Dropout(0.2) -> Dense(32) -> Dense(1)
    Même archi que v6 ; seul le contexte temporel change (10 cycles vs 5).
    Input : (batch, 10 cycles, 11 features)
    """
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, batch_first=True,
                               bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(128, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class LSTMv9(nn.Module):
    """
    Run #9 — BiLSTM(64, 12feat, window=10) -> Dropout(0.2) -> Dense(32) -> Dense(1)
    Même archi que Run #7 (meilleur modèle) ; input_size=12 (ajout ir_proxy).
    Input : (batch, 10 cycles, 12 features)
    """
    def __init__(self, input_size: int = 12):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, batch_first=True,
                               bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(128, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class AdditiveAttention(nn.Module):
    """
    Attention additive (Bahdanau) sur les sorties BiLSTM.
    Input  : (batch, T, hidden_dim)
    Output : (batch, hidden_dim)  — vecteur de contexte
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):                   # (B, T, H)
        weights = torch.softmax(self.score(lstm_out), dim=1)   # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)              # (B, H)
        return context


class LSTMv11(nn.Module):
    """
    Run #11 — BiLSTM(64, 11feat, window=10) + Attention -> Dropout(0.2) -> FC(32) -> FC(1)
    Différences vs Run #7 (meilleur) :
      - Attention additive sur les T=10 sorties BiLSTM (au lieu de prendre seulement le dernier pas)
      - HuberLoss(delta=2.0) à la place de MSELoss (configuré dans train())
      - weight_decay=1e-4 dans Adam
    Input : (batch, 10 cycles, 11 features)
    """
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden_dim=128)
        self.dropout   = nn.Dropout(0.2)
        self.fc1       = nn.Linear(128, 32)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # (B, T, 128)
        ctx    = self.attention(out)     # (B, 128)
        ctx    = self.dropout(ctx)
        return self.fc2(self.relu(self.fc1(ctx))).squeeze(-1)


class LSTMv14(nn.Module):
    """
    Run #14 — BiLSTM(64, 12feat, window=10) + Attention -> Dropout(0.3) -> FC(32) -> FC(1)
    Différences vs Run #13 :
      - Données pipeline v8 (soh_delta au lieu de soh_prev absolu)
      - Dropout augmenté 0.2 → 0.3 pour réduire le gap train/val
    Input : (batch, 10 cycles, 12 features)
    """
    def __init__(self, input_size: int = 12):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden_dim=128)
        self.dropout   = nn.Dropout(0.3)
        self.fc1       = nn.Linear(128, 32)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # (B, T, 128)
        ctx    = self.attention(out)     # (B, 128)
        ctx    = self.dropout(ctx)
        return self.fc2(self.relu(self.fc1(ctx))).squeeze(-1)


class LSTMv10(nn.Module):
    """
    Run #10 — 2x BiLSTM(64, 11feat, window=10) stacked -> Dropout(0.2) -> FC(32) -> FC(1)
    Différences vs Run #7 (meilleur) :
      - 2 couches BiLSTM empilées (dropout=0.2 entre couches)
      - weight_decay=1e-4 dans Adam (L2 — configuré dans RUN_HPARAMS)
      - CosineAnnealingWarmRestarts au lieu de ReduceLROnPlateau
    Input : (batch, 10 cycles, 11 features)
    """
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(128, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


class LSTMv8(nn.Module):
    """
    Run #8 — BiLSTM(128, 11feat, window=10) -> Dropout(0.1)
             -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)
    Augmentation capacité vs v7 : LSTM units 64→128, dropout 0.2→0.1.
    BiLSTM output : 128*2 = 256 dim.
    Input : (batch, 10 cycles, 11 features)
    """
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 128, batch_first=True,
                               bidirectional=True)   # output: (B, T, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc1     = nn.Linear(256, 64)
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(64, 32)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])    # dernier pas : (B, 256)
        out = self.relu1(self.fc1(out))      # (B, 64)
        out = self.relu2(self.fc2(out))      # (B, 32)
        return self.fc3(out).squeeze(-1)     # (B,)


MODELS = {
    3:  (LSTMv3,  "LSTM(64)->Drop(0.2)->FC(32)->FC(1)                  [7feat w=5]"),
    4:  (LSTMv4,  "BiLSTM(64)->Drop(0.3)->FC(64)->FC(32)->FC(1)        [7feat w=5]"),
    5:  (LSTMv5,  "BiLSTM(64)->Drop(0.3)->FC(64)->FC(32)->FC(1)        [11feat w=5]"),
    6:  (LSTMv6,  "BiLSTM(64)->Drop(0.2)->FC(32)->FC(1)                [11feat w=5]"),
    7:  (LSTMv7,  "BiLSTM(64)->Drop(0.2)->FC(32)->FC(1)                [11feat w=10]"),
    8:  (LSTMv8,  "BiLSTM(128)->Drop(0.1)->FC(64)->FC(32)->FC(1)       [11feat w=10]"),
    9:  (LSTMv9,  "BiLSTM(64)->Drop(0.2)->FC(32)->FC(1)                [12feat w=10]"),
    10: (LSTMv10, "2xBiLSTM(64)->Drop(0.2)->FC(32)->FC(1)             [11feat w=10]"),
    11: (LSTMv11, "BiLSTM(64)+Attn->Drop(0.2)->FC(32)->FC(1)         [11feat w=10]"),
    12: (LSTMv11, "BiLSTM(64)+Attn->Drop(0.2)->FC(32)->FC(1)         [12feat v6 w=10]"),
    13: (LSTMv11, "BiLSTM(64)+Attn->Drop(0.2)->FC(32)->FC(1)         [12feat v7+soh_prev w=10]"),
    14: (LSTMv14, "BiLSTM(64)+Attn->Drop(0.3)->FC(32)->FC(1)         [12feat v8+soh_delta w=10]"),
    15: (LSTMv11, "BiLSTM(64)+Attn->Drop(0.2)->FC(32)->FC(1)         [12feat v7+soh_prev bat-holdout w=10]"),
}

# Hyperparamètres spécifiques par run
# weight_decay : L2 regularisation dans Adam (défaut 0)
# scheduler    : "plateau" (ReduceLROnPlateau) ou "cosine" (CosineAnnealingWarmRestarts)
RUN_HPARAMS = {
    3:  {"lr": 1e-3,  "patience_es": 20, "patience_lr":  8, "weight_decay": 0,    "scheduler": "plateau"},
    4:  {"lr": 5e-4,  "patience_es": 25, "patience_lr": 10, "weight_decay": 0,    "scheduler": "plateau"},
    5:  {"lr": 5e-4,  "patience_es": 25, "patience_lr": 10, "weight_decay": 0,    "scheduler": "plateau"},
    6:  {"lr": 5e-4,  "patience_es": 25, "patience_lr": 10, "weight_decay": 0,    "scheduler": "plateau"},
    7:  {"lr": 5e-4,  "patience_es": 25, "patience_lr": 10, "weight_decay": 0,    "scheduler": "plateau"},
    8:  {"lr": 5e-4,  "patience_es": 25, "patience_lr": 10, "weight_decay": 0,    "scheduler": "plateau"},
    9:  {"lr": 5e-4,  "patience_es": 25, "patience_lr": 10, "weight_decay": 0,    "scheduler": "plateau"},
    10: {"lr": 5e-4,  "patience_es": 30, "patience_lr": 10, "weight_decay": 1e-4, "scheduler": "cosine"},
    11: {"lr": 5e-4,  "patience_es": 30, "patience_lr": 10, "weight_decay": 1e-4, "scheduler": "plateau", "loss": "huber"},
    12: {"lr": 5e-4,  "patience_es": 30, "patience_lr": 10, "weight_decay": 1e-4, "scheduler": "plateau", "loss": "huber"},
    13: {"lr": 5e-4,  "patience_es": 30, "patience_lr": 10, "weight_decay": 1e-4, "scheduler": "plateau", "loss": "huber"},
    14: {"lr": 5e-4,  "patience_es": 30, "patience_lr": 10, "weight_decay": 1e-4, "scheduler": "plateau", "loss": "huber"},
    15: {"lr": 5e-4,  "patience_es": 30, "patience_lr": 10, "weight_decay": 1e-4, "scheduler": "plateau", "loss": "huber"},
}

# Batteries réservées comme val set (holdout batterie-niveau) pour certains runs.
# Avantage vs split temporel : le val set couvre toute la durée de vie (early+late)
# → même difficulté que le train → gap train/val plus réaliste.
RUN_VAL_BATTERY = {
    15: ["B0007"],   # 158 fenêtres = 16.7% de 944 — batterie la plus longue du train set
}


# ---------------------------------------------------------------------------
# Validation split temporel (inter-cycle)
# ---------------------------------------------------------------------------
def battery_holdout_val_split(X_train_np, y_train_np,
                               train_batteries: list,
                               val_batteries: list,
                               window_size: int = WINDOW_SIZE):
    """
    Réserve des batteries entières comme val set.
    Le val set couvre toute la durée de vie (early + late cycles)
    → même difficulté que le train → gap train/val réaliste.
    """
    df_cycle = _rebuild_window_order(train_batteries, window_size=window_size)

    train_idx, val_idx = [], []
    offset = 0

    for bat in train_batteries:
        bat_windows = df_cycle[df_cycle["battery_id"] == bat]["n_windows"].values
        if len(bat_windows) == 0:
            continue
        n_win = int(bat_windows[0])
        if n_win == 0:
            continue
        idx = np.arange(offset, offset + n_win)
        if bat in val_batteries:
            val_idx.extend(idx.tolist())
        else:
            train_idx.extend(idx.tolist())
        offset += n_win

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)

    log.info("Val split batterie-holdout %s — train: %d  val: %d  (ratio %.0f%%)",
             val_batteries, len(train_idx), len(val_idx),
             len(val_idx) / (len(train_idx) + len(val_idx)) * 100)

    return (X_train_np[train_idx], y_train_np[train_idx],
            X_train_np[val_idx],   y_train_np[val_idx])


def temporal_val_split(X_train_np, y_train_np,
                       train_batteries: list, val_ratio: float = VAL_RATIO,
                       window_size: int = WINDOW_SIZE):
    """
    Pour chaque batterie, prend les (1-val_ratio) premières fenêtres en train
    et les val_ratio dernières en validation.
    Les fenêtres sont déjà ordonnées par cycle croissant dans les .npy.
    """
    df_cycle = _rebuild_window_order(train_batteries, window_size=window_size)

    train_idx, val_idx = [], []
    offset = 0

    for bat in train_batteries:
        bat_windows = df_cycle[df_cycle["battery_id"] == bat]["n_windows"].values
        if len(bat_windows) == 0:
            continue
        n_win = int(bat_windows[0])
        if n_win == 0:
            continue

        idx = np.arange(offset, offset + n_win)
        n_val = max(1, math.ceil(n_win * val_ratio))

        val_idx.extend(idx[-n_val:].tolist())
        train_idx.extend(idx[:-n_val].tolist())
        offset += n_win

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)

    log.info("Val split temporel — train: %d  val: %d  (ratio %.0f%%)",
             len(train_idx), len(val_idx),
             len(val_idx) / (len(train_idx) + len(val_idx)) * 100)

    return (X_train_np[train_idx], y_train_np[train_idx],
            X_train_np[val_idx],   y_train_np[val_idx])


def _rebuild_window_order(train_batteries, window_size=WINDOW_SIZE):
    """Retourne un DF avec le nombre de fenêtres par batterie (même ordre que le pipeline)."""
    df = pd.read_csv(_ROOT / "data" / "raw" / "battery_health_dataset.csv")
    df["SoH"]                  = df["SoH"].clip(upper=100.0)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=60.0)

    records = []
    for bat in train_batteries:
        n_cycles = df[df["battery_id"] == bat]["cycle_number"].nunique()
        n_win    = max(0, n_cycles - window_size + 1)
        records.append({"battery_id": bat, "n_windows": n_win})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------
def train(run_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)

    meta       = json.load(open(META_JSON, encoding="utf-8"))
    run_entry  = RUN_DATA.get(run_id, ("X_train.npy", "X_test.npy", N_FEATURES, None))
    x_train_file, x_test_file, n_feat, y_suffix = run_entry
    y_train_file = f"y_train_{y_suffix}.npy" if y_suffix else "y_train.npy"
    y_test_file  = f"y_test_{y_suffix}.npy"  if y_suffix else "y_test.npy"
    win_size     = RUN_WINDOW.get(run_id, WINDOW_SIZE)

    X_train_np = np.load(_DATA / x_train_file)
    y_train_np = np.load(_DATA / y_train_file)
    X_test_np  = np.load(_DATA / x_test_file)
    y_test_np  = np.load(_DATA / y_test_file)
    log.info("Data : %s / %s  (y: %s / %s)", x_train_file, x_test_file,
             y_train_file, y_test_file)

    assert X_train_np.shape[1:] == (win_size, n_feat), \
        f"Shape inattendue : {X_train_np.shape} (attendu (N,{win_size},{n_feat}))"
    log.info("Shapes OK — X_train:%s  X_test:%s  window=%d", X_train_np.shape, X_test_np.shape, win_size)

    # Validation split
    val_bats = RUN_VAL_BATTERY.get(run_id)
    if val_bats:
        X_tr, y_tr, X_val, y_val = battery_holdout_val_split(
            X_train_np, y_train_np, meta["train_batteries"], val_bats, window_size=win_size
        )
    else:
        X_tr, y_tr, X_val, y_val = temporal_val_split(
            X_train_np, y_train_np, meta["train_batteries"], window_size=win_size
        )

    def to_loader(X, y, shuffle=True):
        ds = TensorDataset(
            torch.from_numpy(X).to(device),
            torch.from_numpy(y).to(device)
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          drop_last=False)

    train_loader = to_loader(X_tr,  y_tr,  shuffle=True)
    val_loader   = to_loader(X_val, y_val, shuffle=False)
    log.info("Loaders — train: %d samples (%d batches)  val: %d samples (%d batches)",
             len(y_tr), len(train_loader), len(y_val), len(val_loader))

    # Modèle  (LSTMv5/v6/v7 reçoivent input_size dynamique)
    ModelClass, arch_desc = MODELS[run_id]
    model = (ModelClass(input_size=n_feat).to(device)
             if run_id in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) else ModelClass().to(device))
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Run #%d — %s  (%d params)", run_id, arch_desc, n_params)

    hp = RUN_HPARAMS.get(run_id, {"lr": LR, "patience_es": PATIENCE_ES,
                                   "patience_lr": PATIENCE_LR,
                                   "weight_decay": 0, "scheduler": "plateau"})
    run_lr          = hp["lr"]
    run_patience_es = hp["patience_es"]
    run_patience_lr = hp["patience_lr"]
    run_wd          = hp.get("weight_decay", 0)
    run_sched       = hp.get("scheduler", "plateau")
    run_loss        = hp.get("loss", "mse")
    log.info("Hyperparams — lr=%.0e  patience_es=%d  weight_decay=%.0e  scheduler=%s  loss=%s",
             run_lr, run_patience_es, run_wd, run_sched, run_loss)

    criterion = nn.HuberLoss(delta=2.0) if run_loss == "huber" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=run_lr, weight_decay=run_wd)
    if run_sched == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=MIN_LR
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR,
            patience=run_patience_lr, min_lr=MIN_LR
        )

    log_path   = _EXP / "training_logs" / f"training_log_v{run_id}.csv"
    model_path = (_ROOT / "model" / f"best_lstm_v{run_id}.pt" if run_id == 7 else _EXP / "checkpoints" / f"best_lstm_v{run_id}.pt")
    csv_fields = ["epoch", "train_loss", "val_loss", "train_mae", "val_mae", "lr"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_val_loss    = float("inf")
    best_state_dict  = None
    patience_counter = 0

    log.info("=== Debut entrainement — max %d epochs ===", MAX_EPOCHS)
    for epoch in range(1, MAX_EPOCHS + 1):

        # Train
        model.train()
        tr_loss_sum = tr_mae_sum = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            y_hat = model(X_b)
            loss  = criterion(y_hat, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss_sum += loss.item() * len(y_b)
            tr_mae_sum  += (y_hat - y_b).abs().sum().item()

        train_loss = tr_loss_sum / len(y_tr)
        train_mae  = tr_mae_sum  / len(y_tr)

        # Val
        model.eval()
        val_loss_sum = val_mae_sum = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                y_hat = model(X_b)
                val_loss_sum += criterion(y_hat, y_b).item() * len(y_b)
                val_mae_sum  += (y_hat - y_b).abs().sum().item()

        val_loss = val_loss_sum / len(y_val)
        val_mae  = val_mae_sum  / len(y_val)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if run_sched == "cosine":
            scheduler.step(epoch)
        else:
            scheduler.step(val_loss)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "train_mae":  round(train_mae,  4),
                "val_mae":    round(val_mae,    4),
                "lr":         cur_lr,
            })

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  train=%.4f/%.3f  val=%.4f/%.3f  lr=%.2e",
                     epoch, MAX_EPOCHS,
                     train_loss, train_mae, val_loss, val_mae, cur_lr)

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss   = val_loss
            best_state_dict = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= run_patience_es:
            log.info("Early stopping epoch %d (patience=%d)", epoch, PATIENCE_ES)
            break

    # Restore best
    model.load_state_dict(best_state_dict)
    torch.save({"model_state":   best_state_dict,
                "run_id":        run_id,
                "arch":          arch_desc,
                "n_params":      n_params,
                "best_val_loss": best_val_loss},
               model_path)
    log.info("Meilleur modele sauvegarde -> %s  (val_loss=%.4f)", model_path, best_val_loss)

    # Evaluation test set
    model.eval()
    with torch.no_grad():
        y_pred_np = model(torch.from_numpy(X_test_np).to(device)).cpu().numpy()

    mae_  = mean_absolute_error(y_test_np, y_pred_np)
    rmse_ = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    r2_   = r2_score(y_test_np, y_pred_np)
    bias_ = float((y_pred_np - y_test_np).mean())
    pred_range = (float(y_pred_np.min()), float(y_pred_np.max()))
    true_range = (float(y_test_np.min()), float(y_test_np.max()))

    # Log expériences
    exp_path    = _EXP / "experiments_log.csv"
    exp_fields  = ["run_id","architecture","batch_size","lr","epochs_run",
                   "best_val_loss","MAE_test","RMSE_test","R2_test",
                   "delta_MAE","delta_R2","bias","pred_range","notes"]
    write_header = not exp_path.exists()
    log_df = pd.read_csv(log_path, encoding="utf-8")
    note = ("OBJECTIF CIBLE ATTEINT" if r2_ > 0.85
            else "Objectif minimum atteint" if r2_ > 0.75
            else "PIRE BASELINE - STOP" if r2_ < BASELINE["R2"]
            else f"R2<0.75 - iterer run#{run_id+1}")

    with open(exp_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=exp_fields)
        if write_header:
            w.writeheader()
        w.writerow({
            "run_id":        run_id,
            "architecture":  arch_desc,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "epochs_run":    int(log_df["epoch"].max()),
            "best_val_loss": round(best_val_loss, 6),
            "MAE_test":      round(mae_,  4),
            "RMSE_test":     round(rmse_, 4),
            "R2_test":       round(r2_,   4),
            "delta_MAE":     round(mae_  - BASELINE["MAE"],  4),
            "delta_R2":      round(r2_   - BASELINE["R2"],   4),
            "bias":          round(bias_, 4),
            "pred_range":    f"{pred_range[0]:.2f}-{pred_range[1]:.2f}",
            "notes":         note,
        })

    # Rapport
    sep = "=" * 62
    print(); print(sep)
    print(f"  Run #{run_id} -- {arch_desc}")
    print(sep)
    print(f"  {'Metrique':<12} {'Baseline':>10} {'LSTM v'+str(run_id):>10} {'Delta':>10}")
    print("-" * 62)
    print(f"  {'MAE %':<12} {BASELINE['MAE']:>10.3f} {mae_:>10.3f} {mae_-BASELINE['MAE']:>+10.3f}")
    print(f"  {'RMSE %':<12} {BASELINE['RMSE']:>10.3f} {rmse_:>10.3f} {rmse_-BASELINE['RMSE']:>+10.3f}")
    print(f"  {'R2':<12} {BASELINE['R2']:>10.3f} {r2_:>10.3f} {r2_-BASELINE['R2']:>+10.3f}")
    print("-" * 62)
    print(f"  Biais residus  : {bias_:+.4f} %  (ideal=0)")
    print(f"  Range y_pred   : [{pred_range[0]:.2f}, {pred_range[1]:.2f}]")
    print(f"  Range y_test   : [{true_range[0]:.2f}, {true_range[1]:.2f}]")
    print("-" * 62)
    if r2_ < BASELINE["R2"]:
        print("  [STOP] PIRE QUE BASELINE -- diagnostic requis")
    elif r2_ > 0.85:
        print("  [OK] OBJECTIF CIBLE ATTEINT (R2 > 0.85)")
    elif r2_ > 0.75:
        print("  [OK] Objectif minimum atteint -- analyser residus")
    else:
        print(f"  [ITER] R2 < 0.75 -- passer au Run #{run_id+1}")
    print(sep); print()

    np.save(_DATA.parent / "predictions" / f"y_pred_v{run_id}.npy", y_pred_np)
    return y_test_np, y_pred_np, mae_, rmse_, r2_, bias_, run_id


# ---------------------------------------------------------------------------
# Analyse des résidus
# ---------------------------------------------------------------------------
def residual_analysis(y_true, y_pred, run_id):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    meta      = json.load(open(META_JSON, encoding="utf-8"))
    test_bats = meta["test_batteries"]

    # Reconstruction mapping fenetre -> batterie (inter-cycle)
    win_size = RUN_WINDOW.get(run_id, WINDOW_SIZE)
    df = pd.read_csv(_ROOT / "data" / "raw" / "battery_health_dataset.csv")
    df["SoH"] = df["SoH"].clip(upper=100.0)
    bat_tags, cycle_tags = [], []
    for bat in test_bats:
        n_cyc = df[df["battery_id"] == bat]["cycle_number"].nunique()
        n_win = max(0, n_cyc - win_size + 1)
        for i in range(n_win):
            bat_tags.append(bat)
            cycle_tags.append(i + win_size)   # dernier cycle de la fenetre

    assert len(bat_tags) == len(y_true), \
        f"Mismatch mapping : {len(bat_tags)} vs {len(y_true)}"

    residuals = y_pred - y_true

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue")
    lims = [min(y_true.min(), y_pred.min())-1, max(y_true.max(), y_pred.max())+1]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("SoH reel (%)"); ax.set_ylabel("SoH predit (%)")
    ax.set_title(f"Run #{run_id} — Pred vs True\n"
                 f"MAE={mean_absolute_error(y_true,y_pred):.3f}  "
                 f"R2={r2_score(y_true,y_pred):.3f}")

    ax = axes[0, 1]
    sns.histplot(residuals, bins=40, kde=True, ax=ax, color="coral")
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(residuals.mean(), color="red", linestyle="--",
               label=f"biais={residuals.mean():.3f}")
    ax.set_xlabel("Residu (predit - reel)")
    ax.set_title(f"Residus — biais={residuals.mean():.3f}  std={residuals.std():.3f}")
    ax.legend()

    ax = axes[0, 2]
    bat_mae = {b: np.abs(residuals[np.array(bat_tags)==b]).mean()
               for b in test_bats}
    bat_mae_s = pd.Series(bat_mae).sort_values()
    ax.barh(bat_mae_s.index, bat_mae_s.values,
            color=sns.color_palette("tab10", len(bat_mae_s)))
    ax.axvline(np.abs(residuals).mean(), color="red", linestyle="--",
               label=f"MAE={np.abs(residuals).mean():.2f}")
    ax.set_xlabel("MAE (%)"); ax.legend(); ax.set_title("MAE par batterie test")

    ax = axes[1, 0]
    ax.scatter(cycle_tags, residuals, alpha=0.3, s=10, color="purple")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Dernier cycle de la fenetre")
    ax.set_ylabel("Residu"); ax.set_title("Residus vs position cycle")

    ax = axes[1, 1]
    soh_true_bat = {b: y_true[np.array(bat_tags)==b].mean() for b in test_bats}
    mae_bat_vals = [bat_mae[b] for b in test_bats]
    soh_bat_vals = [soh_true_bat[b] for b in test_bats]
    ax.scatter(soh_bat_vals, mae_bat_vals, s=80,
               color=sns.color_palette("tab10", len(test_bats)))
    for i, b in enumerate(test_bats):
        ax.annotate(b, (soh_bat_vals[i], mae_bat_vals[i]),
                    textcoords="offset points", xytext=(4,3), fontsize=8)
    ax.set_xlabel("SoH moyen batterie (%)"); ax.set_ylabel("MAE (%)")
    ax.set_title("Erreur vs SoH moyen — biais par batterie ?")

    ax = axes[1, 2]
    sns.histplot(y_true, bins=30, kde=True, ax=ax, color="steelblue",
                 alpha=0.6, label="y_test")
    sns.histplot(y_pred, bins=30, kde=True, ax=ax, color="tomato",
                 alpha=0.6, label="y_pred")
    ax.set_xlabel("SoH (%)"); ax.legend()
    ax.set_title(f"Distribution\ny_pred [{y_pred.min():.1f}, {y_pred.max():.1f}] "
                 f"vs y_test [{y_true.min():.1f}, {y_true.max():.1f}]")

    plt.suptitle(f"Analyse residus — Run #{run_id} [inter-cycle]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = _ROOT / "reports" / "figures" / "03_training" / f"fig_residuals_v{run_id}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Figure residus -> %s", path)


# ---------------------------------------------------------------------------
# Courbes d'entraînement
# ---------------------------------------------------------------------------
def plot_training_curves(run_id):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    log_df = pd.read_csv(_EXP / "training_logs" / f"training_log_v{run_id}.csv", encoding="utf-8")
    best_epoch = int(log_df.loc[log_df["val_loss"].idxmin(), "epoch"])
    gap_final  = float(log_df["val_loss"].iloc[-1] - log_df["train_loss"].iloc[-1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train loss (MSE)")
    ax.plot(log_df["epoch"], log_df["val_loss"],   label="Val loss (MSE)")
    ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.2,
               label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend()
    ax.set_title(f"Loss — Run #{run_id}")

    ax = axes[1]
    ax.plot(log_df["epoch"], log_df["train_mae"], label="Train MAE")
    ax.plot(log_df["epoch"], log_df["val_mae"],   label="Val MAE")
    ax.axhline(BASELINE["MAE"], color="red", linestyle="--",
               linewidth=1, label=f"Baseline {BASELINE['MAE']}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (%)"); ax.legend()
    ax.set_title(f"MAE — Run #{run_id}")

    ax = axes[2]
    ax.plot(log_df["epoch"], log_df["lr"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_yscale("log")
    ax.set_title("LR schedule")

    overfit = "OVERFITTING" if gap_final > 5 else "OK"
    plt.suptitle(f"Run #{run_id} — gap val-train final: {gap_final:.2f} MSE ({overfit})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = _ROOT / "reports" / "figures" / "03_training" / f"fig_training_v{run_id}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Training curves -> %s", path)

    best_val = log_df["val_loss"].min()
    log.info("Best val_loss=%.4f @ epoch %d | final gap=%.2f | overfitting=%s",
             best_val, best_epoch, gap_final, overfit)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def run_ensemble():
    """
    Calcule l'ensemble pondéré de tous les runs disponibles.
    Stratégie : inverse-MAE weighting (meilleur run = poids plus fort).
    Affiche le résultat vs chaque run individuel et sauvegarde y_pred_ensemble.npy
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pred_dir = _DATA.parent / "predictions"
    y_test   = np.load(_DATA / "y_test_v4b.npy")

    # Candidate runs — filtrés automatiquement sur R² > seuil
    candidate_runs = [7, 10, 11, 12, 13]
    R2_THRESHOLD   = 0.75   # exclut les runs trop faibles

    preds, weights, kept_runs = [], [], []

    log.info("=== Ensemble — chargement des prédictions (seuil R²>%.2f) ===", R2_THRESHOLD)
    for rid in candidate_runs:
        p = pred_dir / f"y_pred_v{rid}.npy"
        if not p.exists():
            log.warning("Run #%d — y_pred absent, ignoré", rid)
            continue
        y_pred = np.load(p)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        if r2 < R2_THRESHOLD:
            log.warning("  Run #%d  R²=%.4f < seuil %.2f — exclu", rid, r2, R2_THRESHOLD)
            continue
        preds.append(y_pred)
        weights.append(1.0 / mae)
        kept_runs.append(rid)
        log.info("  Run #%d  MAE=%.4f  R²=%.4f  poids_brut=%.4f", rid, mae, r2, 1.0 / mae)

    if len(preds) < 2:
        log.error("Pas assez de runs au-dessus du seuil R²>%.2f (min 2).", R2_THRESHOLD)
        return

    weights = np.array(weights) / sum(weights)
    log.info("Poids normalisés : %s", dict(zip(kept_runs, weights.round(3))))

    y_ensemble = sum(w * p for w, p in zip(weights, preds))

    mae_e  = mean_absolute_error(y_test, y_ensemble)
    rmse_e = np.sqrt(mean_squared_error(y_test, y_ensemble))
    r2_e   = r2_score(y_test, y_ensemble)

    # Analyse résidus par batterie test
    meta     = json.load(open(META_JSON, encoding="utf-8"))
    test_bats = meta["test_batteries"]
    df_raw   = pd.read_csv(_ROOT / "data" / "raw" / "battery_health_dataset.csv")
    bat_tags = []
    for bat in test_bats:
        n_cyc = df_raw[df_raw["battery_id"] == bat]["cycle_number"].nunique()
        n_win = max(0, n_cyc - 10 + 1)
        bat_tags.extend([bat] * n_win)
    bat_tags = np.array(bat_tags)

    sep = "=" * 60
    print(); print(sep)
    print("  ENSEMBLE — Résultats Test Set")
    print(sep)
    print(f"  {'Run':<8} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 60)
    for rid, pred in zip(kept_runs, preds):
        m = mean_absolute_error(y_test, pred)
        r = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        print(f"  Run #{rid:<4} {m:>8.4f} {r:>8.4f} {r2:>8.4f}")
    print("-" * 60)
    print(f"  {'Ensemble':<8} {mae_e:>8.4f} {rmse_e:>8.4f} {r2_e:>8.4f}")
    if r2_e > 0.85:
        print("  [OK] OBJECTIF ATTEINT — R2 > 0.85")
    else:
        print(f"  [INFO] R2={r2_e:.4f} — gap restant: {0.85 - r2_e:.4f}")
    print(sep)

    # Résidus par batterie (ensemble)
    residuals = y_ensemble - y_test
    print("\n  MAE par batterie test (Ensemble) :")
    print(f"  {'Batterie':<10} {'N win':>6} {'MAE':>8} {'Biais':>8} {'Max err':>8}")
    print("  " + "-" * 46)
    for bat in test_bats:
        mask = bat_tags == bat
        if mask.sum() == 0:
            continue
        bat_mae  = np.abs(residuals[mask]).mean()
        bat_bias = residuals[mask].mean()
        bat_max  = np.abs(residuals[mask]).max()
        flag = "  <-- problème" if bat_mae > mae_e * 1.5 else ""
        print(f"  {bat:<10} {mask.sum():>6} {bat_mae:>8.3f} {bat_bias:>+8.3f} {bat_max:>8.3f}{flag}")
    print()

    out = pred_dir / "y_pred_ensemble.npy"
    np.save(out, y_ensemble)
    log.info("y_pred_ensemble.npy sauvegardé -> %s", out)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=3,
                        choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    parser.add_argument("--ensemble", action="store_true",
                        help="Calcule l'ensemble des meilleurs runs")
    args = parser.parse_args()

    if args.ensemble:
        run_ensemble()
        return

    log.info("=== Phase 3 Run #%d === [%s]",
             args.run, datetime.now().strftime("%Y-%m-%d %H:%M"))

    y_true, y_pred, mae, rmse, r2, bias, run_id = train(args.run)
    plot_training_curves(run_id)

    if r2 < BASELINE["R2"]:
        log.error("R2=%.3f < baseline=%.3f -- STOP", r2, BASELINE["R2"])
        return

    if r2 > 0.75:
        residual_analysis(y_true, y_pred, run_id)

    log.info("=== Run #%d DONE ===", run_id)


if __name__ == "__main__":
    main()
