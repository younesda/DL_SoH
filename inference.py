"""
inference.py — Script d'inférence production
Battery SoH LSTM Project — Run #7
Auteur : Younes Hachami

Usage :
    from inference import SoHPredictor
    predictor = SoHPredictor()
    soh = predictor.predict(battery_cycles)   # array (10, 11)
    sohs = predictor.predict_batch(X)         # array (N, 10, 11)

Ou en ligne de commande :
    python inference.py --input battery_cycles.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture Run #7 (auto-contenu, pas d'import train_lstm)
# ---------------------------------------------------------------------------
class _LSTMv7(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(11, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(128, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


# ---------------------------------------------------------------------------
# Prédicateur principal
# ---------------------------------------------------------------------------
class SoHPredictor:
    """
    Charge best_lstm_v7.pt et le scaler v4b.
    Accepte des features brutes (non normalisées) et retourne SoH en %.

    Paramètres
    ----------
    model_path   : chemin vers best_lstm_v7.pt  (défaut : même dossier)
    meta_path    : chemin vers metadata_v4b.json (défaut : même dossier)
    device       : "cpu" | "cuda" | "auto"

    Exemple
    -------
    predictor = SoHPredictor()

    # Prédire pour 1 fenêtre (10 cycles x 11 features)
    x = np.load("X_test_v4b.npy")[0]    # shape (10, 11) — déjà normalisé
    # Pour des données brutes :
    soh = predictor.predict_raw(x_raw)   # (10, 11) non normalisé
    # Pour des données déjà normalisées (sorti du pipeline) :
    soh = predictor.predict(x)           # (10, 11) normalisé
    """

    FEATURE_NAMES = [
        "mean_V", "std_V", "min_V", "mean_T", "std_T", "mean_I",
        "slope_SoC", "voltage_drop", "capacity_proxy", "temp_rise", "voltage_end",
    ]
    WINDOW_SIZE = 10
    N_FEATURES  = 11

    def __init__(self,
                 model_path: str | Path = None,
                 meta_path:  str | Path = None,
                 device: str = "auto"):

        base = Path(__file__).parent
        model_path = Path(model_path) if model_path else base / "model" / "best_lstm_v7.pt"
        meta_path  = Path(meta_path)  if meta_path  else base / "data" / "processed" / "final" / "metadata_v4b.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata introuvable : {meta_path}")

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Modèle
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self._model = _LSTMv7().to(self.device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        # Scaler (paramètres issus de metadata_v4b.json)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._scaler_mean  = np.array(meta["scaler_mean"],  dtype=np.float32)
        self._scaler_scale = np.array(meta["scaler_std"],   dtype=np.float32)

    # ------------------------------------------------------------------
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Applique StandardScaler (mean/scale issus du train set)."""
        shape = X.shape
        X_2d = X.reshape(-1, self.N_FEATURES)
        X_norm = (X_2d - self._scaler_mean) / self._scaler_scale
        return X_norm.reshape(shape).astype(np.float32)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> float | np.ndarray:
        """
        Prédit SoH à partir de features DÉJÀ NORMALISÉES.

        Paramètres
        ----------
        X : shape (10, 11) pour 1 fenêtre
            shape (N, 10, 11) pour un batch

        Retourne
        --------
        float si X.ndim == 2, np.ndarray(N,) si X.ndim == 3
        """
        single = X.ndim == 2
        if single:
            X = X[np.newaxis]   # (1, 10, 11)

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = self._model(X_t).cpu().numpy()

        return float(out[0]) if single else out

    # ------------------------------------------------------------------
    def predict_raw(self, X_raw: np.ndarray) -> float | np.ndarray:
        """
        Prédit SoH à partir de features BRUTES (non normalisées).
        Applique le StandardScaler v4b avant l'inférence.

        Paramètres
        ----------
        X_raw : shape (10, 11) ou (N, 10, 11)
                Features dans l'ordre FEATURE_NAMES

        Retourne
        --------
        float si X_raw.ndim == 2, np.ndarray(N,) si X_raw.ndim == 3
        """
        X_norm = self._normalize(X_raw)
        return self.predict(X_norm)

    # ------------------------------------------------------------------
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Alias explicite pour batch : prédit SoH pour N fenêtres normalisées."""
        assert X.ndim == 3 and X.shape[1:] == (self.WINDOW_SIZE, self.N_FEATURES), \
            f"Shape attendue (N, {self.WINDOW_SIZE}, {self.N_FEATURES}), reçu {X.shape}"
        return self.predict(X)

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"SoHPredictor(device={self.device}, "
                f"window={self.WINDOW_SIZE}, features={self.N_FEATURES})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="Prédire SoH à partir d'un fichier .npy (N, 10, 11) normalisé"
    )
    parser.add_argument("--input",  required=True, help="Fichier .npy (N,10,11)")
    parser.add_argument("--output", default=None,  help="Fichier de sortie .npy (optionnel)")
    parser.add_argument("--raw",    action="store_true",
                        help="Données brutes (applique normalisation)")
    parser.add_argument("--model",  default=None,  help="Chemin best_lstm_v7.pt")
    parser.add_argument("--meta",   default=None,  help="Chemin metadata_v4b.json")
    args = parser.parse_args()

    predictor = SoHPredictor(
        model_path=args.model,
        meta_path=args.meta,
    )
    print(predictor)

    X = np.load(args.input)
    if X.ndim == 2:
        X = X[np.newaxis]
    print(f"Input shape : {X.shape}")

    if args.raw:
        y_pred = predictor.predict_raw(X)
    else:
        y_pred = predictor.predict_batch(X)

    print(f"SoH prédit : min={y_pred.min():.2f}%  max={y_pred.max():.2f}%  "
          f"mean={y_pred.mean():.2f}%  (N={len(y_pred)})")

    if args.output:
        np.save(args.output, y_pred)
        print(f"Sauvegardé -> {args.output}")


if __name__ == "__main__":
    _cli()
