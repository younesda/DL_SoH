"""
pipeline_v8.py — window_size=10 cycles, 12 features
Battery SoH LSTM Project
Auteur : Younes Hachami

Différence vs v7 : remplacement de soh_prev (valeur absolue) par soh_delta
  (variation ΔSoH entre cycles consécutifs).

Motivation :
  - v7 fournissait les valeurs absolues SoH[t] pour t in [0..8], ce qui biaisait
    le modèle vers les niveaux SoH des batteries d'entraînement (max ~93%).
  - Les batteries test peuvent avoir SoH jusqu'à 98.67% → le modèle ne sait pas
    prédire au-delà de ce qu'il a vu en absolu.
  - soh_delta = ΔSoH entre cycles consécutifs encode les TENDANCES de dégradation,
    pas les niveaux absolus → généralise aux batteries à SoH différent.

Construction soh_delta (sans data leakage) :
  timestep i=0         : delta = 0 (pas de cycle précédent dans la fenêtre)
  timestep i=1..W-2    : delta = SoH[start+i] - SoH[start+i-1]
  timestep i=W-1 (last): delta = SoH[start+i-1] - SoH[start+i-2]  ← pas leakage

Usage : python pipeline_v8.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pathlib import Path as _Path
_ROOT          = _Path(__file__).resolve().parents[2]
DATA_PATH      = str(_ROOT / "data" / "raw" / "battery_health_dataset.csv")
OUTPUT_DIR     = _ROOT / "data" / "processed" / "final"
FIGS_DIR       = _ROOT / "reports" / "figures" / "02_pipeline"
BINS_PER_CYCLE = 20
WINDOW_SIZE    = 10
SOH_CLIP_MAX   = 100.0
TEMP_CLIP_MAX  = 60.0

TRAIN_BATTERIES = [
    "B0005","B0007","B0025","B0026","B0027","B0029","B0030","B0031",
    "B0032","B0033","B0036","B0038","B0040","B0042","B0043","B0044",
    "B0046","B0047","B0048"
]
TEST_BATTERIES = ["B0006","B0018","B0028","B0034","B0039"]

FEATURE_NAMES = [
    "mean_V","std_V","min_V","mean_T","std_T","mean_I","slope_SoC",
    "voltage_drop","capacity_proxy","temp_rise","voltage_end",
    "soh_delta",   # ← nouvelle feature (12e) : ΔSoH entre cycles consécutifs
]
N_FEATURES = len(FEATURE_NAMES)   # 12

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agrégation par cycle — identique v4b/v7
# ---------------------------------------------------------------------------
ELEC_FEATURES = [f for f in FEATURE_NAMES if f != "soh_delta"]   # 11 features électriques

def aggregate_cycles(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = df.sort_values(
        ["battery_id", "cycle_number", "SoC"], ascending=[True, True, False]
    ).groupby(["battery_id", "cycle_number"], sort=False)

    for (bat, cyc), grp in groups:
        if len(grp) != BINS_PER_CYCLE:
            continue
        volt = grp["Voltage_measured"].values
        temp = grp["Temperature_measured"].values
        curr = grp["Current_measured"].values
        soc  = grp["SoC"].values

        mean_i       = float(curr.mean())
        voltage_drop = float(volt.max() - volt.min())

        records.append({
            "battery_id":     bat,
            "cycle_number":   int(cyc),
            "mean_V":         float(volt.mean()),
            "std_V":          float(volt.std()),
            "min_V":          float(volt.min()),
            "mean_T":         float(temp.mean()),
            "std_T":          float(temp.std()),
            "mean_I":         mean_i,
            "slope_SoC":      float(np.polyfit(range(BINS_PER_CYCLE), soc, 1)[0]),
            "voltage_drop":   voltage_drop,
            "capacity_proxy": float(abs(mean_i) * BINS_PER_CYCLE),
            "temp_rise":      float(temp.max() - temp.min()),
            "voltage_end":    float(volt[-1]),
            "SoH":            float(grp["SoH"].iloc[0]),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fenêtres — construction de soh_delta par différence consécutive
# ---------------------------------------------------------------------------
def build_windows(cycle_df: pd.DataFrame,
                  batteries: list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Pour chaque fenêtre [start, ..., start+WINDOW_SIZE-1] :
      - features électriques (11) : valeurs au cycle t
      - soh_delta (1) :
          * timestep i=0         : 0
          * timestep i=1..W-2    : SoH[start+i] - SoH[start+i-1]
          * timestep i=W-1 (last): SoH[start+i-1] - SoH[start+i-2]  (no leakage)
    """
    X_list, y_list, bat_tags = [], [], []
    subset = cycle_df[cycle_df["battery_id"].isin(batteries)]

    for bat in batteries:
        bat_cyc = (subset[subset["battery_id"] == bat]
                   .sort_values("cycle_number")
                   .reset_index(drop=True))
        n = len(bat_cyc)
        n_win = n - WINDOW_SIZE + 1

        if n < WINDOW_SIZE:
            log.warning("  %s : %d cycles < window_size=%d — ignoree", bat, n, WINDOW_SIZE)
            continue

        elec = bat_cyc[ELEC_FEATURES].values   # (n, 11)
        soh  = bat_cyc["SoH"].values            # (n,)

        for start in range(n_win):
            end    = start + WINDOW_SIZE           # exclusive
            elec_w = elec[start:end]              # (10, 11)
            soh_w  = soh[start:end]               # (10,)

            # soh_delta : ΔSoH entre cycles consécutifs (sans leakage au dernier timestep)
            soh_delta_w = np.zeros(WINDOW_SIZE, dtype=np.float32)
            for i in range(1, WINDOW_SIZE - 1):
                soh_delta_w[i] = soh_w[i] - soh_w[i - 1]
            # Dernier timestep : utilise l'avant-dernier delta (no leakage sur target)
            soh_delta_w[-1] = soh_w[-2] - soh_w[-3]

            window = np.concatenate(
                [elec_w, soh_delta_w.reshape(-1, 1)], axis=1
            )  # (10, 12)

            X_list.append(window)
            y_list.append(soh_w[-1])    # target = SoH au dernier cycle
            bat_tags.append(bat)

        log.info("  %s : %d cycles -> %d fenetres", bat, n, max(0, n_win))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, bat_tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Pipeline v8 — window_size=%d, %d features (soh_delta) — START ===",
             WINDOW_SIZE, N_FEATURES)

    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    cycle_df = aggregate_cycles(df)
    log.info("Cycles agreges : %d", len(cycle_df))

    log.info("Correlations Pearson features vs SoH :")
    for feat in ELEC_FEATURES + ["SoH"]:
        r = float(cycle_df[[feat, "SoH"]].corr().iloc[0, 1])
        mk = "***" if abs(r) > 0.50 else "**" if abs(r) > 0.30 else ""
        log.info("  %-32s : r=%+.4f  %s", feat, r, mk)

    log.info("Train batteries (%d) :", len(TRAIN_BATTERIES))
    X_train_raw, y_train, bat_tr = build_windows(cycle_df, TRAIN_BATTERIES)
    log.info("Test batteries (%d) :", len(TEST_BATTERIES))
    X_test_raw,  y_test,  bat_te = build_windows(cycle_df, TEST_BATTERIES)

    assert len(set(bat_tr) & set(bat_te)) == 0, "Overlap batteries!"
    assert X_train_raw.shape == (944, WINDOW_SIZE, N_FEATURES), \
        f"Shape train inattendue : {X_train_raw.shape}"
    assert X_test_raw.shape  == (299, WINDOW_SIZE, N_FEATURES), \
        f"Shape test inattendue  : {X_test_raw.shape}"
    assert not np.isnan(X_train_raw).any()
    assert not np.isnan(X_test_raw).any()
    # Vérification no-leakage : soh_delta au dernier timestep != cible directe
    last_delta = X_train_raw[:, -1, -1]
    assert not np.allclose(last_delta, y_train), "LEAKAGE potentiel : soh_delta[-1] == y_train"
    log.info("OK  Shapes, overlap, NaN, no-leakage soh_delta")

    # Normalisation (fit sur train uniquement)
    N_tr, W, F = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)
    X_test = scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(len(X_test_raw), W, F).astype(np.float32)
    log.info("Scaler mean soh_delta (feature 12) = %.4f  std = %.4f",
             scaler.mean_[-1], scaler.scale_[-1])

    # Baseline Ridge
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    yp = ridge.predict(X_test.reshape(len(X_test), -1))
    ridge_r2   = r2_score(y_test, yp)
    ridge_mae  = mean_absolute_error(y_test, yp)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, yp))
    log.info("Baseline Ridge v8 : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             ridge_mae, ridge_rmse, ridge_r2)

    # Export
    np.save(OUTPUT_DIR / "X_train_v8.npy", X_train)
    np.save(OUTPUT_DIR / "X_test_v8.npy",  X_test)
    np.save(OUTPUT_DIR / "y_train_v8.npy", y_train)
    np.save(OUTPUT_DIR / "y_test_v8.npy",  y_test)
    log.info("Exports : X_train_v8%s  X_test_v8%s", X_train.shape, X_test.shape)

    metadata = {
        "created_at":      datetime.now().isoformat(timespec="seconds"),
        "approach":        "inter_cycle",
        "window_size":     WINDOW_SIZE,
        "n_features":      N_FEATURES,
        "feature_names":   FEATURE_NAMES,
        "new_feature":     "soh_delta : ΔSoH entre cycles (no leakage au dernier timestep)",
        "train_batteries": TRAIN_BATTERIES,
        "test_batteries":  TEST_BATTERIES,
        "n_train_samples": int(len(y_train)),
        "n_test_samples":  int(len(y_test)),
        "scaler_mean":     np.round(scaler.mean_,  6).tolist(),
        "scaler_std":      np.round(scaler.scale_, 6).tolist(),
        "baseline_ridge":  {"MAE": round(ridge_mae, 4),
                            "RMSE": round(ridge_rmse, 4),
                            "R2": round(ridge_r2, 4)},
        "feature_correlations_with_SoH": {
            feat: round(float(cycle_df[[feat, "SoH"]].corr().iloc[0, 1]), 4)
            for feat in ELEC_FEATURES
        },
    }
    (OUTPUT_DIR / "metadata_v8.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("metadata_v8.json ecrit")

    print()
    print("=" * 60)
    print("  PIPELINE v8 — RÉSULTATS")
    print("=" * 60)
    print(f"  X_train_v8 : {X_train.shape}  float32")
    print(f"  X_test_v8  : {X_test.shape}   float32")
    print(f"  window_size : {WINDOW_SIZE} cycles")
    print(f"  n_features  : {N_FEATURES} (11 v4b + soh_delta)")
    print("-" * 60)
    print(f"  Baseline Ridge v8  : MAE={ridge_mae:.3f}  R2={ridge_r2:.3f}")
    print(f"  Baseline Ridge v7  : (ref soh_prev)")
    print("=" * 60)
    log.info("=== Pipeline v8 — DONE ===")


if __name__ == "__main__":
    main()
