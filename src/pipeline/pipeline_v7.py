"""
pipeline_v7.py — window_size=10 cycles, 12 features
Battery SoH LSTM Project
Auteur : Younes Hachami

Seul changement vs v4b : ajout de la 12e feature
  soh_prev = SoH du cycle précédent (décalage de 1)

Justification :
  - Run #7 prédit SoH max ~91% alors que le test set monte à 99%
  - Le modèle ne sait pas où en est la batterie dans sa dégradation —
    il doit l'inférer depuis les features électriques, ce qui échoue
    pour des batteries aux profils de dégradation différents
  - soh_prev donne accès DIRECT à la trajectoire SoH :
      * timestep i (i < WINDOW_SIZE-1) : feature = SoH[i]      (cycle courant)
      * timestep WINDOW_SIZE-1          : feature = SoH[i-1]    (cycle précédent)
    → pas de data leakage (on ne donne jamais SoH[target])
  - Le LSTM a maintenant accès à toute la trajectoire SoH sur la fenêtre
    et peut prédire le prochain point directement depuis la tendance observée

Usage : python pipeline_v7.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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

# Feature 12 : soh_prev — construit dans build_windows, pas dans aggregate_cycles
FEATURE_NAMES = [
    "mean_V","std_V","min_V","mean_T","std_T","mean_I","slope_SoC",
    "voltage_drop","capacity_proxy","temp_rise","voltage_end",
    "soh_prev",   # ← nouvelle feature (12e) : SoH du cycle précédent (décalé de 1)
]
N_FEATURES = len(FEATURE_NAMES)   # 12

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agrégation par cycle — identique v4b, on ajoute soh_value pour build_windows
# ---------------------------------------------------------------------------
ELEC_FEATURES = [f for f in FEATURE_NAMES if f != "soh_prev"]   # 11 features électriques

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
# Fenêtres — construction de soh_prev par décalage
# ---------------------------------------------------------------------------
def build_windows(cycle_df: pd.DataFrame,
                  batteries: list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Pour chaque fenêtre [start, ..., start+WINDOW_SIZE-1] :
      - features électriques (11) : valeurs au cycle t
      - soh_prev (1) :
          * timesteps 0..WINDOW_SIZE-2 : SoH[t]       (SoH courant du cycle t)
          * timestep  WINDOW_SIZE-1    : SoH[t-1]      (SoH du cycle précédent — pas leakage)
    Le modèle voit toute la trajectoire SoH sur la fenêtre sauf le dernier point.
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

        elec  = bat_cyc[ELEC_FEATURES].values   # (n, 11)
        soh   = bat_cyc["SoH"].values            # (n,)

        for start in range(n_win):
            end   = start + WINDOW_SIZE           # exclusive
            elec_w = elec[start:end]              # (10, 11)
            soh_w  = soh[start:end]               # (10,) — SoH pour chaque cycle de la fenêtre

            # soh_prev : décalé de 1, dernier timestep = avant-dernier SoH
            soh_prev_w         = soh_w.copy()               # (10,) : SoH[t] pour t in window
            soh_prev_w[-1]     = soh_w[-2]                  # dernier timestep : SoH[t-1], pas leakage

            window = np.concatenate(
                [elec_w, soh_prev_w.reshape(-1, 1)], axis=1
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
    log.info("=== Pipeline v7 — window_size=%d, %d features — START ===",
             WINDOW_SIZE, N_FEATURES)

    # Chargement & corrections
    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    # Agrégation cycles
    cycle_df = aggregate_cycles(df)
    log.info("Cycles agreges : %d", len(cycle_df))

    # Corrélations
    log.info("Correlations Pearson features vs SoH :")
    for feat in ELEC_FEATURES + ["SoH"]:
        r = float(cycle_df[[feat, "SoH"]].corr().iloc[0, 1])
        mk = "***" if abs(r) > 0.50 else "**" if abs(r) > 0.30 else ""
        log.info("  %-32s : r=%+.4f  %s", feat, r, mk)

    # Fenêtres
    log.info("Train batteries (%d) :", len(TRAIN_BATTERIES))
    X_train_raw, y_train, bat_tr = build_windows(cycle_df, TRAIN_BATTERIES)
    log.info("Test batteries (%d) :", len(TEST_BATTERIES))
    X_test_raw,  y_test,  bat_te = build_windows(cycle_df, TEST_BATTERIES)

    # Validations
    assert len(set(bat_tr) & set(bat_te)) == 0, "Overlap batteries!"
    assert X_train_raw.shape == (944, WINDOW_SIZE, N_FEATURES), \
        f"Shape train inattendue : {X_train_raw.shape}"
    assert X_test_raw.shape  == (299, WINDOW_SIZE, N_FEATURES), \
        f"Shape test inattendue  : {X_test_raw.shape}"
    assert not np.isnan(X_train_raw).any()
    assert not np.isnan(X_test_raw).any()
    # Vérification no-leakage : soh_prev au dernier timestep != target
    last_soh_prev = X_train_raw[:, -1, -1]   # soh_prev au dernier timestep
    assert not np.allclose(last_soh_prev, y_train), "LEAKAGE détecté : soh_prev[-1] == y_train"
    log.info("OK  Shapes, overlap, NaN, no-leakage soh_prev")

    # Normalisation (fit sur train uniquement)
    N_tr, W, F = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)
    X_test = scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(len(X_test_raw), W, F).astype(np.float32)
    log.info("Scaler mean (12) : %s", np.round(scaler.mean_, 3))
    log.info("Scaler mean soh_prev (feature 12) = %.3f  std = %.3f",
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
    log.info("Baseline Ridge v7 : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             ridge_mae, ridge_rmse, ridge_r2)

    # Vérification sanity : Ridge doit battre v4b (R2=0.671) si soh_prev est utile
    if ridge_r2 > 0.75:
        log.info("OK  soh_prev utile — baseline Ridge deja > 0.75")
    elif ridge_r2 > 0.671:
        log.info("OK  soh_prev ameliore la baseline (%.3f > 0.671)", ridge_r2)
    else:
        log.warning("soh_prev n'ameliore pas la baseline Ridge — verifier la construction")

    # Export
    np.save(OUTPUT_DIR / "X_train_v7.npy", X_train)
    np.save(OUTPUT_DIR / "X_test_v7.npy",  X_test)
    np.save(OUTPUT_DIR / "y_train_v7.npy", y_train)
    np.save(OUTPUT_DIR / "y_test_v7.npy",  y_test)
    log.info("Exports : X_train_v7%s  X_test_v7%s", X_train.shape, X_test.shape)

    metadata = {
        "created_at":      datetime.now().isoformat(timespec="seconds"),
        "approach":        "inter_cycle",
        "window_size":     WINDOW_SIZE,
        "n_features":      N_FEATURES,
        "feature_names":   FEATURE_NAMES,
        "new_feature":     "soh_prev : SoH[t] pour t<last, SoH[t-1] pour t=last (no leakage)",
        "train_batteries": TRAIN_BATTERIES,
        "test_batteries":  TEST_BATTERIES,
        "n_train_samples": int(len(y_train)),
        "n_test_samples":  int(len(y_test)),
        "scaler_mean":     np.round(scaler.mean_,  6).tolist(),
        "scaler_std":      np.round(scaler.scale_, 6).tolist(),
        "baseline_ridge":  {"MAE": round(ridge_mae,4),
                            "RMSE": round(ridge_rmse,4),
                            "R2": round(ridge_r2,4)},
        "feature_correlations_with_SoH": {
            feat: round(float(cycle_df[[feat,"SoH"]].corr().iloc[0,1]), 4)
            for feat in ELEC_FEATURES
        },
    }
    (OUTPUT_DIR / "metadata_v7.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("metadata_v7.json ecrit")

    # Résumé
    print()
    print("=" * 60)
    print("  PIPELINE v7 — RÉSULTATS")
    print("=" * 60)
    print(f"  X_train_v7 : {X_train.shape}  float32")
    print(f"  X_test_v7  : {X_test.shape}   float32")
    print(f"  window_size : {WINDOW_SIZE} cycles")
    print(f"  n_features  : {N_FEATURES} (11 v4b + soh_prev)")
    print("-" * 60)
    print(f"  Baseline Ridge v7  : MAE={ridge_mae:.3f}  R2={ridge_r2:.3f}")
    print(f"  Baseline Ridge v4b : MAE=3.537        R2=0.671  (ref)")
    if ridge_r2 > 0.85:
        print("  [!!] Ridge seul > 0.85 — soh_prev très fort")
    print("=" * 60)
    log.info("=== Pipeline v7 — DONE ===")


if __name__ == "__main__":
    main()
