"""
pipeline_v9.py — window_size=10 cycles, 13 features
Battery SoH LSTM Project
Auteur : Younes Hachami

Changement vs v7 : ajout de la 13e feature
  soh_lin_extrap = soh_prev[8] + slope_lineaire_sur_soh_prev[0:9]

Justification :
  - Run #13 (v7, 12feat) : R²=0.914  vs  Ridge v7 : R²=0.9484
  - Le LSTM sous-performe Ridge parce que Ridge calcule implicitement
    des combinaisons cross-timesteps de soh_prev (pente de dégradation).
  - Le modèle plafonne à ~93% SoH car les batteries à SoH > 93% sont
    sous-représentées en train : il ne sait pas extrapoler la tendance.
  - soh_lin_extrap donne DIRECTEMENT la projection linéaire du prochain SoH :
      soh_lin_extrap = soh_prev[8] + slope
    où slope est la pente de régression linéaire sur soh_prev[0:9]
  - Avantages :
      * Rend explicite ce que Ridge calcule implicitement
      * Résout le problème de plafonnement : si la batterie est à 95%,
        soh_lin_extrap sera ~95%, même si le modèle n'a jamais vu 95% en train
      * Pas de data leakage : utilise uniquement soh[t] à soh[t+8]
      * Disponible en production (BMS a accès à l'historique SoH)

Usage : python pipeline_v9.py
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

ELEC_FEATURES = [
    "mean_V","std_V","min_V","mean_T","std_T","mean_I","slope_SoC",
    "voltage_drop","capacity_proxy","temp_rise","voltage_end",
]
FEATURE_NAMES = ELEC_FEATURES + ["soh_prev", "soh_lin_extrap"]
N_FEATURES = len(FEATURE_NAMES)   # 13

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agrégation par cycle — identique v7
# ---------------------------------------------------------------------------
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
# Fenêtres — soh_prev (v7) + soh_lin_extrap (nouveau)
# ---------------------------------------------------------------------------
def build_windows(cycle_df: pd.DataFrame,
                  batteries: list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Pour chaque fenêtre [start, ..., start+WINDOW_SIZE-1] :

      Feature 12 — soh_prev (identique v7) :
        * timesteps 0..WINDOW_SIZE-2 : SoH[t]    (SoH courant du cycle t)
        * timestep  WINDOW_SIZE-1    : SoH[t-1]   (SoH précédent — pas leakage)

      Feature 13 — soh_lin_extrap (nouveau) :
        * Pente linéaire sur soh_prev[0:9] (9 valeurs connues, sans leakage)
        * soh_lin_extrap = soh_prev[8] + slope
        * Constante sur les 10 timesteps (contexte global de fenêtre)
        * Interprétation : projection linéaire du prochain SoH
        * Aucun leakage : utilise seulement soh[t] à soh[t+8]
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
            end    = start + WINDOW_SIZE
            elec_w = elec[start:end]             # (10, 11)
            soh_w  = soh[start:end]              # (10,) — SoH des 10 cycles

            # soh_prev : décalé de 1 au dernier timestep (pas de leakage)
            soh_prev_w      = soh_w.copy()
            soh_prev_w[-1]  = soh_w[-2]          # timestep 9 ← SoH[t+8], pas SoH[t+9]

            # soh_lin_extrap : régression linéaire sur les 9 valeurs connues
            # soh_prev_w[0:9] = soh[t] .. soh[t+8]  (9 points sans leakage)
            timesteps  = np.arange(9, dtype=np.float64)
            slope_val  = np.polyfit(timesteps, soh_prev_w[:9].astype(np.float64), 1)[0]
            # Extrapolation 1 pas en avant depuis le dernier point connu
            extrap_val = float(soh_prev_w[8] + slope_val)

            # Constante sur toute la fenêtre (contexte global)
            extrap_col = np.full((WINDOW_SIZE, 1), extrap_val, dtype=np.float32)

            window = np.concatenate(
                [elec_w,
                 soh_prev_w.reshape(-1, 1),
                 extrap_col],
                axis=1
            )  # (10, 13)

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
    log.info("=== Pipeline v9 — window_size=%d, %d features — START ===",
             WINDOW_SIZE, N_FEATURES)

    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    cycle_df = aggregate_cycles(df)
    log.info("Cycles agreges : %d", len(cycle_df))

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
    last_soh_prev = X_train_raw[:, -1, 11]   # feature 12 = soh_prev
    assert not np.allclose(last_soh_prev, y_train), \
        "LEAKAGE détecté : soh_prev[-1] == y_train"

    # Vérification no-leakage : soh_lin_extrap != target (corrélation proche de 1 est OK, égalité = leakage)
    extrap_vals = X_train_raw[:, 0, 12]      # feature 13 = soh_lin_extrap (constante, prendre t=0)
    assert not np.allclose(extrap_vals, y_train), \
        "LEAKAGE détecté : soh_lin_extrap == y_train"
    log.info("OK  Shapes, overlap, NaN, no-leakage (soh_prev + soh_lin_extrap)")

    # Corrélation soh_lin_extrap vs target
    corr_extrap = float(np.corrcoef(extrap_vals, y_train)[0, 1])
    log.info("Corrélation soh_lin_extrap vs SoH_target (train) : r=%.4f", corr_extrap)

    # Normalisation
    N_tr, W, F = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)
    X_test = scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(len(X_test_raw), W, F).astype(np.float32)

    # Baseline Ridge
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    yp = ridge.predict(X_test.reshape(len(X_test), -1))
    ridge_r2   = r2_score(y_test, yp)
    ridge_mae  = mean_absolute_error(y_test, yp)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, yp))
    log.info("Baseline Ridge v9 : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             ridge_mae, ridge_rmse, ridge_r2)

    # Export
    np.save(OUTPUT_DIR / "X_train_v9.npy", X_train)
    np.save(OUTPUT_DIR / "X_test_v9.npy",  X_test)
    np.save(OUTPUT_DIR / "y_train_v9.npy", y_train)
    np.save(OUTPUT_DIR / "y_test_v9.npy",  y_test)
    log.info("Exports : X_train_v9%s  X_test_v9%s", X_train.shape, X_test.shape)

    metadata = {
        "created_at":       datetime.now().isoformat(timespec="seconds"),
        "approach":         "inter_cycle",
        "window_size":      WINDOW_SIZE,
        "n_features":       N_FEATURES,
        "feature_names":    FEATURE_NAMES,
        "new_feature":      "soh_lin_extrap : soh_prev[8] + slope_lineaire(soh_prev[0:9])",
        "train_batteries":  TRAIN_BATTERIES,
        "test_batteries":   TEST_BATTERIES,
        "n_train_samples":  int(len(y_train)),
        "n_test_samples":   int(len(y_test)),
        "scaler_mean":      np.round(scaler.mean_,  6).tolist(),
        "scaler_std":       np.round(scaler.scale_, 6).tolist(),
        "baseline_ridge_v7":{"MAE": 1.063, "RMSE": 1.756, "R2": 0.9484},
        "baseline_ridge_v9":{"MAE": round(ridge_mae,4),
                             "RMSE": round(ridge_rmse,4),
                             "R2":   round(ridge_r2,4)},
        "corr_soh_lin_extrap_vs_target": round(corr_extrap, 4),
    }
    (OUTPUT_DIR / "metadata_v9.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("metadata_v9.json ecrit")

    print()
    print("=" * 60)
    print("  PIPELINE v9 — RÉSULTATS")
    print("=" * 60)
    print(f"  X_train_v9 : {X_train.shape}  float32")
    print(f"  X_test_v9  : {X_test.shape}   float32")
    print(f"  n_features : {N_FEATURES} (11 elec + soh_prev + soh_lin_extrap)")
    print(f"  Corr soh_lin_extrap vs target : r={corr_extrap:.4f}")
    print("-" * 60)
    print(f"  Baseline Ridge v9 : MAE={ridge_mae:.3f}  R2={ridge_r2:.3f}")
    print(f"  Baseline Ridge v7 : MAE=1.063        R2=0.948  (ref)")
    print("=" * 60)
    log.info("=== Pipeline v9 — DONE ===")


if __name__ == "__main__":
    main()
