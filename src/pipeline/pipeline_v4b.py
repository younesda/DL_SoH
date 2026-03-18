"""
pipeline_v4b.py — window_size=10 cycles, 11 features agrégées
Battery SoH LSTM Project
Auteur : Younes Hachami

Seul changement vs pipeline_v3 : WINDOW_SIZE 5 → 10.
Features inchangées : 11 (7 v2 + 4 nouvelles).

Usage : python pipeline_v4b.py
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
WINDOW_SIZE    = 10          # ← seul changement vs v3
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
]
N_FEATURES = len(FEATURE_NAMES)   # 11

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agrégation par cycle (identique pipeline_v3)
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

        records.append({
            "battery_id":     bat,
            "cycle_number":   int(cyc),
            "mean_V":         float(volt.mean()),
            "std_V":          float(volt.std()),
            "min_V":          float(volt.min()),
            "mean_T":         float(temp.mean()),
            "std_T":          float(temp.std()),
            "mean_I":         float(curr.mean()),
            "slope_SoC":      float(np.polyfit(range(BINS_PER_CYCLE), soc, 1)[0]),
            "voltage_drop":   float(volt.max() - volt.min()),
            "capacity_proxy": float(abs(curr.mean()) * BINS_PER_CYCLE),
            "temp_rise":      float(temp.max() - temp.min()),
            "voltage_end":    float(volt[-1]),
            "SoH":            float(grp["SoH"].iloc[0]),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fenêtres de WINDOW_SIZE cycles (stride=1)
# ---------------------------------------------------------------------------
def build_windows(cycle_df: pd.DataFrame,
                  batteries: list) -> tuple[np.ndarray, np.ndarray, list]:
    X_list, y_list, bat_tags = [], [], []
    subset = cycle_df[cycle_df["battery_id"].isin(batteries)]

    for bat in batteries:
        bat_cyc = (subset[subset["battery_id"] == bat]
                   .sort_values("cycle_number")
                   .reset_index(drop=True))
        n = len(bat_cyc)
        n_win = n - WINDOW_SIZE + 1

        if n < WINDOW_SIZE:
            log.warning("  %s : %d cycles < window_size=%d — ignorée", bat, n, WINDOW_SIZE)
            continue

        feats = bat_cyc[FEATURE_NAMES].values
        soh   = bat_cyc["SoH"].values

        for start in range(n_win):
            X_list.append(feats[start : start + WINDOW_SIZE])
            y_list.append(soh[start + WINDOW_SIZE - 1])
            bat_tags.append(bat)

        log.info("  %s : %d cycles -> %d fenêtres", bat, n, max(0, n_win))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, bat_tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Pipeline v4b — window_size=%d, %d features — START ===",
             WINDOW_SIZE, N_FEATURES)

    # Chargement & corrections
    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    # Agrégation cycles
    cycle_df = aggregate_cycles(df)
    log.info("Cycles agrégés : %d", len(cycle_df))

    # Fenêtres
    log.info("Train batteries (%d) :", len(TRAIN_BATTERIES))
    X_train_raw, y_train, bat_tr = build_windows(cycle_df, TRAIN_BATTERIES)
    log.info("Test batteries (%d) :", len(TEST_BATTERIES))
    X_test_raw,  y_test,  bat_te = build_windows(cycle_df, TEST_BATTERIES)

    # Validations
    assert len(set(bat_tr) & set(bat_te)) == 0, "Overlap batteries!"
    assert X_train_raw.shape == (len(y_train), WINDOW_SIZE, N_FEATURES), \
        f"Shape train inattendue : {X_train_raw.shape}"
    assert X_test_raw.shape  == (len(y_test),  WINDOW_SIZE, N_FEATURES), \
        f"Shape test inattendue  : {X_test_raw.shape}"
    assert not np.isnan(X_train_raw).any()
    assert not np.isnan(X_test_raw).any()
    log.info("OK  Shapes  — X_train:%s  X_test:%s", X_train_raw.shape, X_test_raw.shape)
    log.info("OK  Pas d'overlap, pas de NaN")

    # Référence window=5 pour comparaison
    n_train_v3, n_test_v3 = 1039, 324
    log.info("Nb fenêtres window=5  : train=%d  test=%d", n_train_v3, n_test_v3)
    log.info("Nb fenêtres window=10 : train=%d  test=%d  (delta: %+d / %+d)",
             len(y_train), len(y_test),
             len(y_train) - n_train_v3, len(y_test) - n_test_v3)

    # Normalisation (fit sur train uniquement)
    N_tr, W, F = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)
    X_test = scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(len(X_test_raw), W, F).astype(np.float32)
    log.info("Scaler mean (11) : %s", np.round(scaler.mean_, 3))

    # Baseline Ridge
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    yp = ridge.predict(X_test.reshape(len(X_test), -1))
    ridge_r2   = r2_score(y_test, yp)
    ridge_mae  = mean_absolute_error(y_test, yp)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, yp))
    log.info("Baseline Ridge v4b (window=10) : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             ridge_mae, ridge_rmse, ridge_r2)

    # Export (sans écraser les fichiers existants)
    np.save(OUTPUT_DIR / "X_train_v4b.npy", X_train)
    np.save(OUTPUT_DIR / "X_test_v4b.npy",  X_test)
    np.save(OUTPUT_DIR / "y_train_v4b.npy", y_train)
    np.save(OUTPUT_DIR / "y_test_v4b.npy",  y_test)
    log.info("Exports : X_train_v4b%s  X_test_v4b%s", X_train.shape, X_test.shape)

    metadata = {
        "created_at":      datetime.now().isoformat(timespec="seconds"),
        "approach":        "inter_cycle",
        "window_size":     WINDOW_SIZE,
        "n_features":      N_FEATURES,
        "feature_names":   FEATURE_NAMES,
        "train_batteries": TRAIN_BATTERIES,
        "test_batteries":  TEST_BATTERIES,
        "n_train_samples": int(len(y_train)),
        "n_test_samples":  int(len(y_test)),
        "y_train_min":     float(y_train.min()),  "y_train_max": float(y_train.max()),
        "y_train_mean":    float(y_train.mean()), "y_train_std":  float(y_train.std()),
        "y_test_min":      float(y_test.min()),   "y_test_max":  float(y_test.max()),
        "y_test_mean":     float(y_test.mean()),  "y_test_std":   float(y_test.std()),
        "scaler_mean":     np.round(scaler.mean_,  6).tolist(),
        "scaler_std":      np.round(scaler.scale_, 6).tolist(),
        "baseline_ridge":  {"MAE": round(ridge_mae,4),
                            "RMSE": round(ridge_rmse,4),
                            "R2": round(ridge_r2,4)},
    }
    (OUTPUT_DIR / "metadata_v4b.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("metadata_v4b.json écrit")

    # Visualisation distribution y
    import matplotlib.pyplot as plt, seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    sns.histplot(y_train, bins=35, kde=True, ax=ax, color="steelblue",
                 alpha=0.7, label=f"Train n={len(y_train)}")
    sns.histplot(y_test,  bins=35, kde=True, ax=ax, color="tomato",
                 alpha=0.7, label=f"Test  n={len(y_test)}")
    ax.set_xlabel("SoH (%)"); ax.legend()
    ax.set_title(f"Distribution SoH — window={WINDOW_SIZE} cycles")

    ax = axes[1]
    # Comparaison nb fenêtres window=5 vs window=10
    cats = ["Train w=5", "Train w=10", "Test w=5", "Test w=10"]
    vals = [n_train_v3, len(y_train), n_test_v3, len(y_test)]
    colors = ["steelblue","cornflowerblue","tomato","lightsalmon"]
    bars = ax.bar(cats, vals, color=colors)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Nb fenêtres"); ax.set_title("Impact window_size sur N samples")
    ax.set_ylim(0, max(vals) * 1.15)

    plt.suptitle("Pipeline v4b — window_size=10, 11 features", fontweight="bold")
    plt.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS_DIR / "fig14_pipeline_v4b.png", bbox_inches="tight")
    plt.close()
    log.info("Figure -> fig14_pipeline_v4b.png")

    # Résumé livraison
    print()
    print("=" * 58)
    print("  PIPELINE v4b — RÉSULTATS")
    print("=" * 58)
    print(f"  X_train_v4b : {X_train.shape}  float32")
    print(f"  X_test_v4b  : {X_test.shape}   float32")
    print(f"  y_train_v4b : {y_train.shape}   [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  y_test_v4b  : {y_test.shape}    [{y_test.min():.1f},  {y_test.max():.1f}]")
    print(f"  window_size : {WINDOW_SIZE} cycles  (vs 5 en v3)")
    print(f"  n_features  : {N_FEATURES}")
    print("-" * 58)
    print(f"  Baseline Ridge v4b : MAE={ridge_mae:.3f}  R2={ridge_r2:.3f}")
    print(f"  Baseline Ridge v3  : MAE=3.507        R2=0.707  (réf)")
    print("=" * 58)
    log.info("=== Pipeline v4b — DONE ===")


if __name__ == "__main__":
    main()
