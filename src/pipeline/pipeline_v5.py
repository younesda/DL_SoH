"""
pipeline_v5.py — window_size=10 cycles, 12 features
Battery SoH LSTM Project
Auteur : Younes Hachami

Seul changement vs v4b : ajout de la 12e feature
  internal_resistance_proxy = voltage_drop / abs(mean_current)

Justification : la résistance interne augmente avec la dégradation.
C'est l'indicateur de vieillissement le plus direct.

Usage : python pipeline_v5.py
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

FEATURE_NAMES = [
    "mean_V","std_V","min_V","mean_T","std_T","mean_I","slope_SoC",
    "voltage_drop","capacity_proxy","temp_rise","voltage_end",
    "internal_resistance_proxy",   # ← nouvelle feature (12e)
]
N_FEATURES = len(FEATURE_NAMES)   # 12

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agrégation par cycle (v4b + internal_resistance_proxy)
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

        mean_i        = float(curr.mean())
        voltage_drop  = float(volt.max() - volt.min())
        # Résistance interne proxy : ΔV / |I_mean|
        # abs(mean_i) garanti > 0 car courant de décharge (vérifié à la validation)
        ir_proxy      = voltage_drop / abs(mean_i)

        records.append({
            "battery_id":                   bat,
            "cycle_number":                 int(cyc),
            "mean_V":                       float(volt.mean()),
            "std_V":                        float(volt.std()),
            "min_V":                        float(volt.min()),
            "mean_T":                       float(temp.mean()),
            "std_T":                        float(temp.std()),
            "mean_I":                       mean_i,
            "slope_SoC":                    float(np.polyfit(range(BINS_PER_CYCLE), soc, 1)[0]),
            "voltage_drop":                 voltage_drop,
            "capacity_proxy":               float(abs(mean_i) * BINS_PER_CYCLE),
            "temp_rise":                    float(temp.max() - temp.min()),
            "voltage_end":                  float(volt[-1]),
            "internal_resistance_proxy":    float(ir_proxy),
            "SoH":                          float(grp["SoH"].iloc[0]),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fenêtres de WINDOW_SIZE cycles (stride=1)  — identique v4b
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
            log.warning("  %s : %d cycles < window_size=%d — ignoree", bat, n, WINDOW_SIZE)
            continue

        feats = bat_cyc[FEATURE_NAMES].values
        soh   = bat_cyc["SoH"].values

        for start in range(n_win):
            X_list.append(feats[start : start + WINDOW_SIZE])
            y_list.append(soh[start + WINDOW_SIZE - 1])
            bat_tags.append(bat)

        log.info("  %s : %d cycles -> %d fenetres", bat, n, max(0, n_win))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, bat_tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Pipeline v5 — window_size=%d, %d features — START ===",
             WINDOW_SIZE, N_FEATURES)

    # Chargement & corrections
    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    # Agrégation cycles
    cycle_df = aggregate_cycles(df)
    log.info("Cycles agreges : %d", len(cycle_df))

    # ---------------------------------------------------------------------------
    # Validation internal_resistance_proxy
    # ---------------------------------------------------------------------------
    bad_ir = (cycle_df["internal_resistance_proxy"] <= 0).sum()
    assert bad_ir == 0, f"internal_resistance_proxy <= 0 pour {bad_ir} cycles"
    log.info("OK  internal_resistance_proxy > 0 pour 100%% des cycles")

    r_ir = float(cycle_df[["internal_resistance_proxy", "SoH"]].corr().iloc[0, 1])
    flag = "FORT" if abs(r_ir) > 0.50 else "MOYEN" if abs(r_ir) > 0.30 else "faible"
    log.info("Correlation Pearson internal_resistance_proxy vs SoH : r=%+.4f  [%s]", r_ir, flag)
    if r_ir > -0.30:
        log.warning("r > -0.30 : correlation plus faible qu'attendu (attendu < -0.40)")

    # Corrélations de toutes les features pour comparaison
    log.info("Correlations Pearson toutes features vs SoH :")
    for feat in FEATURE_NAMES:
        r = float(cycle_df[[feat, "SoH"]].corr().iloc[0, 1])
        mk = "***" if abs(r) > 0.50 else "**" if abs(r) > 0.30 else ""
        log.info("  %-32s : r=%+.4f  %s", feat, r, mk)

    # ---------------------------------------------------------------------------
    # Fenêtres
    # ---------------------------------------------------------------------------
    log.info("Train batteries (%d) :", len(TRAIN_BATTERIES))
    X_train_raw, y_train, bat_tr = build_windows(cycle_df, TRAIN_BATTERIES)
    log.info("Test batteries (%d) :", len(TEST_BATTERIES))
    X_test_raw,  y_test,  bat_te = build_windows(cycle_df, TEST_BATTERIES)

    # Validations shapes
    assert len(set(bat_tr) & set(bat_te)) == 0, "Overlap batteries!"
    assert X_train_raw.shape == (944, WINDOW_SIZE, N_FEATURES), \
        f"Shape train inattendue : {X_train_raw.shape}"
    assert X_test_raw.shape  == (299, WINDOW_SIZE, N_FEATURES), \
        f"Shape test inattendue  : {X_test_raw.shape}"
    assert not np.isnan(X_train_raw).any()
    assert not np.isnan(X_test_raw).any()
    log.info("OK  Shapes  — X_train:%s  X_test:%s", X_train_raw.shape, X_test_raw.shape)
    log.info("OK  Pas d'overlap, pas de NaN")

    # ---------------------------------------------------------------------------
    # Normalisation (fit sur train uniquement)
    # ---------------------------------------------------------------------------
    N_tr, W, F = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)
    X_test = scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(len(X_test_raw), W, F).astype(np.float32)
    log.info("Scaler mean (12) : %s", np.round(scaler.mean_, 3))

    # ---------------------------------------------------------------------------
    # Baseline Ridge
    # ---------------------------------------------------------------------------
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    yp = ridge.predict(X_test.reshape(len(X_test), -1))
    ridge_r2   = r2_score(y_test, yp)
    ridge_mae  = mean_absolute_error(y_test, yp)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, yp))
    log.info("Baseline Ridge v5 (window=10, 12feat) : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             ridge_mae, ridge_rmse, ridge_r2)

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------
    np.save(OUTPUT_DIR / "X_train_v5.npy", X_train)
    np.save(OUTPUT_DIR / "X_test_v5.npy",  X_test)
    log.info("Exports : X_train_v5%s  X_test_v5%s", X_train.shape, X_test.shape)
    # y_train / y_test inchangés — on réutilise y_train_v4b / y_test_v4b

    metadata = {
        "created_at":      datetime.now().isoformat(timespec="seconds"),
        "approach":        "inter_cycle",
        "window_size":     WINDOW_SIZE,
        "n_features":      N_FEATURES,
        "feature_names":   FEATURE_NAMES,
        "new_feature":     "internal_resistance_proxy = voltage_drop / abs(mean_current)",
        "train_batteries": TRAIN_BATTERIES,
        "test_batteries":  TEST_BATTERIES,
        "n_train_samples": int(len(y_train)),
        "n_test_samples":  int(len(y_test)),
        "scaler_mean":     np.round(scaler.mean_,  6).tolist(),
        "scaler_std":      np.round(scaler.scale_, 6).tolist(),
        "ir_proxy_pearson_r": round(r_ir, 4),
        "baseline_ridge":  {"MAE": round(ridge_mae,4),
                            "RMSE": round(ridge_rmse,4),
                            "R2": round(ridge_r2,4)},
        "feature_correlations_with_SoH": {
            feat: round(float(cycle_df[[feat,"SoH"]].corr().iloc[0,1]), 4)
            for feat in FEATURE_NAMES
        },
    }
    (OUTPUT_DIR / "metadata_v5.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("metadata_v5.json ecrit")

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------
    import matplotlib.pyplot as plt, seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    corr_vals = pd.Series({
        f: float(cycle_df[[f,"SoH"]].corr().iloc[0,1]) for f in FEATURE_NAMES
    }).sort_values()
    colors = ["tomato" if f == "internal_resistance_proxy" else "steelblue"
              for f in corr_vals.index]
    ax.barh(corr_vals.index, corr_vals.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.30, color="green", linestyle="--", linewidth=1, label="|r|=0.30")
    ax.axvline(-0.30, color="green", linestyle="--", linewidth=1)
    ax.set_xlabel("r de Pearson avec SoH")
    ax.set_title("Correlations features vs SoH\n(rouge=nouvelle feature)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(cycle_df["internal_resistance_proxy"], cycle_df["SoH"],
               alpha=0.3, s=10, color="tomato")
    ax.set_xlabel("internal_resistance_proxy (V/A)")
    ax.set_ylabel("SoH (%)")
    ax.set_title(f"internal_resistance_proxy vs SoH\nr={r_ir:+.4f}")

    plt.suptitle("Pipeline v5 — 12 features (ajout internal_resistance_proxy)",
                 fontweight="bold")
    plt.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS_DIR / "fig15_pipeline_v5.png", bbox_inches="tight")
    plt.close()
    log.info("Figure -> fig15_pipeline_v5.png")

    # ---------------------------------------------------------------------------
    # Résumé livraison
    # ---------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  PIPELINE v5 — RÉSULTATS")
    print("=" * 60)
    print(f"  X_train_v5 : {X_train.shape}  float32")
    print(f"  X_test_v5  : {X_test.shape}   float32")
    print(f"  y_train    : réutiliser y_train_v4b.npy  (944,)")
    print(f"  y_test     : réutiliser y_test_v4b.npy   (299,)")
    print(f"  window_size : {WINDOW_SIZE} cycles")
    print(f"  n_features  : {N_FEATURES} (11 v4b + internal_resistance_proxy)")
    print("-" * 60)
    print(f"  internal_resistance_proxy  r={r_ir:+.4f}  [{flag}]")
    print(f"  (attendu < -0.40)")
    print("-" * 60)
    print(f"  Baseline Ridge v5  : MAE={ridge_mae:.3f}  R2={ridge_r2:.3f}")
    print(f"  Baseline Ridge v4b : MAE=3.537        R2=0.671  (ref)")
    print("=" * 60)
    log.info("=== Pipeline v5 — DONE ===")


if __name__ == "__main__":
    main()
