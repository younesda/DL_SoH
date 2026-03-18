"""
pipeline_v3.py — Feature enrichment (7 → 11 features par cycle)
Battery SoH LSTM Project

Nouvelles features ajoutées :
  voltage_drop    = max(V) - min(V)
  capacity_proxy  = abs(mean(I)) * n_bins
  temp_rise       = max(T) - min(T)
  voltage_end     = V du dernier bin (fin de décharge, SoC le plus bas)

Usage : python pipeline_v3.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config (identique pipeline_v2)
# ---------------------------------------------------------------------------
DATA_PATH   = "battery_health_dataset.csv"
OUTPUT_DIR  = Path(".")
BINS_PER_CYCLE = 20
WINDOW_SIZE    = 5
SOH_CLIP_MAX   = 100.0
TEMP_CLIP_MAX  = 60.0

TRAIN_BATTERIES = [
    "B0005","B0007","B0025","B0026","B0027","B0029","B0030","B0031",
    "B0032","B0033","B0036","B0038","B0040","B0042","B0043","B0044",
    "B0046","B0047","B0048"
]
TEST_BATTERIES = ["B0006","B0018","B0028","B0034","B0039"]

# Features v2 (existantes) + nouvelles
FEATURES_V2 = ["mean_V", "std_V", "min_V",
                "mean_T", "std_T", "mean_I", "slope_SoC"]
FEATURES_NEW = ["voltage_drop", "capacity_proxy", "temp_rise", "voltage_end"]
FEATURE_NAMES = FEATURES_V2 + FEATURES_NEW
N_FEATURES = len(FEATURE_NAMES)   # 11

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agrégation enrichie par cycle
# ---------------------------------------------------------------------------
def aggregate_cycles_v3(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = df.sort_values(
        ["battery_id", "cycle_number", "SoC"], ascending=[True, True, False]
    ).groupby(["battery_id", "cycle_number"], sort=False)

    skipped = 0
    for (bat, cyc), grp in groups:
        if len(grp) != BINS_PER_CYCLE:
            skipped += 1
            continue

        volt = grp["Voltage_measured"].values    # trié SoC décroissant
        temp = grp["Temperature_measured"].values
        curr = grp["Current_measured"].values
        soc  = grp["SoC"].values
        soh  = grp["SoH"].iloc[0]

        # --- Features v2 ---
        slope_soc = float(np.polyfit(range(BINS_PER_CYCLE), soc, 1)[0])

        # --- Nouvelles features ---
        voltage_drop   = float(volt.max() - volt.min())       # amplitude tension
        capacity_proxy = float(abs(curr.mean()) * BINS_PER_CYCLE)  # proxy capacité
        temp_rise      = float(temp.max() - temp.min())        # montée en température
        voltage_end    = float(volt[-1])                       # tension fin décharge (SoC min)

        records.append({
            "battery_id":     bat,
            "cycle_number":   int(cyc),
            # v2
            "mean_V":         float(volt.mean()),
            "std_V":          float(volt.std()),
            "min_V":          float(volt.min()),
            "mean_T":         float(temp.mean()),
            "std_T":          float(temp.std()),
            "mean_I":         float(curr.mean()),
            "slope_SoC":      slope_soc,
            # nouvelles
            "voltage_drop":   voltage_drop,
            "capacity_proxy": capacity_proxy,
            "temp_rise":      temp_rise,
            "voltage_end":    voltage_end,
            "SoH":            float(soh),
        })

    if skipped:
        log.warning("%d cycles ignores (n_bins != %d)", skipped, BINS_PER_CYCLE)

    cycle_df = pd.DataFrame(records)
    log.info("Cycles agrégés v3 : %d  (batteries : %d)",
             len(cycle_df), cycle_df["battery_id"].nunique())
    return cycle_df


# ---------------------------------------------------------------------------
# Validation des nouvelles features
# ---------------------------------------------------------------------------
def validate_features(cycle_df: pd.DataFrame) -> None:
    log.info("--- Validation features v3 ---")

    # voltage_drop > 0
    bad_vdrop = (cycle_df["voltage_drop"] <= 0).sum()
    assert bad_vdrop == 0, f"voltage_drop <= 0 pour {bad_vdrop} cycles"
    log.info("OK  voltage_drop > 0 pour 100%% des cycles")

    # capacity_proxy > 0
    bad_cap = (cycle_df["capacity_proxy"] <= 0).sum()
    assert bad_cap == 0, f"capacity_proxy <= 0 pour {bad_cap} cycles"
    log.info("OK  capacity_proxy > 0 pour 100%% des cycles")

    # Corrélations Pearson des nouvelles features avec SoH
    log.info("Corrélations Pearson avec SoH :")
    corrs = {}
    for feat in FEATURE_NAMES:
        r = float(cycle_df[[feat, "SoH"]].corr().iloc[0, 1])
        corrs[feat] = r
        flag = "***" if abs(r) > 0.50 else "**" if abs(r) > 0.30 else ""
        log.info("  %-18s : r=%+.4f  %s", feat, r, flag)

    new_feat_corrs = {f: corrs[f] for f in FEATURES_NEW}
    n_strong = sum(abs(v) > 0.30 for v in new_feat_corrs.values())
    log.info("Nouvelles features avec |r| > 0.30 : %d/4  (attendu >= 2)", n_strong)
    if n_strong < 2:
        log.warning("Attention : moins de 2 nouvelles features fortement corrélées avec SoH")

    # Stats descriptives
    log.info("Stats nouvelles features :\n%s",
             cycle_df[FEATURES_NEW].describe().round(4).to_string())


# ---------------------------------------------------------------------------
# Fenêtres inter-cycles (identique v2)
# ---------------------------------------------------------------------------
def build_windows(cycle_df, batteries):
    X_list, y_list, bat_tags = [], [], []
    subset = cycle_df[cycle_df["battery_id"].isin(batteries)]

    for bat in batteries:
        bat_cyc = (subset[subset["battery_id"] == bat]
                   .sort_values("cycle_number")
                   .reset_index(drop=True))
        n = len(bat_cyc)
        if n < WINDOW_SIZE:
            log.warning("Batterie %s : %d cycles < %d — ignorée", bat, n, WINDOW_SIZE)
            continue

        feats = bat_cyc[FEATURE_NAMES].values
        soh   = bat_cyc["SoH"].values

        for start in range(0, n - WINDOW_SIZE + 1):
            X_list.append(feats[start:start + WINDOW_SIZE])
            y_list.append(soh[start + WINDOW_SIZE - 1])
            bat_tags.append(bat)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, bat_tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Pipeline v3 — Feature enrichment (7 -> 11) — START ===")

    # Chargement & corrections
    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)
    log.info("Dataset : %d lignes, %d batteries", len(df), df["battery_id"].nunique())

    # Agrégation enrichie
    cycle_df = aggregate_cycles_v3(df)
    validate_features(cycle_df)

    # Fenêtres
    X_train_raw, y_train, bat_tr = build_windows(cycle_df, TRAIN_BATTERIES)
    X_test_raw,  y_test,  bat_te = build_windows(cycle_df, TEST_BATTERIES)
    log.info("Shapes brutes — train:%s  test:%s", X_train_raw.shape, X_test_raw.shape)

    # Vérif overlap
    assert len(set(bat_tr) & set(bat_te)) == 0, "Overlap batteries!"

    # Normalisation
    N_tr, W, F = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train_raw.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)
    X_test = scaler.transform(
        X_test_raw.reshape(-1, F)
    ).reshape(len(X_test_raw), W, F).astype(np.float32)

    # Backup des anciens fichiers (5,7)
    import shutil
    for fname in ["X_train.npy", "X_test.npy"]:
        src = OUTPUT_DIR / fname
        dst = OUTPUT_DIR / fname.replace(".npy", "_v2_backup.npy")
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            log.info("Backup : %s -> %s", fname, dst.name)

    # Export v3
    np.save(OUTPUT_DIR / "X_train_v3.npy", X_train)
    np.save(OUTPUT_DIR / "X_test_v3.npy",  X_test)
    np.save(OUTPUT_DIR / "y_train.npy",     y_train)   # y inchangé
    np.save(OUTPUT_DIR / "y_test.npy",      y_test)
    log.info("Exports : X_train_v3%s  X_test_v3%s", X_train.shape, X_test.shape)

    # Baseline Ridge sur v3
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    yp = ridge.predict(X_test.reshape(len(X_test), -1))
    ridge_mae  = mean_absolute_error(y_test, yp)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, yp))
    ridge_r2   = r2_score(y_test, yp)
    log.info("Baseline Ridge v3 (11 feat) : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             ridge_mae, ridge_rmse, ridge_r2)

    # Metadata v3
    metadata_v3 = {
        "created_at":       datetime.now().isoformat(timespec="seconds"),
        "approach":         "inter_cycle",
        "window_size":      WINDOW_SIZE,
        "n_features":       N_FEATURES,
        "feature_names":    FEATURE_NAMES,
        "features_v2":      FEATURES_V2,
        "features_new":     FEATURES_NEW,
        "aggregation":      "per_cycle",
        "train_batteries":  TRAIN_BATTERIES,
        "test_batteries":   TEST_BATTERIES,
        "n_train_samples":  int(len(y_train)),
        "n_test_samples":   int(len(y_test)),
        "scaler_mean":      np.round(scaler.mean_,  6).tolist(),
        "scaler_std":       np.round(scaler.scale_, 6).tolist(),
        "baseline_ridge_v3": {
            "MAE": round(ridge_mae, 4),
            "RMSE": round(ridge_rmse, 4),
            "R2": round(ridge_r2, 4),
        },
        "feature_correlations_with_SoH": {
            feat: round(float(cycle_df[[feat,"SoH"]].corr().iloc[0,1]), 4)
            for feat in FEATURE_NAMES
        },
    }
    (OUTPUT_DIR / "metadata_v3.json").write_text(
        json.dumps(metadata_v3, indent=2), encoding="utf-8"
    )
    log.info("metadata_v3.json écrit")

    # Heatmap corrélations
    import matplotlib.pyplot as plt, seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    corr_vals = pd.Series({f: float(cycle_df[[f,"SoH"]].corr().iloc[0,1])
                           for f in FEATURE_NAMES}).sort_values()
    colors = ["tomato" if f in FEATURES_NEW else "steelblue" for f in corr_vals.index]
    ax.barh(corr_vals.index, corr_vals.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline( 0.30, color="green", linestyle="--", linewidth=1, label="|r|=0.30")
    ax.axvline(-0.30, color="green", linestyle="--", linewidth=1)
    ax.set_xlabel("r de Pearson avec SoH")
    ax.set_title("Corrélations features vs SoH\n(rouge=nouvelles, bleu=existantes)")
    ax.legend(fontsize=8)

    ax = axes[1]
    corr_matrix = cycle_df[FEATURE_NAMES + ["SoH"]].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.3,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Matrice corrélation — 11 features + SoH")
    plt.suptitle("Pipeline v3 — Feature enrichment (7 -> 11)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig13_features_v3_correlations.png", bbox_inches="tight")
    plt.close()
    log.info("Figure corrélations -> fig13_features_v3_correlations.png")

    # Résumé
    print()
    print("=" * 58)
    print("  PIPELINE v3 — LIVRAISON")
    print("=" * 58)
    print(f"  X_train_v3 : {X_train.shape}  float32")
    print(f"  X_test_v3  : {X_test.shape}   float32")
    print(f"  Features   : {N_FEATURES} = 7 existantes + 4 nouvelles")
    print(f"  Nouvelles  : {FEATURES_NEW}")
    print()
    print("  Corrélations nouvelles features avec SoH :")
    for feat in FEATURES_NEW:
        r = metadata_v3["feature_correlations_with_SoH"][feat]
        flag = "FORT" if abs(r) > 0.50 else "MOD" if abs(r) > 0.30 else "faible"
        print(f"    {feat:<18} : r={r:+.4f}  [{flag}]")
    print()
    print(f"  Baseline Ridge v3 : MAE={ridge_mae:.3f}  R2={ridge_r2:.3f}")
    print(f"  Baseline Ridge v2 : MAE=3.507        R2=0.707  (reference)")
    print("=" * 58)
    print()

    log.info("=== Pipeline v3 — DONE ===")


if __name__ == "__main__":
    main()
