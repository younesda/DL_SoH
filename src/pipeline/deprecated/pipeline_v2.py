"""
pipeline_v2.py — Inter-cycle windowing pipeline
Battery SoH LSTM Project

Approche : fenêtres de 5 cycles consécutifs (vs intra-cycle abandonné)
Chaque cycle est agrégé en 7 features avant la création des fenêtres.

Usage : python pipeline_v2.py
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
DATA_PATH   = "battery_health_dataset.csv"
OUTPUT_DIR  = Path(".")

WINDOW_SIZE  = 5      # cycles consécutifs
STRIDE       = 1
BINS_PER_CYCLE = 20

FEATURE_NAMES = ["mean_V", "std_V", "min_V",
                 "mean_T", "std_T", "mean_I", "slope_SoC"]
N_FEATURES = len(FEATURE_NAMES)

SOH_CLIP_MAX  = 100.0
TEMP_CLIP_MAX = 60.0

# Split identique au pipeline v1
TRAIN_BATTERIES = [
    "B0005","B0007","B0025","B0026","B0027","B0029","B0030","B0031",
    "B0032","B0033","B0036","B0038","B0040","B0042","B0043","B0044",
    "B0046","B0047","B0048"
]
TEST_BATTERIES = ["B0006","B0018","B0028","B0034","B0039"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Étape 0 — Chargement & corrections
# ---------------------------------------------------------------------------
def load_and_fix(path: str) -> pd.DataFrame:
    log.info("Chargement : %s", path)
    df = pd.read_csv(path)

    n_soh  = (df["SoH"] > SOH_CLIP_MAX).sum()
    n_temp = (df["Temperature_measured"] > TEMP_CLIP_MAX).sum()

    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP_MAX)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    log.info("Corrections : SoH>100 clippé (%d lignes) | Temp>60 clippée (%d lignes)",
             n_soh, n_temp)
    return df


# ---------------------------------------------------------------------------
# Étape 1 — Agrégation par cycle
# ---------------------------------------------------------------------------
def aggregate_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (battery_id, cycle_number), calcule les 7 features agrégées.
    Retourne un DataFrame avec 1 ligne par cycle.
    """
    records = []

    groups = df.sort_values(
        ["battery_id", "cycle_number", "SoC"], ascending=[True, True, False]
    ).groupby(["battery_id", "cycle_number"], sort=False)

    skipped = 0
    for (bat, cyc), grp in groups:
        if len(grp) != BINS_PER_CYCLE:
            skipped += 1
            continue

        volt  = grp["Voltage_measured"].values
        temp  = grp["Temperature_measured"].values
        curr  = grp["Current_measured"].values
        soc   = grp["SoC"].values
        soh   = grp["SoH"].iloc[0]

        # slope SoC : pente linéaire sur les 20 bins (négatif = décharge)
        slope_soc = np.polyfit(range(BINS_PER_CYCLE), soc, 1)[0]

        records.append({
            "battery_id":   bat,
            "cycle_number": int(cyc),
            "mean_V":       float(volt.mean()),
            "std_V":        float(volt.std()),
            "min_V":        float(volt.min()),
            "mean_T":       float(temp.mean()),
            "std_T":        float(temp.std()),
            "mean_I":       float(curr.mean()),
            "slope_SoC":    float(slope_soc),
            "SoH":          float(soh),
        })

    if skipped:
        log.warning("%d cycles ignorés (n_bins != %d)", skipped, BINS_PER_CYCLE)

    cycle_df = pd.DataFrame(records)
    log.info("Cycles agrégés : %d  (batteries : %d)",
             len(cycle_df), cycle_df["battery_id"].nunique())
    return cycle_df


def validate_aggregation(cycle_df: pd.DataFrame) -> None:
    """Vérifications sur le DataFrame agrégé."""
    # SoH constant par cycle (déjà garanti mais on re-vérifie)
    soh_var = cycle_df.groupby(["battery_id","cycle_number"])["SoH"].std().fillna(0)
    assert (soh_var <= 1e-6).all(), "SoH non-constant dans certains cycles"
    log.info("OK  SoH constant par cycle")

    # slope_SoC < 0 pour la quasi-totalité des cycles
    pct_negative = (cycle_df["slope_SoC"] < 0).mean() * 100
    log.info("OK  slope_SoC < 0 pour %.1f%% des cycles (attendu > 95%%)", pct_negative)
    if pct_negative < 95:
        log.warning("Attention : %.1f%% de cycles avec slope_SoC >= 0", 100 - pct_negative)

    # Stats descriptives des 7 features
    log.info("Stats features agrégées :\n%s",
             cycle_df[FEATURE_NAMES].describe().round(4).to_string())


# ---------------------------------------------------------------------------
# Étape 2 — Fenêtres de 5 cycles (inter-cycle)
# ---------------------------------------------------------------------------
def build_windows(cycle_df: pd.DataFrame,
                  batteries: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Pour chaque batterie du sous-ensemble, crée des fenêtres glissantes
    de WINDOW_SIZE cycles consécutifs (stride=1).

    Retourne :
        X        : (N, WINDOW_SIZE, N_FEATURES)
        y        : (N,)  — SoH du dernier cycle de la fenêtre
        bat_tags : liste de battery_id pour chaque fenêtre (anti-leakage check)
    """
    X_list, y_list, bat_tags = [], [], []

    subset = cycle_df[cycle_df["battery_id"].isin(batteries)].copy()

    for bat in batteries:
        bat_cycles = (
            subset[subset["battery_id"] == bat]
            .sort_values("cycle_number")
            .reset_index(drop=True)
        )
        n = len(bat_cycles)
        n_windows = (n - WINDOW_SIZE) // STRIDE + 1

        if n < WINDOW_SIZE:
            log.warning("Batterie %s : %d cycles < window_size=%d — ignorée",
                        bat, n, WINDOW_SIZE)
            continue

        feats = bat_cycles[FEATURE_NAMES].values  # (n, 7)
        soh   = bat_cycles["SoH"].values           # (n,)

        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            X_list.append(feats[start : start + WINDOW_SIZE])
            y_list.append(soh[start + WINDOW_SIZE - 1])
            bat_tags.append(bat)

        log.info("  %s : %d cycles -> %d fenêtres", bat, n, n_windows)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, bat_tags


# ---------------------------------------------------------------------------
# Étape 3 — Normalisation
# ---------------------------------------------------------------------------
def normalize(X_train: np.ndarray,
              X_test:  np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    N_tr, W, F = X_train.shape
    N_te       = X_test.shape[0]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(
        X_train.reshape(-1, F)
    ).reshape(N_tr, W, F).astype(np.float32)

    X_test_sc = scaler.transform(
        X_test.reshape(-1, F)
    ).reshape(N_te, W, F).astype(np.float32)

    log.info("Scaler mean : %s", np.round(scaler.mean_,  4))
    log.info("Scaler std  : %s", np.round(scaler.scale_, 4))
    return X_train_sc, X_test_sc, scaler.mean_, scaler.scale_


# ---------------------------------------------------------------------------
# Étape 4 — Export
# ---------------------------------------------------------------------------
def export(X_train, X_test, y_train, y_test,
           scaler_mean, scaler_std,
           train_bats, test_bats,
           out_dir: Path) -> None:

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy",  y_test)

    log.info("Exports : X_train%s  X_test%s  y_train%s  y_test%s",
             X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    metadata = {
        "created_at":          datetime.now().isoformat(timespec="seconds"),
        "approach":            "inter_cycle",
        "previous_approach":   "intra_cycle (abandonne - R2=0.379)",
        "window_size":         WINDOW_SIZE,
        "stride":              STRIDE,
        "n_features":          N_FEATURES,
        "feature_names":       FEATURE_NAMES,
        "aggregation":         "per_cycle",
        "bins_per_cycle":      BINS_PER_CYCLE,
        "split_strategy":      "by_battery_id",
        "train_batteries":     train_bats,
        "test_batteries":      test_bats,
        "n_train_samples":     int(len(y_train)),
        "n_test_samples":      int(len(y_test)),
        "soh_clip_value":      SOH_CLIP_MAX,
        "temp_clip_value":     TEMP_CLIP_MAX,
        "cycle_number_removed": True,
        "reason":              "distribution shift inter-battery",
        "scaler_mean":         np.round(scaler_mean, 6).tolist(),
        "scaler_std":          np.round(scaler_std,  6).tolist(),
        "y_train_min":         float(np.round(y_train.min(), 4)),
        "y_train_max":         float(np.round(y_train.max(), 4)),
        "y_train_mean":        float(np.round(y_train.mean(), 4)),
        "y_test_min":          float(np.round(y_test.min(), 4)),
        "y_test_max":          float(np.round(y_test.max(), 4)),
        "y_test_mean":         float(np.round(y_test.mean(), 4)),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    log.info("metadata.json mis a jour")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(X_train, X_test, y_train, y_test,
             bat_tags_train, bat_tags_test) -> None:
    log.info("--- Validation ---")

    assert X_train.shape[2] == N_FEATURES, \
        f"n_features attendu={N_FEATURES}, obtenu={X_train.shape[2]}"
    log.info("OK  X_train.shape = %s", X_train.shape)

    assert X_test.shape[2] == N_FEATURES
    log.info("OK  X_test.shape  = %s", X_test.shape)

    overlap = set(bat_tags_train) & set(bat_tags_test)
    assert len(overlap) == 0, f"Overlap batteries : {overlap}"
    log.info("OK  Aucun overlap batteries train/test")

    assert np.all(y_train <= 100) and np.all(y_train >= 0)
    assert np.all(y_test  <= 100) and np.all(y_test  >= 0)
    log.info("OK  SoH dans [0, 100]")

    assert not np.isnan(X_train).any(), "NaN dans X_train"
    assert not np.isnan(X_test).any(),  "NaN dans X_test"
    log.info("OK  Aucun NaN")

    log.info("  y_train : min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
             y_train.min(), y_train.max(), y_train.mean(), y_train.std())
    log.info("  y_test  : min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
             y_test.min(),  y_test.max(),  y_test.mean(),  y_test.std())
    log.info("--- Toutes les validations passees ---")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_validation(cycle_df, X_train, X_test, y_train, y_test,
                    out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Distribution SoH train vs test
    ax = axes[0, 0]
    sns.histplot(y_train, bins=35, kde=True, ax=ax, color="steelblue",
                 alpha=0.7, label=f"Train (n={len(y_train)})")
    sns.histplot(y_test,  bins=35, kde=True, ax=ax, color="tomato",
                 alpha=0.7, label=f"Test  (n={len(y_test)})")
    ax.set_xlabel("SoH (%)"); ax.set_ylabel("Frequence")
    ax.set_title("Distribution SoH — Train vs Test")
    ax.legend()

    # 2. Nombre de fenetres par batterie
    ax = axes[0, 1]
    all_bats   = TRAIN_BATTERIES + TEST_BATTERIES
    wins_per_bat = {}
    for bat in all_bats:
        n = len(cycle_df[cycle_df["battery_id"] == bat])
        wins_per_bat[bat] = max(0, n - WINDOW_SIZE + 1)
    bat_s   = pd.Series(wins_per_bat).sort_values()
    colors  = ["steelblue" if b in TRAIN_BATTERIES else "tomato"
               for b in bat_s.index]
    ax.barh(bat_s.index, bat_s.values, color=colors)
    ax.set_xlabel("Nb fenetres"); ax.set_title("Fenetres par batterie\n(bleu=train, rouge=test)")

    # 3. SoH moyen par cycle (toutes batteries)
    ax = axes[0, 2]
    for bat in sorted(TRAIN_BATTERIES):
        sub = cycle_df[cycle_df["battery_id"] == bat].sort_values("cycle_number")
        ax.plot(sub["cycle_number"], sub["SoH"], alpha=0.5, linewidth=1,
                color="steelblue")
    for bat in TEST_BATTERIES:
        sub = cycle_df[cycle_df["battery_id"] == bat].sort_values("cycle_number")
        ax.plot(sub["cycle_number"], sub["SoH"], alpha=0.9, linewidth=1.5,
                color="tomato", label=bat)
    ax.axhline(80, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
    ax.set_title("SoH par cycle (bleu=train, rouge=test)")
    ax.legend(fontsize=7)

    # 4. Distribution slope_SoC
    ax = axes[1, 0]
    sns.histplot(cycle_df["slope_SoC"], bins=50, kde=True, ax=ax, color="mediumseagreen")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    pct_neg = (cycle_df["slope_SoC"] < 0).mean() * 100
    ax.set_xlabel("slope_SoC (par bin)")
    ax.set_title(f"slope_SoC — {pct_neg:.1f}% < 0 (decharge)")

    # 5. Premiere fenetre X_train[0] — heatmap
    ax = axes[1, 1]
    im = ax.imshow(X_train[0], aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(FEATURE_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(WINDOW_SIZE))
    ax.set_yticklabels([f"cycle {i}" for i in range(WINDOW_SIZE)], fontsize=8)
    ax.set_title(f"X_train[0] — shape {X_train[0].shape}\n(valeurs normalisees)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(WINDOW_SIZE):
        for j in range(N_FEATURES):
            ax.text(j, i, f"{X_train[0, i, j]:.2f}", ha="center", va="center",
                    fontsize=7)

    # 6. Taille splits
    ax = axes[1, 2]
    bars = ax.bar(["X_train", "X_test"], [len(y_train), len(y_test)],
                  color=["steelblue","tomato"], width=0.45)
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_ylabel("Nb fenetres (cycles)")
    ax.set_title(f"Taille splits — inter-cycle\n(window={WINDOW_SIZE} cycles)")
    ax.set_ylim(0, max(len(y_train), len(y_test)) * 1.2)

    plt.suptitle("Pipeline v2 — Inter-cycle windowing (5 cycles x 7 features)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = out_dir / "fig12_pipeline_v2_validation.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    log.info("Figure validation -> %s", fig_path)


# ---------------------------------------------------------------------------
# Nouvelle baseline Ridge (pour comparaison)
# ---------------------------------------------------------------------------
def compute_new_baseline(X_train, X_test, y_train, y_test) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train.reshape(len(X_train), -1), y_train)
    y_pred = ridge.predict(X_test.reshape(len(X_test), -1))

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    log.info("Nouvelle baseline Ridge (inter-cycle) : MAE=%.4f  RMSE=%.4f  R2=%.4f",
             mae, rmse, r2)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Pipeline v2 — Inter-cycle windowing — START ===")

    # 0. Chargement & corrections
    df = load_and_fix(DATA_PATH)

    # 1. Agrégation par cycle
    cycle_df = aggregate_cycles(df)
    validate_aggregation(cycle_df)

    # 2. Fenêtres inter-cycles
    log.info("Construction des fenetres — train (%d batteries)...",
             len(TRAIN_BATTERIES))
    X_train_raw, y_train, bat_tags_train = build_windows(cycle_df, TRAIN_BATTERIES)

    log.info("Construction des fenetres — test (%d batteries)...",
             len(TEST_BATTERIES))
    X_test_raw, y_test, bat_tags_test = build_windows(cycle_df, TEST_BATTERIES)

    log.info("Shapes avant normalisation : X_train=%s  X_test=%s",
             X_train_raw.shape, X_test_raw.shape)

    # 3. Normalisation
    X_train, X_test, scaler_mean, scaler_std = normalize(X_train_raw, X_test_raw)

    # 4. Export
    export(X_train, X_test, y_train, y_test,
           scaler_mean, scaler_std,
           TRAIN_BATTERIES, TEST_BATTERIES,
           OUTPUT_DIR)

    # Validation
    validate(X_train, X_test, y_train, y_test, bat_tags_train, bat_tags_test)

    # Nouvelle baseline Ridge
    baseline = compute_new_baseline(X_train, X_test, y_train, y_test)

    # Visualisation
    plot_validation(cycle_df, X_train, X_test, y_train, y_test, OUTPUT_DIR)

    # Résumé livraison
    print()
    print("=" * 55)
    print("  PIPELINE v2 — RÉSULTATS")
    print("=" * 55)
    print(f"  X_train : {X_train.shape}  float32")
    print(f"  X_test  : {X_test.shape}   float32")
    print(f"  y_train : {y_train.shape}  "
          f"[{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  y_test  : {y_test.shape}   "
          f"[{y_test.min():.1f}, {y_test.max():.1f}]")
    print(f"  Features ({N_FEATURES}) : {FEATURE_NAMES}")
    print(f"  Approche : inter-cycle, window={WINDOW_SIZE} cycles")
    print("-" * 55)
    print(f"  Nouvelle baseline Ridge :")
    print(f"    MAE={baseline['MAE']}  RMSE={baseline['RMSE']}  R2={baseline['R2']}")
    print("=" * 55)
    print()

    log.info("=== Pipeline v2 — DONE ===")


if __name__ == "__main__":
    main()
