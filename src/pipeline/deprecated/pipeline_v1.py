"""
pipeline.py — Phase 2 : Feature Engineering Pipeline
Battery SoH LSTM Project

Produit :
  X_train.npy, X_test.npy, y_train.npy, y_test.npy
  metadata.json, anomalies_log.txt

Usage : python pipeline.py
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

WINDOW_SIZE = 5
STRIDE      = 1
FEATURES    = ["Voltage_measured", "Current_measured",
               "Temperature_measured", "SoC"]
N_FEATURES  = len(FEATURES)
BINS_PER_CYCLE = 20          # constaté Phase 1 (uniforme)

SOH_CLIP_MAX  = 100.0
TEMP_CLIP_MAX = 60.0

# Batteries exclues du test set (trop peu de cycles)
EXCLUDE_FROM_TEST = {"B0046", "B0047"}

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Étape 0 — Chargement & corrections
# ---------------------------------------------------------------------------
def load_and_fix(path: str) -> tuple[pd.DataFrame, dict]:
    """Charge le CSV, applique les clips SoH et Température, loggue les anomalies."""
    log.info("Chargement : %s", path)
    df = pd.read_csv(path)
    log.info("Shape brute : %s", df.shape)

    anomaly_report = {}

    # ---- SoH clip ----
    soh_mask = df["SoH"] > SOH_CLIP_MAX
    if soh_mask.any():
        affected = df.loc[soh_mask, "battery_id"].unique().tolist()
        detail = (
            df.loc[soh_mask]
            .groupby("battery_id")["SoH"]
            .agg(count="count", min="min", max="max")
            .round(3)
            .to_dict(orient="index")
        )
        anomaly_report["SoH_clip"] = {
            "n_rows":             int(soh_mask.sum()),
            "batteries_affected": affected,
            "detail":             detail,
            "clip_value":         SOH_CLIP_MAX,
        }
        log.warning("SoH > %.0f : %d mesures sur %s → clippées",
                    SOH_CLIP_MAX, soh_mask.sum(), affected)
    df["SoH"] = df["SoH"].clip(upper=SOH_CLIP_MAX)

    # ---- Température clip ----
    temp_mask = df["Temperature_measured"] > TEMP_CLIP_MAX
    if temp_mask.any():
        affected_t = df.loc[temp_mask, "battery_id"].unique().tolist()
        detail_t = (
            df.loc[temp_mask]
            .groupby("battery_id")["Temperature_measured"]
            .agg(count="count", min="min", max="max")
            .round(3)
            .to_dict(orient="index")
        )
        anomaly_report["Temperature_clip"] = {
            "n_rows":             int(temp_mask.sum()),
            "batteries_affected": affected_t,
            "detail":             detail_t,
            "clip_value":         TEMP_CLIP_MAX,
        }
        log.warning("Temperature > %.0f : %d mesures sur %s → clippées",
                    TEMP_CLIP_MAX, temp_mask.sum(), affected_t)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP_MAX)

    return df, anomaly_report


def write_anomaly_log(report: dict, path: Path) -> None:
    lines = [
        "=" * 60,
        f"  ANOMALIES LOG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]
    if not report:
        lines.append("  Aucune anomalie détectée.")
    for key, info in report.items():
        lines += [
            f"[{key}]",
            f"  Clip appliqué à : {info['clip_value']}",
            f"  Rows affectées  : {info['n_rows']}",
            f"  Batteries       : {info['batteries_affected']}",
            "  Détail par batterie :",
        ]
        for bat, stats in info["detail"].items():
            lines.append(f"    {bat:8s} — count={stats['count']}, "
                         f"min={stats['min']}, max={stats['max']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Anomalies loggées → %s", path)


# ---------------------------------------------------------------------------
# Étape 1 — Fenêtres glissantes intra-cycle
# ---------------------------------------------------------------------------
def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Pour chaque (battery_id, cycle_number) de BINS_PER_CYCLE bins :
      - trie les bins par SoC décroissant (ordre de décharge)
      - crée (BINS_PER_CYCLE - WINDOW_SIZE) / STRIDE + 1 fenêtres
      - shape par fenêtre : (WINDOW_SIZE, N_FEATURES)
      - y = SoH du cycle (scalaire)

    Retourne X (N, WINDOW_SIZE, N_FEATURES), y (N,)
    """
    n_windows_per_cycle = (BINS_PER_CYCLE - WINDOW_SIZE) // STRIDE + 1
    log.info("Windows par cycle : %d  (BINS=%d, W=%d, stride=%d)",
             n_windows_per_cycle, BINS_PER_CYCLE, WINDOW_SIZE, STRIDE)

    groups = df.sort_values(
        ["battery_id", "cycle_number", "SoC"], ascending=[True, True, False]
    ).groupby(["battery_id", "cycle_number"], sort=False)

    X_list, y_list = [], []
    skipped = 0

    for (bat, cyc), grp in groups:
        if len(grp) != BINS_PER_CYCLE:
            skipped += 1
            continue  # garde-fou (ne devrait pas arriver)

        feat_matrix = grp[FEATURES].values  # (20, 5)
        soh_label   = grp["SoH"].iloc[0]

        for start in range(0, BINS_PER_CYCLE - WINDOW_SIZE + 1, STRIDE):
            X_list.append(feat_matrix[start : start + WINDOW_SIZE])
            y_list.append(soh_label)

    if skipped:
        log.warning("%d cycles ignorés (n_bins ≠ %d)", skipped, BINS_PER_CYCLE)

    X = np.array(X_list, dtype=np.float32)  # (N, 5, 5)
    y = np.array(y_list, dtype=np.float32)  # (N,)
    log.info("Windows créées : X=%s  y=%s", X.shape, y.shape)
    return X, y


def get_window_battery_map(df: pd.DataFrame) -> np.ndarray:
    """
    Retourne un tableau (N,) de battery_id correspondant à chaque fenêtre.
    Même ordre que build_windows.
    """
    n_windows_per_cycle = (BINS_PER_CYCLE - WINDOW_SIZE) // STRIDE + 1
    groups = df.sort_values(
        ["battery_id", "cycle_number", "SoC"], ascending=[True, True, False]
    ).groupby(["battery_id", "cycle_number"], sort=False)

    bat_labels = []
    for (bat, _), grp in groups:
        if len(grp) != BINS_PER_CYCLE:
            continue
        bat_labels.extend([bat] * n_windows_per_cycle)

    return np.array(bat_labels)


# ---------------------------------------------------------------------------
# Étape 2 — Split par battery_id (anti-leakage strict)
# ---------------------------------------------------------------------------
def split_by_battery(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Sélectionne ~20% des batteries pour le test.
    Critères :
      - B0046 et B0047 exclus du test (peu de cycles)
      - Sélection stratifiée par SoH moyen (représentativité)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # SoH moyen par batterie (pour stratification)
    soh_by_bat = (
        df.groupby("battery_id")["SoH"]
        .mean()
        .sort_values()
        .reset_index()
    )
    soh_by_bat.columns = ["battery_id", "mean_soh"]

    all_batteries      = soh_by_bat["battery_id"].tolist()
    test_eligible      = [b for b in all_batteries if b not in EXCLUDE_FROM_TEST]
    n_test             = max(1, round(len(all_batteries) * (1 - TRAIN_RATIO)))

    # Stratification : découpage en n_test quantiles, 1 batterie par quantile
    soh_eligible = soh_by_bat[soh_by_bat["battery_id"].isin(test_eligible)].copy()
    soh_eligible["quantile"] = pd.qcut(
        soh_eligible["mean_soh"], q=n_test, labels=False, duplicates="drop"
    )
    test_batteries = (
        soh_eligible.groupby("quantile")
        .apply(lambda g: g.sample(1, random_state=RANDOM_SEED)["battery_id"].iloc[0])
        .tolist()
    )
    # Ajustement au cas où qcut produit moins de bins que n_test
    if len(test_batteries) < n_test:
        remaining = [b for b in test_eligible if b not in test_batteries]
        extra = rng.choice(remaining,
                           size=n_test - len(test_batteries),
                           replace=False).tolist()
        test_batteries.extend(extra)

    train_batteries = [b for b in all_batteries if b not in test_batteries]

    log.info("Train batteries (%d) : %s", len(train_batteries), sorted(train_batteries))
    log.info("Test  batteries (%d) : %s", len(test_batteries),  sorted(test_batteries))
    return sorted(train_batteries), sorted(test_batteries)


# ---------------------------------------------------------------------------
# Étape 3 — Normalisation
# ---------------------------------------------------------------------------
def normalize(X_train: np.ndarray, X_test: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit StandardScaler sur X_train (aplati sur les dimensions 0 et 1),
    transforme X_train et X_test.
    Retourne X_train_sc, X_test_sc, scaler_mean, scaler_std.
    """
    N_tr, W, F = X_train.shape
    N_te       = X_test.shape[0]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    X_train_sc = scaler.transform(X_train_2d).reshape(N_tr, W, F).astype(np.float32)
    X_test_sc  = scaler.transform(X_test.reshape(-1, F)).reshape(N_te, W, F).astype(np.float32)

    log.info("Scaler mean : %s", np.round(scaler.mean_, 4))
    log.info("Scaler std  : %s", np.round(scaler.scale_, 4))
    return X_train_sc, X_test_sc, scaler.mean_, scaler.scale_


# ---------------------------------------------------------------------------
# Étape 4 — Export
# ---------------------------------------------------------------------------
def export(X_train, X_test, y_train, y_test,
           train_bats, test_bats,
           scaler_mean, scaler_std,
           soh_clipped: bool, temp_clipped: bool,
           out_dir: Path) -> None:

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy",  y_test)
    log.info("Arrays exportés → X_train%s  X_test%s  y_train%s  y_test%s",
             X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    metadata = {
        "created_at":       datetime.now().isoformat(timespec="seconds"),
        "window_size":      WINDOW_SIZE,
        "stride":           STRIDE,
        "n_features":       N_FEATURES,
        "feature_names":    FEATURES,
        "bins_per_cycle":   BINS_PER_CYCLE,
        "split_strategy":   "by_battery_id",
        "train_batteries":  train_bats,
        "test_batteries":   test_bats,
        "n_train_samples":  int(len(y_train)),
        "n_test_samples":   int(len(y_test)),
        "soh_clip_applied": soh_clipped,
        "soh_clip_value":   SOH_CLIP_MAX,
        "temp_clip_applied":temp_clipped,
        "temp_clip_value":  TEMP_CLIP_MAX,
        "cycle_number_removed": True,
        "reason":           "distribution shift inter-battery",
        "scaler_mean":      np.round(scaler_mean, 6).tolist(),
        "scaler_std":       np.round(scaler_std,  6).tolist(),
        "y_train_min":      float(np.round(y_train.min(), 4)),
        "y_train_max":      float(np.round(y_train.max(), 4)),
        "y_train_mean":     float(np.round(y_train.mean(), 4)),
        "y_test_min":       float(np.round(y_test.min(), 4)),
        "y_test_max":       float(np.round(y_test.max(), 4)),
        "y_test_mean":      float(np.round(y_test.mean(), 4)),
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Métadonnées → %s", meta_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(X_train, X_test, y_train, y_test, train_bats, test_bats,
             bat_map_train, bat_map_test) -> None:
    log.info("─── Validation ───")

    # SoH ≤ 100
    assert np.all(y_train <= 100), "⛔ y_train contient SoH > 100"
    assert np.all(y_test  <= 100), "⛔ y_test  contient SoH > 100"
    log.info("✓ SoH ≤ 100 dans train et test")

    # Pas de fuite inter-split
    overlap = set(train_bats) & set(test_bats)
    assert len(overlap) == 0, f"⛔ Batteries en commun : {overlap}"
    log.info("✓ Aucun battery_id partagé entre train et test")

    # Vérification via bat_map (niveau window)
    overlap_w = set(bat_map_train) & set(bat_map_test)
    assert len(overlap_w) == 0, f"⛔ Overlap au niveau fenêtre : {overlap_w}"
    log.info("✓ Aucun overlap au niveau des fenêtres")

    # Shapes cohérentes
    assert X_train.shape[1:] == (WINDOW_SIZE, N_FEATURES)
    assert X_test.shape[1:]  == (WINDOW_SIZE, N_FEATURES)
    assert len(y_train) == len(X_train)
    assert len(y_test)  == len(X_test)
    log.info("✓ Shapes cohérentes")

    # Distribution SoH
    log.info("  y_train — min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
             y_train.min(), y_train.max(), y_train.mean(), y_train.std())
    log.info("  y_test  — min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
             y_test.min(),  y_test.max(),  y_test.mean(),  y_test.std())

    log.info("─── Toutes les validations passées ✓ ───")


# ---------------------------------------------------------------------------
# Visualisation validation
# ---------------------------------------------------------------------------
def plot_validation(y_train, y_test, X_train, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Distribution y_train vs y_test
    ax = axes[0]
    sns.histplot(y_train, bins=40, kde=True, ax=ax, color="steelblue",
                 alpha=0.7, label=f"Train (n={len(y_train):,})")
    sns.histplot(y_test,  bins=40, kde=True, ax=ax, color="tomato",
                 alpha=0.7, label=f"Test  (n={len(y_test):,})")
    ax.set_xlabel("SoH (%)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution SoH — Train vs Test")
    ax.legend()

    # ── Première fenêtre X_train[0] — heatmap
    ax = axes[1]
    import matplotlib
    feat_labels = FEATURES
    bin_labels  = [f"bin {i}" for i in range(WINDOW_SIZE)]
    im = ax.imshow(X_train[0], aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(feat_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(WINDOW_SIZE))
    ax.set_yticklabels(bin_labels, fontsize=9)
    ax.set_title(f"X_train[0] — shape {X_train[0].shape}\n(valeurs normalisées)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(WINDOW_SIZE):
        for j in range(N_FEATURES):
            ax.text(j, i, f"{X_train[0, i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    # ── Taille des splits
    ax = axes[2]
    bars = ax.bar(["X_train", "X_test"], [len(y_train), len(y_test)],
                  color=["steelblue", "tomato"], width=0.5)
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_ylabel("Nombre de fenêtres")
    ax.set_title("Taille des splits")
    ax.set_ylim(0, max(len(y_train), len(y_test)) * 1.15)

    plt.suptitle("Validation Phase 2 — Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = out_dir / "fig7_pipeline_validation.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    log.info("Figure validation → %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("═══ Pipeline Phase 2 — START ═══")

    # 0. Chargement & corrections
    df, anomaly_report = load_and_fix(DATA_PATH)
    write_anomaly_log(anomaly_report, OUTPUT_DIR / "anomalies_log.txt")
    soh_clipped  = "SoH_clip"         in anomaly_report
    temp_clipped = "Temperature_clip" in anomaly_report

    # 1. Fenêtres glissantes + map batterie
    X, y      = build_windows(df)
    bat_map   = get_window_battery_map(df)

    # 2. Split par battery_id
    train_bats, test_bats = split_by_battery(df)

    train_mask = np.isin(bat_map, train_bats)
    test_mask  = np.isin(bat_map, test_bats)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    bat_map_train    = bat_map[train_mask]
    bat_map_test     = bat_map[test_mask]

    log.info("Avant normalisation — X_train:%s  X_test:%s", X_train.shape, X_test.shape)

    # 3. Normalisation
    X_train_sc, X_test_sc, scaler_mean, scaler_std = normalize(X_train, X_test)

    # 4. Export
    export(X_train_sc, X_test_sc, y_train, y_test,
           train_bats, test_bats,
           scaler_mean, scaler_std,
           soh_clipped, temp_clipped,
           OUTPUT_DIR)

    # Validation
    validate(X_train_sc, X_test_sc, y_train, y_test,
             train_bats, test_bats, bat_map_train, bat_map_test)

    # Visualisation
    plot_validation(y_train, y_test, X_train_sc, OUTPUT_DIR)

    log.info("═══ Pipeline Phase 2 — DONE ═══")


if __name__ == "__main__":
    main()
