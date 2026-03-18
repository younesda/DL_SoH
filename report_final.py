"""
report_final.py — Phase 4 : Rapport final Battery SoH LSTM Project
Auteur : Younes Hachami
Génère :
  - fig_report_runs_comparison.png   Tableau visuel + progression R² tous runs
  - fig_report_residuals.png         Analyse résidus Run #7 (6 panneaux)
  - fig_report_feature_importance.png Permutation importance sur Run #7
  - final_report.txt                 Rapport texte complet

Usage : python report_final.py
"""

import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parent
_DATA    = _ROOT / "data" / "processed" / "final"
_FIGS    = _ROOT / "reports" / "figures"
_EXP     = _ROOT / "experiments"
OUT_DIR = _ROOT
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Modèle Run #7 (copie locale pour éviter import circular)
# ---------------------------------------------------------------------------
class LSTMv7(nn.Module):
    def __init__(self, input_size: int = 11):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(128, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out))).squeeze(-1)


def load_model_v7(device):
    ckpt  = torch.load(_ROOT / "model" / "best_lstm_v7.pt", map_location=device, weights_only=True)
    model = LSTMv7(input_size=11).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# A — Figure 1 : Comparaison tous runs
# ---------------------------------------------------------------------------
def fig_runs_comparison():
    log.info("A — Figure comparaison runs...")
    _cols = ['run_id','architecture','batch_size','lr','epochs_run','best_val_loss',
             'MAE_test','RMSE_test','R2_test','delta_MAE','delta_R2','bias','pred_range','notes']
    df = pd.read_csv(_EXP / "experiments_log.csv", names=_cols, skiprows=1, encoding='utf-8-sig')

    # Garder la meilleure ligne par run_id (certains ont des doublons anciens)
    df = (df.sort_values("R2_test", ascending=False)
            .drop_duplicates(subset="run_id", keep="first")
            .sort_values("run_id")
            .reset_index(drop=True))

    # Séparer les runs valides (inter-cycle, run >= 3)
    df_valid = df[df["run_id"] >= 3].copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    # Panel 1 — Progression R²
    ax = axes[0]
    colors = ["gold" if r == 7 else "steelblue" for r in df_valid["run_id"]]
    bars = ax.bar(df_valid["run_id"].astype(str), df_valid["R2_test"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(0.707, color="red",    linestyle="--", linewidth=1.2,
               label="Baseline Ridge w=5 (0.707)")
    ax.axhline(0.671, color="orange", linestyle=":",  linewidth=1.2,
               label="Baseline Ridge w=10 (0.671)")
    ax.axhline(0.85,  color="green",  linestyle="--", linewidth=1,
               label="Objectif cible (0.85)", alpha=0.6)
    ax.set_xlabel("Run #"); ax.set_ylabel("R²")
    ax.set_title("Progression R² — tous runs inter-cycle\n(doré = modèle retenu Run #7)")
    ax.legend(fontsize=7); ax.set_ylim(0.5, 0.95)

    # Panel 2 — MAE
    ax = axes[1]
    colors2 = ["gold" if r == 7 else "tomato" for r in df_valid["run_id"]]
    bars2 = ax.bar(df_valid["run_id"].astype(str), df_valid["MAE_test"], color=colors2,
                   edgecolor="white", linewidth=0.5)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(3.507, color="red", linestyle="--", linewidth=1.2, label="Baseline MAE (3.507)")
    ax.set_xlabel("Run #"); ax.set_ylabel("MAE (%)")
    ax.set_title("MAE test — tous runs\n(doré = modèle retenu)")
    ax.legend(fontsize=7)

    # Panel 3 — Tableau résumé
    ax = axes[2]
    ax.axis("off")
    table_data = []
    headers = ["Run", "Architecture (résumé)", "R²", "MAE", "Biais", "y_pred max"]
    arch_short = {
        3: "LSTM(64) w=5 7f",
        4: "BiLSTM(64) w=5 7f",
        5: "BiLSTM(64) w=5 11f",
        6: "BiLSTM(64) w=5 11f lite",
        7: "BiLSTM(64) w=10 11f ★",
        8: "BiLSTM(128) w=10 11f",
        9: "BiLSTM(64) w=10 12f",
    }
    for _, row in df_valid.iterrows():
        rid = int(row["run_id"])
        pred_max = row["pred_range"].split("-")[1] if isinstance(row.get("pred_range"), str) else "—"
        bias_val = row.get("bias", "—")
        table_data.append([
            f"#{rid}",
            arch_short.get(rid, "—"),
            f"{row['R2_test']:.3f}",
            f"{row['MAE_test']:.3f}",
            f"{float(bias_val):+.3f}" if bias_val != "—" else "—",
            pred_max,
        ])
    tbl = ax.table(cellText=table_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.6)
    # Highlight Run #7
    for j in range(len(headers)):
        tbl[(5, j)].set_facecolor("#FFD700")
        tbl[(5, j)].set_text_props(fontweight="bold")
    ax.set_title("Tableau comparatif complet", pad=15)

    plt.suptitle("Battery SoH LSTM — Comparaison tous runs Phase 3",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = _FIGS / "04_final" / "fig_report_runs_comparison.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("  -> %s", path)


# ---------------------------------------------------------------------------
# B — Figure 2 : Analyse résidus Run #7 (6 panneaux)
# ---------------------------------------------------------------------------
def fig_residuals_v7():
    log.info("B — Figure résidus Run #7...")

    y_true = np.load(_DATA / "y_test_v4b.npy")
    y_pred = np.load(_DATA.parent / "predictions" / "y_pred_v7.npy")
    residuals = y_pred - y_true

    # Mapping fenetre -> batterie (window=10, test batteries)
    df_raw = pd.read_csv(_ROOT / "data" / "raw" / "battery_health_dataset.csv")
    test_bats = ["B0006","B0018","B0028","B0034","B0039"]
    bat_tags, cycle_tags = [], []
    for bat in test_bats:
        n_cyc = df_raw[df_raw["battery_id"] == bat]["cycle_number"].nunique()
        n_win = max(0, n_cyc - 10 + 1)
        for i in range(n_win):
            bat_tags.append(bat)
            cycle_tags.append(i + 10)

    bat_arr   = np.array(bat_tags)
    cycle_arr = np.array(cycle_tags)
    palette   = dict(zip(test_bats, sns.color_palette("tab10", 5)))

    mae_  = mean_absolute_error(y_true, y_pred)
    rmse_ = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_   = r2_score(y_true, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1 — Pred vs True
    ax = axes[0, 0]
    for bat in test_bats:
        mask = bat_arr == bat
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.55, s=18,
                   color=palette[bat], label=bat)
    lims = [69, 100]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y=x")
    ax.fill_betweenx(lims, [l-3 for l in lims], [l+3 for l in lims],
                     alpha=0.08, color="red", label="±3%")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("SoH réel (%)"); ax.set_ylabel("SoH prédit (%)")
    ax.set_title(f"Pred vs True — Run #7\nMAE={mae_:.3f}%  RMSE={rmse_:.3f}%  R²={r2_:.3f}")
    ax.legend(fontsize=7, ncol=2)

    # 2 — Distribution résidus
    ax = axes[0, 1]
    sns.histplot(residuals, bins=40, kde=True, ax=ax, color="coral", alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(residuals.mean(), color="red", linestyle="--",
               label=f"biais={residuals.mean():.3f}%")
    ax.axvline( 3, color="orange", linestyle=":", linewidth=1, label="±3%")
    ax.axvline(-3, color="orange", linestyle=":", linewidth=1)
    pct_3 = (np.abs(residuals) <= 3).mean() * 100
    ax.set_xlabel("Résidu (prédit − réel) %")
    ax.set_title(f"Distribution résidus\nbiais={residuals.mean():.3f}%  std={residuals.std():.3f}%  "
                 f"{pct_3:.1f}% dans ±3%")
    ax.legend(fontsize=8)

    # 3 — MAE par batterie
    ax = axes[0, 2]
    bat_mae = {b: np.abs(residuals[bat_arr == b]).mean() for b in test_bats}
    bat_mae_s = pd.Series(bat_mae).sort_values()
    ax.barh(bat_mae_s.index, bat_mae_s.values,
            color=[palette[b] for b in bat_mae_s.index])
    ax.axvline(mae_, color="red", linestyle="--", linewidth=1,
               label=f"MAE global {mae_:.3f}%")
    ax.set_xlabel("MAE (%)"); ax.legend(fontsize=8)
    ax.set_title("MAE par batterie test")

    # 4 — Résidus vs position cycle
    ax = axes[1, 0]
    for bat in test_bats:
        mask = bat_arr == bat
        ax.scatter(cycle_arr[mask], residuals[mask], alpha=0.4, s=12,
                   color=palette[bat], label=bat)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Dernier cycle de la fenêtre")
    ax.set_ylabel("Résidu (%)")
    ax.set_title("Résidus vs position temporelle\n(biais en fin de vie ?)")
    ax.legend(fontsize=7)

    # 5 — Biais par zone SoH (le plafond à 91%)
    ax = axes[1, 1]
    bins_soh = [70, 80, 85, 90, 95, 101]
    labels_soh = ["70-80", "80-85", "85-90", "90-95", "95-100"]
    zone_mae, zone_bias, zone_n = [], [], []
    for lo, hi, lbl in zip(bins_soh[:-1], bins_soh[1:], labels_soh):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            zone_mae.append(np.abs(residuals[mask]).mean())
            zone_bias.append(residuals[mask].mean())
            zone_n.append(mask.sum())
        else:
            zone_mae.append(0); zone_bias.append(0); zone_n.append(0)

    x = np.arange(len(labels_soh))
    w = 0.35
    b1 = ax.bar(x - w/2, zone_mae,  w, label="MAE",  color="steelblue", alpha=0.8)
    b2 = ax.bar(x + w/2, zone_bias, w, label="Biais", color="coral",     alpha=0.8)
    ax.bar_label(b1, [f"n={n}" for n in zone_n], padding=2, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels_soh)
    ax.set_xlabel("Zone SoH réel (%)"); ax.set_ylabel("Erreur (%)")
    ax.set_title("Erreur par zone SoH\n(plafond ~91% : biais négatif en zone haute ?)")
    ax.legend(fontsize=8)

    # 6 — Distribution y_pred vs y_true
    ax = axes[1, 2]
    sns.histplot(y_true, bins=30, kde=True, ax=ax, color="steelblue",
                 alpha=0.6, label=f"y_test  [{y_true.min():.1f}, {y_true.max():.1f}]")
    sns.histplot(y_pred, bins=30, kde=True, ax=ax, color="tomato",
                 alpha=0.6, label=f"y_pred  [{y_pred.min():.1f}, {y_pred.max():.1f}]")
    ax.axvline(91, color="purple", linestyle="--", linewidth=1.5,
               label="Plafond ~91%")
    ax.set_xlabel("SoH (%)"); ax.legend(fontsize=8)
    ax.set_title("Distribution SoH\nPlafond structurel à ~91%")

    plt.suptitle("Run #7 — Analyse résidus complète  [BiLSTM(64), w=10, 11feat]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = _FIGS / "04_final" / "fig_report_residuals.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("  -> %s", path)


# ---------------------------------------------------------------------------
# C — Figure 3 : Permutation feature importance
# ---------------------------------------------------------------------------
def fig_feature_importance():
    log.info("C — Permutation feature importance Run #7...")

    device = torch.device("cpu")
    model  = load_model_v7(device)

    X_test = np.load(_DATA / "X_test_v4b.npy")
    y_test = np.load(_DATA / "y_test_v4b.npy")
    X_t    = torch.from_numpy(X_test).to(device)

    feature_names = [
        "mean_V","std_V","min_V","mean_T","std_T","mean_I","slope_SoC",
        "voltage_drop","capacity_proxy","temp_rise","voltage_end",
    ]
    N_FEAT = len(feature_names)

    with torch.no_grad():
        y_pred_base = model(X_t).cpu().numpy()
    r2_base = r2_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_pred_base)

    N_REPEATS = 30
    rng = np.random.default_rng(42)
    importance_r2  = np.zeros(N_FEAT)
    importance_mae = np.zeros(N_FEAT)

    for f in range(N_FEAT):
        r2_drops, mae_increases = [], []
        for _ in range(N_REPEATS):
            X_perm = X_test.copy()
            # Permuter la feature f sur tous les timesteps
            idx = rng.permutation(len(X_test))
            X_perm[:, :, f] = X_test[idx, :, f]
            with torch.no_grad():
                y_perm = model(torch.from_numpy(X_perm).to(device)).cpu().numpy()
            r2_drops.append(r2_base - r2_score(y_test, y_perm))
            mae_increases.append(mean_absolute_error(y_test, y_perm) - mae_base)
        importance_r2[f]  = np.mean(r2_drops)
        importance_mae[f] = np.mean(mae_increases)
        log.info("  Feature %-22s : ΔR²=%+.4f  ΔMAE=%+.4f",
                 feature_names[f], importance_r2[f], importance_mae[f])

    # Trier par importance R²
    order = np.argsort(importance_r2)[::-1]
    feat_sorted = [feature_names[i] for i in order]
    r2_sorted   = importance_r2[order]
    mae_sorted  = importance_mae[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = ["tomato" if v > 0.02 else "steelblue" if v > 0 else "lightgray"
              for v in r2_sorted]
    bars = ax.barh(feat_sorted[::-1], r2_sorted[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Chute de R² (permutation)")
    ax.set_title(f"Permutation Importance — ΔR²\n(baseline R²={r2_base:.3f}, {N_REPEATS} répétitions)")
    # Annotations
    for bar, val in zip(bars, r2_sorted[::-1]):
        ax.text(max(val + 0.001, 0.001), bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", fontsize=8)

    ax = axes[1]
    colors2 = ["tomato" if v > 0.05 else "steelblue" if v > 0 else "lightgray"
               for v in mae_sorted]
    bars2 = ax.barh(feat_sorted[::-1], mae_sorted[::-1], color=colors2[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hausse de MAE (permutation) %")
    ax.set_title(f"Permutation Importance — ΔMAE\n(baseline MAE={mae_base:.3f}%, {N_REPEATS} répétitions)")
    for bar, val in zip(bars2, mae_sorted[::-1]):
        ax.text(max(val + 0.002, 0.002), bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", va="center", fontsize=8)

    plt.suptitle("Run #7 — Feature Importance par permutation\n[BiLSTM(64), window=10, 11 features]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = _FIGS / "04_final" / "fig_report_feature_importance.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("  -> %s", path)

    return feat_sorted, r2_sorted, mae_sorted


# ---------------------------------------------------------------------------
# D — Rapport texte
# ---------------------------------------------------------------------------
def write_text_report(feat_sorted, r2_sorted, mae_sorted):
    log.info("D — Écriture rapport texte...")

    _cols2 = ['run_id','architecture','batch_size','lr','epochs_run','best_val_loss',
              'MAE_test','RMSE_test','R2_test','delta_MAE','delta_R2','bias','pred_range','notes']
    df_exp = pd.read_csv(_EXP / "experiments_log.csv", names=_cols2, skiprows=1, encoding='utf-8-sig')
    df_exp = (df_exp.sort_values("R2_test", ascending=False)
                    .drop_duplicates(subset="run_id", keep="first")
                    .sort_values("run_id"))

    y_true = np.load(_DATA / "y_test_v4b.npy")
    y_pred = np.load(_DATA.parent / "predictions" / "y_pred_v7.npy")
    residuals = y_pred - y_true
    pct_3 = (np.abs(residuals) <= 3).mean() * 100

    lines = []
    sep  = "=" * 70
    sep2 = "-" * 70

    lines += [
        sep,
        "  BATTERY SoH LSTM PROJECT — RAPPORT FINAL PHASE 3",
        sep,
        "",
        "1. OBJECTIF",
        sep2,
        "  Prédire le State of Health (SoH) de batteries Li-ion à partir",
        "  de mesures électriques et thermiques de décharge.",
        "  Données   : 24 batteries, 1 459 cycles, 29 180 mesures.",
        "  Split     : 19 batteries train / 5 batteries test (anti-leakage).",
        "  Approche  : fenêtres inter-cycles (10 cycles consécutifs × 11",
        "              features agrégées par cycle).",
        "",
        "2. PROGRESSION DES RUNS",
        sep2,
    ]

    # Tableau runs
    arch_short = {
        3: "LSTM(64) w=5 7feat",
        4: "BiLSTM(64) w=5 7feat",
        5: "BiLSTM(64) w=5 11feat",
        6: "BiLSTM(64) lite w=5 11feat",
        7: "BiLSTM(64) lite w=10 11feat  ★",
        8: "BiLSTM(128) w=10 11feat",
        9: "BiLSTM(64) lite w=10 12feat",
    }
    lines.append(f"  {'Run':<5} {'Architecture':<35} {'R²':>6} {'MAE':>7} {'Biais':>8}  {'Note'}")
    lines.append(f"  {'-'*4} {'-'*35} {'-'*6} {'-'*7} {'-'*8}  {'-'*25}")
    for _, row in df_exp[df_exp["run_id"] >= 3].iterrows():
        rid  = int(row["run_id"])
        note = "RETENU" if rid == 7 else ""
        bias = row.get("bias", 0)
        lines.append(
            f"  #{rid:<4} {arch_short.get(rid,''):<35} {row['R2_test']:>6.3f}"
            f" {row['MAE_test']:>7.3f} {float(bias):>+8.3f}  {note}"
        )
    lines += [
        "",
        f"  Baseline Ridge window=5  : R²=0.707  MAE=3.507",
        f"  Baseline Ridge window=10 : R²=0.671  MAE=3.537",
        "",
    ]

    lines += [
        "3. MODÈLE FINAL — Run #7",
        sep2,
        "  Fichier         : best_lstm_v7.pt",
        "  Architecture    : BiLSTM(64) -> Dropout(0.2) -> FC(32) -> FC(1)",
        "  Paramètres      : 43 585",
        "  Input shape     : (batch, 10 cycles, 11 features)",
        "  Features        : mean_V, std_V, min_V, mean_T, std_T, mean_I,",
        "                    slope_SoC, voltage_drop, capacity_proxy,",
        "                    temp_rise, voltage_end",
        "  Scaler          : StandardScaler fitté sur X_train uniquement",
        "",
        "  Métriques test (299 fenêtres, 5 batteries):",
        f"    MAE   = 2.834 %   (baseline -0.673)",
        f"    RMSE  = 3.565 %   (baseline -0.803)",
        f"    R²    = 0.787     (baseline +0.080)",
        f"    Biais = -0.032 %  (quasi-nul)",
        f"    {pct_3:.1f}% des prédictions dans ±3% d'erreur",
        "",
        "  Entraînement    : best epoch 69 / 150, early stop à 94",
        "  LR schedule     : 5e-4 -> 6.25e-5 (ReduceLROnPlateau)",
        "",
    ]

    lines += [
        "4. FEATURE IMPORTANCE (permutation, 30 répétitions)",
        sep2,
    ]
    for feat, dr2, dmae in zip(feat_sorted, r2_sorted, mae_sorted):
        level = "CRITIQUE" if dr2 > 0.05 else "FORT" if dr2 > 0.02 else "MODERE" if dr2 > 0.005 else "faible"
        lines.append(f"  {feat:<28} ΔR²={dr2:+.4f}  ΔMAE={dmae:+.4f}  [{level}]")
    lines.append("")

    lines += [
        "5. LIMITES CONNUES",
        sep2,
        "  a) Plafond y_pred ~91%",
        "     Le modèle ne prédit pas les SoH > 91%.",
        "     Cause : trop peu de fenêtres train avec SoH > 91%",
        "     (batteries B0046/B0047/B0048 très courtes, dégradation rapide).",
        "     Rééchantillonnage refusé : risque d'overfitting sur 2-3 batteries",
        "     spécifiques sans garantie de généralisation.",
        "",
        "  b) Dataset petit (944 samples train effectifs)",
        "     L'augmentation de capacité (Run #8, BiLSTM 128) a dégradé",
        "     les performances — signe de sous-régime de données.",
        "",
        "  c) Feature internal_resistance_proxy (Run #9)",
        "     r=-0.086 avec SoH, redondante avec voltage_drop.",
        "     Cause : mean_current quasi-constant dans ce protocole de",
        "     décharge — la division annule le signal de voltage_drop.",
        "",
        "  d) Généralisation hors-distribution",
        "     Le modèle est entraîné sur des protocoles NASA/CALCE.",
        "     Performances non garanties sur d'autres protocoles.",
        "",
        "6. FICHIERS LIVRÉS",
        sep2,
        "  best_lstm_v7.pt              Checkpoint modèle final",
        "  inference.py                 Script d'inférence production",
        "  model_config.json            Configuration complète du modèle",
        "  X_train_v4b.npy (944,10,11)  Tenseurs train normalisés",
        "  X_test_v4b.npy  (299,10,11)  Tenseurs test normalisés",
        "  metadata_v4b.json            Paramètres scaler + features",
        "  fig_report_runs_comparison.png",
        "  fig_report_residuals.png",
        "  fig_report_feature_importance.png",
        "",
        sep,
    ]

    text = "\n".join(lines)
    (_ROOT / "reports" / "final_report.txt").write_text(text, encoding="utf-8")
    log.info("  -> final_report.txt")
    print("\n" + text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== Phase 4 — Rapport final START ===")
    fig_runs_comparison()
    fig_residuals_v7()
    feat_sorted, r2_sorted, mae_sorted = fig_feature_importance()
    write_text_report(feat_sorted, r2_sorted, mae_sorted)
    log.info("=== Phase 4 — Rapport final DONE ===")


if __name__ == "__main__":
    main()
