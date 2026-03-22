"""
dashboard.py — Battery SoH Dashboard
Auteur : Younes Hachami

Lancement :
    pip install streamlit plotly
    streamlit run dashboard.py

Note Colab : Streamlit nécessite un tunnel pour fonctionner sur Colab.
    !pip install streamlit plotly pyngrok -q
    from pyngrok import ngrok
    ngrok.set_auth_token("VOTRE_TOKEN")  # gratuit sur ngrok.com
    !streamlit run dashboard.py &
    print(ngrok.connect(8501))
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Config — chargée depuis model_config.json
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "model" / "model_config.json"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = json.load(f)

RUN_ID         = CFG["model"]["run_id"]
CHECKPOINT     = ROOT / "model" / CFG["model"]["checkpoint"]
ARCH_DESC      = CFG["model"]["architecture"]
N_PARAMS       = CFG["model"]["n_params"]
WINDOW_SIZE    = CFG["features"]["window_size"]
N_FEATURES     = CFG["features"]["n_features"]
ALL_FEATURES   = CFG["features"]["feature_names"]          # 13 features (incl. soh_prev + soh_lin_extrap)
ELEC_FEATURES  = ALL_FEATURES[:11]                         # 11 features électriques
ENSEMBLE_ALPHA = CFG["ensemble"]["alpha_lstm"]             # 0.05
TRAIN_BATTERIES = CFG["data_split"]["train_batteries"]
TEST_BATTERIES  = CFG["data_split"]["test_batteries"]
BASELINE        = {
    "MAE":  CFG["baselines"]["ridge_inter_cycle"]["MAE"],
    "RMSE": CFG["baselines"]["ridge_inter_cycle"]["RMSE"],
    "R2":   CFG["baselines"]["ridge_inter_cycle"]["R2"],
}

DATA_PATH      = ROOT / "data" / "raw" / "battery_health_dataset.csv"
LOG_CSV        = ROOT / "experiments" / "training_logs" / f"training_log_v{RUN_ID}.csv"

BINS_PER_CYCLE = 20
SOH_CLIP       = 100.0
TEMP_CLIP      = 60.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Modèle — identique à LSTMv11 dans train_lstm.py
# ─────────────────────────────────────────────────────────────────────────────
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):                              # (B, T, H)
        weights = torch.softmax(self.score(lstm_out), dim=1) # (B, T, 1)
        return (weights * lstm_out).sum(dim=1)               # (B, H)


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=13, hidden=64):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden * 2)
        self.dropout   = nn.Dropout(0.2)
        self.fc1       = nn.Linear(hidden * 2, 32)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx    = self.attention(out)
        ctx    = self.dropout(ctx)
        return self.fc2(self.relu(self.fc1(ctx))).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — miroir de pipeline_v7.py
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_cycles(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups  = df.sort_values(
        ["battery_id", "cycle_number", "SoC"], ascending=[True, True, False]
    ).groupby(["battery_id", "cycle_number"], sort=False)

    for (bat, cyc), grp in groups:
        if len(grp) != BINS_PER_CYCLE:
            continue
        V   = grp["Voltage_measured"].values
        T   = grp["Temperature_measured"].values
        I   = grp["Current_measured"].values
        soc = grp["SoC"].values
        records.append({
            "battery_id":     bat,
            "cycle_number":   int(cyc),
            "mean_V":         float(V.mean()),
            "std_V":          float(V.std()),
            "min_V":          float(V.min()),
            "mean_T":         float(T.mean()),
            "std_T":          float(T.std()),
            "mean_I":         float(I.mean()),
            "slope_SoC":      float(np.polyfit(range(BINS_PER_CYCLE), soc, 1)[0]),
            "voltage_drop":   float(V.max() - V.min()),
            "capacity_proxy": float(abs(I.mean()) * BINS_PER_CYCLE),
            "temp_rise":      float(T.max() - T.min()),
            "voltage_end":    float(V[-1]),
            "SoH":            float(grp["SoH"].iloc[0]),
        })
    return pd.DataFrame(records)


def build_windows(cycle_df: pd.DataFrame, batteries: list):
    X_list, y_list, bat_tags = [], [], []
    subset = cycle_df[cycle_df["battery_id"].isin(batteries)]
    for bat in batteries:
        bat_cyc = (subset[subset["battery_id"] == bat]
                   .sort_values("cycle_number").reset_index(drop=True))
        n     = len(bat_cyc)
        n_win = n - WINDOW_SIZE + 1
        if n < WINDOW_SIZE:
            continue
        elec = bat_cyc[ELEC_FEATURES].values
        soh  = bat_cyc["SoH"].values
        for start in range(n_win):
            end      = start + WINDOW_SIZE
            elec_w   = elec[start:end]
            soh_w    = soh[start:end]
            # soh_prev : décalé de 1 au dernier timestep (pas de leakage)
            soh_prev = soh_w.copy()
            soh_prev[-1] = soh_w[-2]
            # soh_lin_extrap : régression linéaire sur les 9 valeurs connues
            slope_val  = np.polyfit(np.arange(9, dtype=np.float64),
                                    soh_prev[:9].astype(np.float64), 1)[0]
            extrap_val = float(soh_prev[8] + slope_val)
            extrap_col = np.full((WINDOW_SIZE, 1), extrap_val, dtype=np.float32)
            window = np.concatenate([elec_w,
                                     soh_prev.reshape(-1, 1),
                                     extrap_col], axis=1)   # (10, 13)
            X_list.append(window)
            y_list.append(soh_w[-1])
            bat_tags.append(bat)
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            bat_tags)


def predict_window(cycle_df: pd.DataFrame, bat: str, start_idx: int,
                   scaler: StandardScaler, model: nn.Module,
                   ridge: Ridge) -> tuple:
    """
    Prédit le SoH au cycle (start_idx + WINDOW_SIZE) pour une batterie donnée.
    Retourne (soh_pred, soh_true, window_cycles, target_cycle, elec_w, soh_w, soh_prev).
    soh_pred = ensemble LSTM*alpha + Ridge*(1-alpha).
    """
    bat_cyc = (cycle_df[cycle_df["battery_id"] == bat]
               .sort_values("cycle_number").reset_index(drop=True))
    elec   = bat_cyc[ELEC_FEATURES].values
    soh    = bat_cyc["SoH"].values
    cycles = bat_cyc["cycle_number"].values

    elec_w   = elec[start_idx : start_idx + WINDOW_SIZE]
    soh_w    = soh[start_idx  : start_idx + WINDOW_SIZE]
    soh_prev = soh_w.copy()
    soh_prev[-1] = soh_w[-2]
    slope_val  = np.polyfit(np.arange(9, dtype=np.float64),
                            soh_prev[:9].astype(np.float64), 1)[0]
    extrap_val = float(soh_prev[8] + slope_val)
    extrap_col = np.full((WINDOW_SIZE, 1), extrap_val, dtype=np.float32)

    window   = np.concatenate([elec_w, soh_prev.reshape(-1, 1), extrap_col], axis=1)  # (10,13)
    F        = window.shape[1]
    window_s = scaler.transform(window.reshape(-1, F)).reshape(1, WINDOW_SIZE, F).astype(np.float32)

    with torch.no_grad():
        soh_lstm = model(torch.FloatTensor(window_s).to(DEVICE)).cpu().item()

    soh_ridge = float(ridge.predict(window_s.reshape(1, -1))[0])
    soh_pred  = ENSEMBLE_ALPHA * soh_lstm + (1 - ENSEMBLE_ALPHA) * soh_ridge

    soh_true      = float(soh[start_idx + WINDOW_SIZE])
    window_cycles = cycles[start_idx : start_idx + WINDOW_SIZE].tolist()
    target_cycle  = int(cycles[start_idx + WINDOW_SIZE])

    return soh_pred, soh_true, window_cycles, target_cycle, elec_w, soh_w, soh_prev


# ─────────────────────────────────────────────────────────────────────────────
# Cache Streamlit
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["SoH"]                  = df["SoH"].clip(upper=SOH_CLIP)
    df["Temperature_measured"] = df["Temperature_measured"].clip(upper=TEMP_CLIP)
    cycle_df = aggregate_cycles(df)

    X_train_raw, y_train, _        = build_windows(cycle_df, TRAIN_BATTERIES)
    X_test_raw,  y_test,  bat_tags = build_windows(cycle_df, TEST_BATTERIES)

    N_tr, W, F = X_train_raw.shape
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(N_tr, W, F).astype(np.float32)
    X_test  = scaler.transform(X_test_raw.reshape(-1, F)).reshape(len(X_test_raw), W, F).astype(np.float32)

    return df, cycle_df, X_train, y_train, X_test, y_test, bat_tags, scaler


@st.cache_resource
def load_model():
    model = BiLSTMAttention(input_size=N_FEATURES).to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@st.cache_data
def load_log():
    if LOG_CSV.exists():
        return pd.read_csv(LOG_CSV)
    return None


@st.cache_resource
def fit_ridge(_X_train, _y_train):
    ridge = Ridge(alpha=1.0)
    ridge.fit(_X_train.reshape(len(_X_train), -1), _y_train)
    return ridge


@st.cache_data
def get_predictions(_model, _ridge, X_test):
    with torch.no_grad():
        y_lstm = _model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
    y_ridge = _ridge.predict(X_test.reshape(len(X_test), -1))
    return ENSEMBLE_ALPHA * y_lstm + (1 - ENSEMBLE_ALPHA) * y_ridge


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery SoH Dashboard",
    page_icon="🔋",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Chargement — avec gestion d'erreur explicite
# ─────────────────────────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    st.error(f"Données introuvables : `{DATA_PATH}`")
    st.stop()
if not CHECKPOINT.exists():
    st.error(f"Checkpoint introuvable : `{CHECKPOINT}`\nLancez : `python src/training/train_lstm.py --run {RUN_ID}`")
    st.stop()

with st.spinner("Chargement des données et du modèle..."):
    df, cycle_df, X_train, y_train, X_test, y_test, bat_tags, scaler = load_data()
    model, ckpt = load_model()
    ridge       = fit_ridge(X_train, y_train)
    log_df      = load_log()
    y_pred      = get_predictions(model, ridge, X_test)
    bat_tags_arr = np.array(bat_tags)

mae_global  = mean_absolute_error(y_test, y_pred)
rmse_global = math.sqrt(mean_squared_error(y_test, y_pred))
r2_global   = r2_score(y_test, y_pred)
bias_global = float((y_pred - y_test).mean())
residuals   = y_pred - y_test

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔋 Battery State of Health — Dashboard")
st.caption(f"Run #{RUN_ID} + Ridge ensemble (α={ENSEMBLE_ALPHA}) | {N_FEATURES} features (soh_prev + soh_lin_extrap) | Battery-wise split")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Section", [
        "📊 Métriques globales",
        "🔍 Analyse par batterie",
        "📈 Courbes d'apprentissage",
        "🎯 Prédiction interactive",
    ])
    st.divider()
    st.markdown("**Modèle**")
    st.code(CHECKPOINT.name, language=None)
    st.markdown(f"**Run** : #{RUN_ID}")
    st.markdown(f"**Architecture** : `{ARCH_DESC}`")
    st.markdown(f"**Paramètres** : {N_PARAMS:,}")
    st.markdown(f"**Features** : {N_FEATURES} (`soh_prev` + `soh_lin_extrap`)")
    st.markdown(f"**Device** : `{DEVICE}`")
    st.markdown(f"**Train** : {len(TRAIN_BATTERIES)} batteries · {len(y_train)} fenêtres")
    st.markdown(f"**Test**  : {len(TEST_BATTERIES)} batteries · {len(y_test)} fenêtres")
    st.divider()
    st.markdown("**Test batteries** ✅ (inconnues du modèle)")
    for b in TEST_BATTERIES:
        st.markdown(f"- `{b}`")
    st.markdown("**Train batteries** ⚠️ (vues à l'entraînement)")
    for b in TRAIN_BATTERIES:
        st.markdown(f"- `{b}`")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Métriques globales
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Métriques globales":

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",   f"{mae_global:.3f} %",
              delta=f"{mae_global - BASELINE['MAE']:+.3f} vs baseline Ridge",
              delta_color="inverse")
    c2.metric("RMSE",  f"{rmse_global:.3f} %",
              delta=f"{rmse_global - BASELINE['RMSE']:+.3f} vs baseline Ridge",
              delta_color="inverse")
    c3.metric("R²",    f"{r2_global:.4f}",
              delta=f"{r2_global - BASELINE['R2']:+.4f} vs baseline Ridge")
    c4.metric("Biais", f"{bias_global:+.3f} %",
              help="Biais moyen des résidus (idéal = 0)")

    st.divider()
    col_left, col_right = st.columns(2)

    # Scatter prédit vs réel — coloré par batterie
    with col_left:
        st.subheader("Prédit vs Réel")
        color_map = {bat: px.colors.qualitative.T10[i % len(px.colors.qualitative.T10)] for i, bat in enumerate(TEST_BATTERIES)}
        fig = go.Figure()
        for bat in TEST_BATTERIES:
            mask = bat_tags_arr == bat
            fig.add_trace(go.Scatter(
                x=y_test[mask].tolist(), y=y_pred[mask].tolist(),
                mode="markers", name=bat,
                marker=dict(color=color_map[bat], size=6, opacity=0.7),
                hovertemplate=f"<b>{bat}</b><br>Réel: %{{x:.2f}}%<br>Prédit: %{{y:.2f}}%",
            ))
        rng = [float(min(y_test.min(), y_pred.min())) - 1,
               float(max(y_test.max(), y_pred.max())) + 1]
        fig.add_trace(go.Scatter(x=rng, y=rng, mode="lines",
                                 line=dict(color="red", dash="dash", width=1.5),
                                 name="Parfait", showlegend=True))
        fig.update_layout(xaxis_title="SoH réel (%)", yaxis_title="SoH prédit (%)",
                          height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    # Distribution résidus
    with col_right:
        st.subheader("Distribution des résidus")
        fig = px.histogram(x=residuals, nbins=50,
                           color_discrete_sequence=["#e07b6a"],
                           labels={"x": "Résidu (prédit − réel, %)"})
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1.5)
        fig.add_vline(x=float(residuals.mean()), line_dash="dot", line_color="red",
                      annotation_text=f"Biais = {residuals.mean():.3f}%",
                      annotation_position="top right")
        fig.update_layout(yaxis_title="Fréquence", height=420, showlegend=False,
                          bargap=0.05)
        st.plotly_chart(fig, width="stretch")

    # Tableau comparatif des approches
    st.subheader("Comparaison des approches")
    bl = CFG["baselines"]
    comp = pd.DataFrame([
        {"Approche": "Baseline Ridge inter-cycle",       "Fenêtre": "—",      "Features": 11, "MAE (%)": bl["ridge_inter_cycle"]["MAE"],  "RMSE (%)": bl["ridge_inter_cycle"]["RMSE"],  "R²": bl["ridge_inter_cycle"]["R2"]},
        {"Approche": "LSTM intra-cycle (sujet)",         "Fenêtre": "3 bins", "Features":  5, "MAE (%)": bl["lstm_intra_cycle"]["MAE"],   "RMSE (%)": bl["lstm_intra_cycle"]["RMSE"],   "R²": bl["lstm_intra_cycle"]["R2"]},
        {"Approche": "BiLSTM Run #7",                    "Fenêtre": "10 cyc", "Features": 11, "MAE (%)": bl["bilstm_run7"]["MAE"],        "RMSE (%)": bl["bilstm_run7"]["RMSE"],        "R²": bl["bilstm_run7"]["R2"]},
        {"Approche": f"★ Run #{RUN_ID} — {ARCH_DESC}",  "Fenêtre": "10 cyc", "Features": N_FEATURES, "MAE (%)": round(mae_global, 3), "RMSE (%)": round(rmse_global, 3), "R²": round(r2_global, 4)},
    ])
    st.dataframe(comp, width="stretch", hide_index=True)
    st.caption(f"★ Run #{RUN_ID} — modèle chargé dans ce dashboard ({CFG['model']['checkpoint']})")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Analyse par batterie
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyse par batterie":

    st.subheader("Résidus par batterie test")
    st.caption("Évaluation sur batteries **inconnues du modèle** (split battery-wise strict)")

    rows = []
    for bat in TEST_BATTERIES:
        mask = bat_tags_arr == bat
        rows.append({
            "Batterie":    bat,
            "N fenêtres":  int(mask.sum()),
            "MAE (%)":     round(float(np.abs(residuals[mask]).mean()), 3),
            "Biais (%)":   round(float(residuals[mask].mean()), 3),
            "Max err (%)": round(float(np.abs(residuals[mask]).max()), 3),
            "R²":          round(float(r2_score(y_test[mask], y_pred[mask])), 4),
        })
    df_bat = pd.DataFrame(rows)

    # Mise en forme : rouge si MAE > 1.5× globale
    def highlight_mae(val):
        return "background-color: #f4c2c2" if isinstance(val, float) and val > mae_global * 1.5 else ""

    st.dataframe(
        df_bat.style.map(highlight_mae, subset=["MAE (%)"]),
        width="stretch", hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df_bat, x="Batterie", y="MAE (%)", color="MAE (%)",
                     color_continuous_scale="RdYlGn_r",
                     title="MAE par batterie test")
        fig.add_hline(y=mae_global, line_dash="dash", line_color="red",
                      annotation_text=f"MAE globale = {mae_global:.3f}%",
                      annotation_position="top left")
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        colors = ["#e07b6a" if v < 0 else "#5b9bd5" for v in df_bat["Biais (%)"]]
        fig = go.Figure(go.Bar(x=df_bat["Batterie"], y=df_bat["Biais (%)"],
                               marker_color=colors,
                               hovertemplate="%{x}<br>Biais: %{y:.3f}%"))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            title="Biais par batterie<br><sup>Négatif = surprédiction | Positif = sous-prédiction</sup>",
            yaxis_title="Biais (%)", height=360
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Trajectoire SoH — Réel vs Prédit")
    bat_sel = st.selectbox("Choisir une batterie test", TEST_BATTERIES)
    mask    = bat_tags_arr == bat_sel
    y_t     = y_test[mask]
    y_p     = y_pred[mask]
    err     = np.abs(y_p - y_t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_t.tolist(), mode="lines+markers", name="SoH réel",
                             line=dict(color="#5b9bd5", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(y=y_p.tolist(), mode="lines+markers", name="SoH prédit",
                             line=dict(color="#e07b6a", width=2, dash="dash"), marker=dict(size=4)))
    fig.add_trace(go.Scatter(y=err.tolist(), mode="lines", name="|Erreur|",
                             line=dict(color="gray", width=1, dash="dot"),
                             fill="tozeroy", fillcolor="rgba(128,128,128,0.1)"))
    fig.update_layout(
        xaxis_title="Fenêtre (index)", yaxis_title="SoH (%)",
        title=(f"{bat_sel} — MAE = {err.mean():.3f}%  |  "
               f"R² = {r2_score(y_t, y_p):.4f}  |  "
               f"Max err = {err.max():.3f}%"),
        height=420, legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Courbes d'apprentissage
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Courbes d'apprentissage":

    if log_df is None:
        st.warning(f"Fichier de log introuvable : `{LOG_CSV}`")
        st.info(f"Lancez d'abord : `python src/training/train_lstm.py --run {RUN_ID}`")
        st.stop()

    best_ep   = int(log_df.loc[log_df["val_loss"].idxmin(), "epoch"])
    n_epochs  = int(log_df["epoch"].max())
    best_vloss = log_df["val_loss"].min()
    gap_final  = float(log_df["val_loss"].iloc[-1] - log_df["train_loss"].iloc[-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Époques",        n_epochs)
    c2.metric("Meilleure époque", best_ep)
    c3.metric("Meilleure val_loss", f"{best_vloss:.6f}")
    c4.metric("Gap val−train final", f"{gap_final:.4f}",
              help="Proche de 0 = pas d'overfitting")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["train_mae"],
                                 name="Train MAE", line=dict(color="#5b9bd5", width=2)))
        fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["val_mae"],
                                 name="Val MAE", line=dict(color="#e07b6a", width=2)))
        fig.add_vline(x=best_ep, line_dash="dot", line_color="green",
                      annotation_text=f"Époque #{best_ep}", annotation_position="top right")
        fig.add_hline(y=BASELINE["MAE"], line_dash="dash", line_color="gray",
                      annotation_text=f"Baseline {BASELINE['MAE']} %",
                      annotation_position="bottom right")
        fig.update_layout(title="MAE train / validation", xaxis_title="Époque",
                          yaxis_title="MAE (%)", height=380)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["train_loss"],
                                 name="Train loss", line=dict(color="#5b9bd5", width=2)))
        fig.add_trace(go.Scatter(x=log_df["epoch"], y=log_df["val_loss"],
                                 name="Val loss", line=dict(color="#e07b6a", width=2)))
        fig.add_vline(x=best_ep, line_dash="dot", line_color="green",
                      annotation_text=f"Époque #{best_ep}", annotation_position="top right")
        loss_name = CFG["training"]["loss"]
        fig.update_layout(title=f"{loss_name} — train / validation", xaxis_title="Époque",
                          yaxis_title="Loss", height=380)
        st.plotly_chart(fig, width="stretch")

    fig = go.Figure(go.Scatter(x=log_df["epoch"], y=log_df["lr"],
                               mode="lines", line=dict(color="purple", width=2),
                               name="Learning rate"))
    fig.add_vline(x=best_ep, line_dash="dot", line_color="green",
                  annotation_text=f"Époque #{best_ep}", annotation_position="top right")
    fig.update_layout(title=f"Learning Rate schedule ({CFG['training']['lr_scheduler']})",
                      xaxis_title="Époque", yaxis_title="LR",
                      yaxis_type="log", height=300)
    st.plotly_chart(fig, width="stretch")

    if gap_final > 5:
        st.warning(f"Gap val−train = {gap_final:.2f} → risque d'overfitting")
    else:
        st.success(f"Gap val−train = {gap_final:.4f} → pas d'overfitting détecté")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Prédiction interactive
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prédiction interactive":

    st.subheader(f"Prédire le SoH à partir d'une fenêtre de {WINDOW_SIZE} cycles")

    # ── Choix batterie avec distinction train/test ────────────────────────────
    col_cfg, col_viz = st.columns([1, 2])

    with col_cfg:
        split_choice = st.radio(
            "Type de batterie",
            ["✅ Test (inconnue du modèle)", "⚠️ Train (vue à l'entraînement)"],
            index=0,
            help=(
                "Test : évaluation réelle — le modèle n'a jamais vu ces batteries.\n\n"
                "Train : démonstration seulement — les résultats sont optimistes car "
                "le modèle a appris sur ces données."
            ),
        )
        is_train = split_choice.startswith("⚠️")
        battery_list = TRAIN_BATTERIES if is_train else TEST_BATTERIES

        if is_train:
            st.warning(
                "⚠️ Batterie d'entraînement sélectionnée.\n\n"
                "Le modèle a été entraîné sur ces données — les performances "
                "sont **non représentatives** de la généralisation réelle."
            )

        bat_choice = st.selectbox("Batterie", battery_list)

        bat_cyc = (cycle_df[cycle_df["battery_id"] == bat_choice]
                   .sort_values("cycle_number").reset_index(drop=True))
        n_cycles = len(bat_cyc)

        if n_cycles < WINDOW_SIZE + 1:
            st.error(f"Pas assez de cycles ({n_cycles} < {WINDOW_SIZE + 1})")
            st.stop()

        max_start = n_cycles - WINDOW_SIZE - 1
        start_idx = st.slider("Cycle de départ (index)", 0, max_start, max_start // 2,
                              help=f"La fenêtre couvre les {WINDOW_SIZE} cycles à partir de cet index.")

        soh_pred, soh_true, window_cycles, target_cycle, elec_w, soh_w, soh_prev = \
            predict_window(cycle_df, bat_choice, start_idx, scaler, model, ridge)

        err = abs(soh_pred - soh_true)

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("SoH prédit", f"{soh_pred:.2f} %")
        m2.metric("SoH réel",   f"{soh_true:.2f} %",
                  delta=f"Erreur : {err:.2f} %", delta_color="inverse")
        st.caption(
            f"Fenêtre : cycles {window_cycles[0]} → {window_cycles[-1]}  |  "
            f"Cible : cycle {target_cycle}"
        )

        # Indicateur fin de vie
        if soh_pred < 80:
            st.error(f"🔴 SoH prédit < 80% — batterie en **fin de vie**")
        elif soh_pred < 85:
            st.warning(f"🟠 SoH prédit entre 80–85% — **surveiller**")
        else:
            st.success(f"🟢 SoH prédit ≥ 85% — batterie en **bon état**")

        # Détail features
        with st.expander("Détail des features de la fenêtre"):
            df_win = pd.DataFrame(elec_w, columns=ELEC_FEATURES)
            df_win.insert(0, "Cycle", window_cycles)
            df_win[ALL_FEATURES[-1]] = soh_prev
            df_win["SoH réel"] = soh_w
            st.dataframe(df_win.round(4), width="stretch", hide_index=True)

    with col_viz:
        # Trajectoire complète de la batterie
        bat_full = cycle_df[cycle_df["battery_id"] == bat_choice].sort_values("cycle_number")

        fig = go.Figure()

        # Courbe SoH complète
        fig.add_trace(go.Scatter(
            x=bat_full["cycle_number"].tolist(), y=bat_full["SoH"].tolist(),
            mode="lines+markers", name="SoH réel",
            line=dict(color="#5b9bd5", width=2), marker=dict(size=4),
        ))

        # Zone fenêtre d'entrée
        fig.add_vrect(
            x0=window_cycles[0], x1=window_cycles[-1],
            fillcolor="orange", opacity=0.15, line_width=0,
            annotation_text="Fenêtre d'entrée",
            annotation_position="top left",
        )

        # Point prédit (étoile)
        fig.add_trace(go.Scatter(
            x=[target_cycle], y=[soh_pred],
            mode="markers", name=f"SoH prédit ({soh_pred:.2f}%)",
            marker=dict(color="tomato", size=16, symbol="star",
                        line=dict(color="darkred", width=1)),
        ))

        # Point réel cible (cercle ouvert)
        fig.add_trace(go.Scatter(
            x=[target_cycle], y=[soh_true],
            mode="markers", name=f"SoH réel ({soh_true:.2f}%)",
            marker=dict(color="green", size=12, symbol="circle-open",
                        line=dict(color="green", width=2.5)),
        ))

        # Ligne d'erreur entre prédit et réel
        fig.add_shape(
            type="line",
            x0=target_cycle, x1=target_cycle,
            y0=min(soh_pred, soh_true), y1=max(soh_pred, soh_true),
            line=dict(color="gray", width=1.5, dash="dot"),
        )

        # Seuil fin de vie
        fig.add_hline(y=80, line_dash="dash", line_color="red", line_width=1,
                      annotation_text="Seuil fin de vie 80%",
                      annotation_position="bottom right")

        fig.update_layout(
            title=f"Trajectoire SoH — {bat_choice}"
                  + (" ⚠️ (batterie d'entraînement)" if is_train else " ✅ (batterie test)"),
            xaxis_title="Numéro de cycle",
            yaxis_title="SoH (%)",
            height=480,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, width="stretch")
