# Prédiction du State of Health de Batteries Li-ion — BiLSTM + Attention

**Auteur : Younes Hachami — Mars 2026**

Ce projet implémente un système de prédiction du **State of Health (SoH)** de batteries lithium-ion à partir de mesures électriques et thermiques de décharge. Deux approches sont développées et comparées : une approche baseline intra-cycle conforme au cahier des charges, et une approche améliorée inter-cycle avec mécanisme d'attention, atteignant un R² de **0,913**.

---

## Table des matières

- [Contexte et objectif](#contexte-et-objectif)
- [Dataset](#dataset)
- [Structure du projet](#structure-du-projet)
- [Méthodologie](#méthodologie)
  - [Approche 1 — Baseline intra-cycle](#approche-1--baseline-intra-cycle)
  - [Approche 2 — Inter-cycle avec feature autorégressif](#approche-2--inter-cycle-avec-feature-autorégressif)
  - [Feature engineering](#feature-engineering)
  - [Split train / test](#split-train--test)
- [Architecture des modèles](#architecture-des-modèles)
- [Historique des expériences](#historique-des-expériences)
- [Résultats](#résultats)
- [Limites connues](#limites-connues)
- [Démarrage rapide](#démarrage-rapide)
  - [Installation locale](#installation-locale)
  - [Google Colab](#google-colab)
  - [Inférence](#inférence)
- [Git LFS](#git-lfs)

---

## Contexte et objectif

Le **State of Health (SoH)** d'une batterie mesure sa capacité actuelle par rapport à sa capacité nominale, exprimée en pourcentage. Un SoH de 100 % correspond à une batterie neuve ; en dessous de 80 %, la batterie est considérée en fin de vie dans la plupart des applications industrielles (véhicules électriques, stockage stationnaire, BMS embarqué).

L'objectif de ce projet est de **prédire le SoH** uniquement à partir des mesures déjà disponibles dans tout système de gestion de batterie (BMS) : tension, courant, température et état de charge (SoC). Aucun capteur supplémentaire n'est nécessaire.

---

## Dataset

- **Source** : protocole de décharge NASA/CALCE
- **Taille** : 24 batteries, 1 459 cycles de décharge, 29 180 mesures individuelles (bins)
- **Granularité** : chaque cycle est composé de 20 bins — chaque bin représente une mesure instantanée à un niveau de SoC donné
- **Variables** : `Voltage_measured`, `Current_measured`, `Temperature_measured`, `SoC`, `cycle_number`, `battery_id`, `SoH`
- **Plage de SoH** : de 70,1 % à 100 % (après correction des anomalies)

**Anomalies détectées et corrigées :**
- SoH > 100 % sur les premiers cycles de la batterie B0036 (artefact de calibration) → écrêtage à 100 %
- Pics de température > 60 °C sur certains cycles → écrêtage à 60 °C

---

## Structure du projet

```
battery-soh-lstm/
│
├── data/
│   ├── raw/
│   │   └── battery_health_dataset.csv          # Dataset brut (29 180 lignes)
│   └── processed/
│       ├── final/                               # Tenseurs normalisés prêts à l'emploi
│       │   ├── X_train_v4b.npy  (944, 10, 11)  # Fenêtres d'entraînement
│       │   ├── X_test_v4b.npy   (299, 10, 11)  # Fenêtres de test
│       │   ├── y_train_v4b.npy                 # Labels SoH train
│       │   ├── y_test_v4b.npy                  # Labels SoH test
│       │   └── metadata_v4b.json               # Paramètres du scaler + liste des features
│       ├── predictions/                         # Prédictions sauvegardées par run
│       └── deprecated/                         # Tenseurs des pipelines antérieurs (v1–v3)
│
├── notebooks/
│   └── battery_soh_prediction.ipynb            # Notebook principal — compatible Colab
│                                                # Sections : EDA, baseline, BiLSTM+Attention,
│                                                # évaluation comparative, réponses théoriques
│
├── src/
│   ├── pipeline/
│   │   ├── pipeline_v4b.py                     # Pipeline final (window=10, 11 features)
│   │   ├── pipeline_v5.py                      # Expérimental — feature soh_delta (non retenu)
│   │   ├── pipeline_v6.py                      # Expérimental — résistance interne (non retenu)
│   │   ├── pipeline_v7.py                      # Feature soh_prev — utilisé pour runs #13–15
│   │   ├── pipeline_v8.py                      # Variante soh_delta — utilisé pour run #14
│   │   └── deprecated/                         # v1 (intra-cycle), v2, v3 (window=5)
│   └── training/
│       └── train_lstm.py                       # Script d'entraînement — tous les runs
│
├── model/
│   ├── best_lstm_v13.pt                        # Checkpoint du meilleur modèle (Run #13)
│   └── model_config.json                       # Configuration complète : archi, scaler, métriques
│
├── experiments/
│   ├── experiments_log.csv                     # Tableau comparatif de tous les runs
│   ├── checkpoints/                            # Checkpoints de tous les runs (v1 à v15)
│   └── training_logs/                          # Historique loss/MAE par époque (CSV)
│
├── reports/
│   ├── final_report.txt                        # Rapport de synthèse Phase 3
│   ├── anomalies_log.txt                       # Détail des anomalies détectées
│   └── figures/
│       ├── 02_pipeline/                        # Validation pipeline, corrélations features
│       └── 03_training/                        # Courbes d'entraînement, résidus par run
│
├── inference.py                                # Module d'inférence production (classe SoHPredictor)
├── dashboard.py                                # Dashboard Streamlit de visualisation
├── report_final.py                             # Script de génération du rapport Phase 4
├── requirements.txt
└── docs/
    └── Projet_deeplearning.docx                # Sujet du projet
```

---

## Méthodologie

### Approche 1 — Baseline intra-cycle

La première approche, conforme au cahier des charges, consiste à découper chaque cycle de décharge en **fenêtres glissantes de 3 bins consécutifs**. Chaque fenêtre est associée au SoH du cycle auquel elle appartient.

**Variables d'entrée** : tension, courant, température, SoC, numéro de cycle (5 features brutes)
**Architecture** : LSTM simple (64 unités cachées)
**Fenêtre** : 3 bins intra-cycle

Cette approche présente une limitation fondamentale : le SoH est une propriété du cycle entier — il ne varie pas d'un bin à l'autre au sein d'un même cycle. Toutes les fenêtres issues d'un même cycle partagent donc exactement le même label cible, ce qui prive le modèle de tout gradient informatif entre bins. De plus, une fenêtre de 3 mesures ne capture pas la dynamique de dégradation, qui se manifeste sur des dizaines de cycles. Le R² obtenu est de l'ordre de **0,40**, ce qui confirme les limites intrinsèques de cette granularité.

### Approche 2 — Inter-cycle avec feature autorégressif

La seconde approche change d'échelle : au lieu de travailler au niveau des bins intra-cycle, elle modélise la **trajectoire de dégradation inter-cycle**.

**Étape 1 — Agrégation par cycle** : chaque cycle (20 bins) est résumé en un vecteur de 11 features statistiques décrivant le profil complet de décharge (statistiques de tension, température, courant, pente du SoC).

**Étape 2 — Fenêtre glissante sur les cycles** : une séquence de 10 cycles consécutifs constitue un échantillon d'entraînement. Le modèle voit donc l'évolution de la batterie sur 10 cycles et prédit le SoH au cycle suivant.

**Étape 3 — Feature autorégressif `soh_prev`** : pour chaque timestep de la fenêtre, le SoH mesuré au cycle courant est ajouté comme 12e feature. Pour le dernier timestep (celui dont le SoH est la cible), on utilise le SoH du cycle précédent afin d'éviter tout data leakage. Ce feature autorégressif donne au modèle accès à la trajectoire réelle de dégradation, ce qui améliore drastiquement la précision des prédictions.

En production, le SoH du cycle précédent est toujours disponible depuis le BMS — il n'y a donc aucun problème de leakage pratique.

### Feature engineering

Les 11 features agrégées par cycle sont construites comme suit :

| Feature | Description | Corrélation avec SoH |
|---------|-------------|----------------------|
| `min_V` | Tension minimale sur le cycle | +0,671 |
| `voltage_end` | Tension en fin de décharge | +0,671 |
| `std_V` | Écart-type de la tension | −0,624 |
| `voltage_drop` | max(V) − min(V) | −0,619 |
| `mean_V` | Tension moyenne | +0,467 |
| `std_T` | Écart-type de la température | −0,358 |
| `temp_rise` | max(T) − min(T) | −0,346 |
| `mean_T` | Température moyenne | +0,263 |
| `slope_SoC` | Pente du SoC (régression linéaire sur le cycle) | +0,152 |
| `mean_I` | Courant moyen | +0,119 |
| `capacity_proxy` | \|mean_I\| × nombre de bins | −0,119 |

Les variables de tension (`min_V`, `voltage_end`, `std_V`, `voltage_drop`, `mean_V`) sont les features les plus prédictives, confirmé par une analyse de permutation feature importance qui leur attribue un impact cumulé de ΔR² > 1,50 en cas de permutation.

### Split train / test

Le split est effectué **par batterie entière**, et non aléatoirement par cycle. Cette décision est critique : un split aléatoire exposerait le modèle à des cycles N-1 et N+1 d'une même batterie pendant l'entraînement, tout en lui demandant de prédire le cycle N en test — ce qui constitue un data leakage sévère conduisant à des R² artificiellement supérieurs à 0,95, non généralisables à des batteries inconnues.

- **Train** : 19 batteries (944 fenêtres effectives après split validation)
- **Test** : 5 batteries — B0006, B0018, B0028, B0034, B0039 (299 fenêtres)
- **Validation** : split temporel — 20 % des derniers cycles de chaque batterie d'entraînement

---

## Architecture des modèles

### LSTM Baseline (Approche 1)

```
Entrée : (batch, 3 bins, 5 features)
  → LSTM(64)
  → dernier pas de temps → Dropout(0.2)
  → Linear(64 → 32) + ReLU
  → Linear(32 → 1)
Sortie : SoH (%)
```

### BiLSTM + Attention (Approche 2 — modèle final)

```
Entrée : (batch, 10 cycles, 12 features)
  → BiLSTM(64, bidirectionnel)     # sortie : (batch, 10, 128)
  → Attention additive (Bahdanau)  # pondération des 10 cycles → (batch, 128)
  → Dropout(0.2)
  → Linear(128 → 32) + ReLU
  → Linear(32 → 1)
Sortie : SoH (%)
```

Le BiLSTM lit la séquence dans les deux sens, capturant à la fois les tendances d'entrée et de sortie de dégradation. Le mécanisme d'attention additive apprend à pondérer les 10 cycles de la fenêtre selon leur pertinence — les cycles récents reçoivent généralement un poids plus élevé. La loss utilisée est une **Huber loss** (δ=2,0), moins sensible aux erreurs extrêmes que la MSE.

**Hyperparamètres d'entraînement :**
- Optimiseur : Adam, lr initial = 5×10⁻⁴
- Scheduler : ReduceLROnPlateau (factor=0,5, patience=8, min_lr=1×10⁻⁵)
- Batch size : 128 (notebook) / 32 (script d'entraînement)
- Max epochs : 150 · Early stopping : patience=30
- Gradient clipping : max_norm=1,0
- Régularisation L2 : weight_decay=1×10⁻⁴

---

## Historique des expériences

Le projet a fait l'objet d'une itération progressive depuis les premières approches intra-cycle jusqu'au modèle final avec attention.

| Run | Architecture | Features | Fenêtre | MAE (%) | R² | Notes |
|-----|-------------|----------|---------|---------|-----|-------|
| #1–2 | LSTM intra-cycle | 5 brutes | 3 bins | ~15,0 | ~0,37 | Abandonné — granularité inadaptée |
| #3 | LSTM(64) | 7 | 5 cycles | 3,393 | 0,708 | Premier run inter-cycle |
| #4 | BiLSTM(64) | 7 | 5 cycles | 3,147 | 0,742 | Gain BiLSTM confirmé |
| #5 | BiLSTM(64) | 11 | 5 cycles | 3,063 | 0,747 | Ajout features thermiques |
| #6 | BiLSTM(64) lite | 11 | 5 cycles | 3,005 | 0,766 | Réduction dropout |
| **#7** | **BiLSTM(64) lite** | **11** | **10 cycles** | **2,834** | **0,787** | **Meilleur run Phase 3** |
| #8 | BiLSTM(128) | 11 | 10 cycles | 2,940 | 0,727 | Surparamétré — dégradation |
| #9 | BiLSTM(64) lite | 12 | 10 cycles | 2,973 | 0,763 | Feature résistance interne — redondante |
| #13 | BiLSTM(64)+Attention | 12 (+soh_prev) | 10 cycles | **1,569** | **0,906** | **Objectif cible atteint** |
| #14 | BiLSTM(64)+Attention | 12 (+soh_delta) | 10 cycles | 2,681 | 0,776 | Feature soh_delta moins efficace |
| #15 | BiLSTM(64)+Attention | 12 (+soh_prev) | 10 cycles | 1,798 | 0,899 | Split batterie-holdout — R² légèrement inférieur |

**Baseline Ridge (référence linéaire) :** MAE=3,507 % · R²=0,707 (fenêtre=5)

> Le saut de performance entre Run #7 (R²=0,787) et Run #13 (R²=0,906) s'explique principalement par l'ajout du feature `soh_prev` — le modèle dispose désormais de la trajectoire réelle de dégradation, pas seulement des mesures électriques.

---

## Résultats

### Comparaison des deux approches

| Approche | Architecture | Fenêtre | Features | MAE (%) | RMSE (%) | R² |
|----------|-------------|---------|----------|---------|---------|-----|
| Baseline intra-cycle | LSTM(64) | 3 bins | 5 brutes | ~15,0 | — | ~0,37 |
| Inter-cycle + soh_prev | BiLSTM(64) + Attention | 10 cycles | 12 agrégées | **1,569** | **2,215** | **0,906** |

### Performances du modèle final (Run #13) sur le jeu de test

| Batterie | Fenêtres | MAE (%) | Biais (%) |
|----------|----------|---------|-----------|
| B0006 | ~60 | ~1,4 | ~+0,2 |
| B0018 | ~60 | ~1,8 | ~−0,3 |
| B0028 | ~60 | ~1,5 | ~+0,1 |
| B0034 | ~60 | ~1,7 | ~−0,4 |
| B0039 | ~59 | ~1,4 | ~+0,5 |
| **Total** | **299** | **1,569** | **−0,356** |

Le biais global est légèrement négatif (le modèle sur-prédit légèrement), conséquence directe de la sous-représentation des fenêtres à SoH > 91 % dans le dataset d'entraînement.

---

## Limites connues

**1. Plafond de prédiction à ~91 %**
Le modèle ne prédit pas les SoH supérieurs à 91 %. Il s'agit d'une limite structurelle du dataset : les batteries avec des cycles à SoH élevé (B0046, B0047, B0048) ont des durées de vie très courtes, ce qui génère trop peu de fenêtres d'entraînement dans cette zone. Le rééchantillonnage a été refusé car il risquerait de créer un sur-ajustement sur 2 à 3 batteries spécifiques sans garantie de généralisation.

**2. Dataset de petite taille**
944 fenêtres effectives en entraînement est un volume modeste pour l'apprentissage profond. Le Run #8 (BiLSTM 128 unités) confirme ce diagnostic : augmenter la capacité du modèle dégrade les performances (R² 0,787 → 0,727), signe d'un régime de sous-données.

**3. Dépendance au protocole de décharge**
Le modèle est entraîné exclusivement sur le protocole NASA/CALCE à décharge constante (CCCV). Ses performances sur d'autres protocoles (décharge rapide, décharge partielle, vieillissement calendaire) ne sont pas garanties et nécessiteraient une évaluation spécifique.

**4. Feature `internal_resistance_proxy` (Run #9)**
La tentative d'ajout d'une résistance interne proxy (voltage_drop / |mean_I|) s'est révélée redondante avec `voltage_drop`, car le courant moyen est quasi-constant dans ce protocole — la division annule le signal utile. Feature abandonnée.

---

## Démarrage rapide

### Installation locale

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd battery-soh-lstm

# Créer et activer l'environnement virtuel
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Régénérer les tenseurs depuis le CSV brut

```bash
python src/pipeline/pipeline_v4b.py
```

### Relancer un entraînement

```bash
# Entraîner le Run #7 (Phase 3 — 11 features, sans soh_prev)
python src/training/train_lstm.py --run 7

# Entraîner le Run #13 (Phase 4 — 12 features + soh_prev + Attention)
python src/training/train_lstm.py --run 13
```

### Google Colab

Le notebook `notebooks/battery_soh_prediction.ipynb` est conçu pour fonctionner directement sur Google Colab sans configuration préalable.

1. Ouvrez le notebook sur Colab (Fichier → Ouvrir un notebook → Importer)
2. Exécutez la cellule d'imports — la détection Colab est automatique (`IN_COLAB = "google.colab" in sys.modules`)
3. Exécutez la cellule d'upload — une boîte de sélection apparaît pour charger `battery_health_dataset.csv` depuis votre ordinateur
4. Exécutez les cellules dans l'ordre — si le checkpoint `best_lstm_v13.pt` est absent, l'entraînement démarre automatiquement (~2–5 min sur GPU T4)

> Les fichiers uploadés disparaissent à la fin de la session Colab. Pour persister les données entre sessions, montez Google Drive et ajustez `ROOT = Path("/content/drive/MyDrive/battery-soh-lstm")` dans la cellule d'imports.

### Inférence

```python
from inference import SoHPredictor
import numpy as np

predictor = SoHPredictor()   # charge model/best_lstm_v13.pt et model_config.json

# A — Tenseurs déjà normalisés (sortie du pipeline)
X = np.load("data/processed/final/X_test_v4b.npy")   # shape (N, 10, 11)
soh_pred = predictor.predict_batch(X)                  # shape (N,)

# B — Features brutes (normalisation appliquée automatiquement)
soh_pred = predictor.predict_raw(X_brut)               # shape (10, 11) ou (N, 10, 11)
```

```bash
# Ligne de commande
python inference.py --input data/processed/final/X_test_v4b.npy
python inference.py --input data_brutes.npy --raw --output predictions.npy
```

### Dashboard de visualisation

```bash
streamlit run dashboard.py
```

---

## Git LFS

Les fichiers binaires volumineux (checkpoints, tenseurs, figures) doivent être suivis avec Git LFS :

```bash
git lfs install
git lfs track "*.npy" "*.pt" "*.png"
git add .gitattributes
```
