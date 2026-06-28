"""
2_preprocess.py
================
Prétraitement de la série temporelle pour le LSTM :
    - Normalisation : RobustScaler (résistant aux pics de pollution)
    - Séquences      : fenêtre 24h → prévision 24h
    - Sauvegarde     : X.npy, y.npy, scaler.joblib (format joblib, cohérent
                       avec le reste du projet)

Pourquoi RobustScaler plutôt que MinMaxScaler ?
-------------------------------------------------
La génération de données inclut des événements extrêmes (crues, pollutions).
MinMaxScaler écrase les valeurs normales vers 0 si un seul pic est très haut.
RobustScaler utilise la médiane et l'IQR — insensible aux outliers.
Cohérent avec le scaler du module de diagnostic (src/).

Fenêtre de 24h
--------------
24 mesures passées capturent un cycle diurne complet, ce qui est
informatif pour prédire les 24 heures suivantes.

Usage
-----
    python prediction/2_preprocess.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from model import FEATURES, HORIZON, WINDOW_SIZE

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

# ── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv(BASE / "data/dataset.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Vérification alignement des colonnes
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(
        f"Colonnes manquantes dans dataset.csv : {missing}\n"
        "Vérifiez que 1_generate_data.py a été lancé avec la version actuelle."
    )

# ── Normalisation ─────────────────────────────────────────────────────────────
scaler   = RobustScaler()
arr      = df[FEATURES].values                      # (n_points, N_FEATURES)
arr_scaled = scaler.fit_transform(arr)

# ── Séquences supervisées ────────────────────────────────────────────────────
# X(t) = mesures de t-WINDOW_SIZE à t-1  →  y(t) = mesures de t à t+HORIZON-1
X, y = [], []
n = len(arr_scaled)
for i in range(n - WINDOW_SIZE - HORIZON):
    X.append(arr_scaled[i           : i + WINDOW_SIZE])
    y.append(arr_scaled[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON])

X = np.array(X, dtype=np.float32)   # (n_seq, WINDOW_SIZE, N_FEATURES)
y = np.array(y, dtype=np.float32)   # (n_seq, HORIZON,     N_FEATURES)

# ── Sauvegarde ───────────────────────────────────────────────────────────────
np.save(BASE / "data/X.npy", X)
np.save(BASE / "data/y.npy", y)
joblib.dump(scaler, BASE / "data/scaler.joblib")

print("✅ Prétraitement terminé")
print("   Scaler         : RobustScaler (médiane + IQR)")
print(f"   Fenêtre entrée : {WINDOW_SIZE} heures")
print(f"   Horizon sortie : {HORIZON} heures")
print(f"   X shape : {X.shape}  →  (nb_séquences, fenêtre, nb_capteurs)")
print(f"   y shape : {y.shape}  →  (nb_séquences, horizon, nb_capteurs)")
print(f"   Features : {FEATURES}")
