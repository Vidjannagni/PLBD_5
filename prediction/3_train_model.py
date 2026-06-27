"""
3_train_model.py
=================
Entraînement du LSTM de prévision avec :
    - Split 70 / 15 / 15 (train / val / test)
    - Early stopping (patience=10) sur la val_loss
    - Seed reproductible (numpy + torch)
    - Sauvegarde du meilleur modèle (val_loss minimale)

Usage
-----
    python prediction/3_train_model.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import FEATURES, HORIZON, LSTMModel, N_FEATURES, WINDOW_SIZE

# ── Reproductibilité ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
(BASE / "models").mkdir(parents=True, exist_ok=True)

# ── Hyperparamètres ──────────────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 0.001
PATIENCE    = 10        # early stopping : arrêt si val_loss ne baisse pas
HIDDEN_SIZE = 64
NUM_LAYERS  = 2

# ── Chargement des données ───────────────────────────────────────────────────
X = np.load(BASE / "data/X.npy")
y = np.load(BASE / "data/y.npy")

# ── Split temporel 70 / 15 / 15 ─────────────────────────────────────────────
# Jamais de mélange aléatoire sur une série temporelle (data leakage)
n          = len(X)
split_val  = int(n * 0.70)
split_test = int(n * 0.85)

X_train, y_train = X[:split_val],            y[:split_val]
X_val,   y_val   = X[split_val:split_test],  y[split_val:split_test]
X_test,  y_test  = X[split_test:],           y[split_test:]

print(f"Split : train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

# ── Tenseurs ─────────────────────────────────────────────────────────────────
device = torch.device("cpu")   # CPU pour compatibilité Raspberry Pi

def to_tensor(arr):
    return torch.FloatTensor(arr).to(device)

X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)
X_val_t,   y_val_t   = to_tensor(X_val),   to_tensor(y_val)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False,
)

# ── Modèle ───────────────────────────────────────────────────────────────────
model     = LSTMModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Entraînement avec early stopping ─────────────────────────────────────────
print(f"\n🚀 Entraînement en cours... (max {EPOCHS} epochs, patience={PATIENCE})")
print(f"   Features : {FEATURES}")
print(f"   Fenêtre  : {WINDOW_SIZE}h → Horizon : {HORIZON}h\n")

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    # — Phase entraînement —
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred  = model(X_batch)
        loss  = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # — Phase validation —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred      = model(X_batch)
            val_loss += criterion(pred, y_batch).item()
    val_loss /= len(val_loader)

    # — Affichage —
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} — train_loss: {train_loss:.5f}  val_loss: {val_loss:.5f}")

    # — Early stopping —
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), BASE / "models/lstm_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹  Early stopping à l'epoch {epoch} (patience={PATIENCE})")
            break

print(f"\n✅ Meilleur val_loss : {best_val_loss:.5f}")
print(f"   Modèle sauvegardé → {BASE / 'models/lstm_model.pth'}")

# ── Sauvegarder le jeu de test pour 4_evaluate.py ───────────────────────────
np.save(BASE / "data/X_test.npy", X_test)
np.save(BASE / "data/y_test.npy", y_test)
print(f"   Jeu de test sauvegardé ({len(X_test)} séquences)")
