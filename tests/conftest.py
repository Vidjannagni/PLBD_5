"""
conftest.py
===========
Configuration pytest globale + fixtures partagées pour les tests du
projet Water Potability.

Rôle de ce fichier
------------------
1. Garantir que la racine du projet est sur ``sys.path``, pour que
   ``from src.models_training.data_processing import ...`` fonctionne
   quel que soit le répertoire depuis lequel ``pytest`` est lancé
   (terminal, IDE, CI/CD).
2. Fournir des fixtures réutilisables (DataFrames synthétiques) pour
   éviter de dupliquer la génération de données dans chaque fichier
   de test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Résolution du sys.path : insère la racine du dépôt (parent de tests/)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# train_model.py et tuning.py font "from config import ..." (import bare, sans
# préfixe de package). Ça ne fonctionne que si le dossier qui contient config.py
# est lui-même sur sys.path. Si ton config.py vit ailleurs que
# src/models_training/, ajuste ce chemin.
SRC_MODULE_DIR = ROOT_DIR / "src"
if SRC_MODULE_DIR.exists() and str(SRC_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_MODULE_DIR))


# ===========================================================================
# FIXTURES — Données synthétiques
# ===========================================================================

@pytest.fixture
def rng():
    """Générateur aléatoire numpy avec graine fixe (reproductibilité)."""
    return np.random.default_rng(42)


@pytest.fixture
def raw_dataframe_clean(rng) -> pd.DataFrame:
    """
    DataFrame brut "propre" : mêmes colonnes que le CSV Kaggle, sans NaN
    ni outlier, valeurs dans les bornes physiques. Sert de point de
    référence pour les tests qui ne portent pas sur le nettoyage.

    Convention Kaggle d'origine : Potability = 1 -> potable, 0 -> non potable
    (avant la relabélisation faite par ``raw_data_processing``).
    """
    n = 200
    df = pd.DataFrame({
        "ph":           rng.normal(7.0, 0.8, n).clip(0, 14),
        "Solids":       rng.normal(600, 900, n).clip(0, 1000),
        "Conductivity": rng.normal(420, 80, n).clip(0, 1500),
        "Turbidity":    rng.normal(4.0, 1.5, n).clip(0, 100),
        "Potability":   rng.integers(0, 2, n),
    })
    return df


@pytest.fixture
def raw_dataframe_dirty(raw_dataframe_clean) -> pd.DataFrame:
    """
    Variante "sale" du DataFrame propre : injecte des NaN dans ``ph``
    et des outliers extrêmes (statistiques + hors bornes physiques)
    dans les autres features, pour tester le pipeline de nettoyage.
    """
    df = raw_dataframe_clean.copy()

    # NaN dans ph (10 % des lignes)
    nan_idx = df.index[:20]
    df.loc[nan_idx, "ph"] = np.nan

    # Outliers statistiques (au-delà de Q3 + 1.5*IQR) mais physiquement valides
    df.loc[df.index[20], "Conductivity"] = 1400.0
    df.loc[df.index[21], "Turbidity"]    = 95.0

    # Outliers hors bornes physiques pour les eaux de boissons (doivent être tronqués par enforce_physical)
    df.loc[df.index[22], "ph"]     = 16.0     # > 14 -> physiquement impossible
    df.loc[df.index[23], "Solids"] = 1500  # > 1000

    return df


@pytest.fixture
def raw_csv_path(tmp_path, raw_dataframe_clean) -> Path:
    """Écrit ``raw_dataframe_clean`` sur disque et retourne son chemin."""
    path = tmp_path / "water_potability.csv"
    raw_dataframe_clean.to_csv(path, index=False)
    return path


# ===========================================================================
# FIXTURES — train_model.py
# ===========================================================================

@pytest.fixture
def clf_Xy():
    """
    Dataset de classification synthétique (4 features nommées comme
    :data:`FEATURES`, cible binaire légèrement déséquilibrée comme le
    dataset réel ~61/39). Suffisamment séparable pour que les modèles
    obtiennent des scores ROC-AUC significatifs dans les tests.
    """
    from sklearn.datasets import make_classification

    from src.data_processing import FEATURES, TARGET

    X_arr, y_arr = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        weights=[0.4, 0.6],
        flip_y=0.03,
        class_sep=1.2,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=FEATURES)
    y = pd.Series(y_arr, name=TARGET)
    return X, y


@pytest.fixture
def clf_split(clf_Xy):
    """Split train/test standardisé, prêt à être injecté dans train_model.py."""
    from src.data_processing import preprocess_for_ml

    X, y = clf_Xy
    return preprocess_for_ml(X, y, test_size=0.25, random_state=42)


@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    """
    Redirige tous les chemins relatifs de :data:`config.PATHS` vers un
    répertoire temporaire (en changeant le cwd du process) et crée
    l'arborescence de sortie attendue. Évite de polluer le vrai dossier
    ``outputs/`` du projet pendant les tests.
    """
    from src.config import ensure_dirs

    monkeypatch.chdir(tmp_path)
    ensure_dirs()
    return tmp_path
