"""
test_config.py
===============
Tests unitaires du module ``config`` (diagnostic immédiat — source
unique de vérité pour les chemins, métriques et seuils de sélection).

Organisation
------------
    1. Reproductibilité (RANDOM_STATE)
    2. DATA (chemins d'entrée)
    3. PATHS + ensure_dirs()
    4. FEATURES / TARGET / CLASS_LABELS / POSITIVE_CLASS
    5. Scorers personnalisés (GMEAN_SCORER, FBETA_SCORER)
    6. METRICS (cohérence des poids du score composite)
    7. REBALANCING / THRESHOLD_TUNING / CV / GRID_SEARCH / MODEL_SELECTION / PLOT
    8. MODEL_REGISTRY

Lancer la suite
----------------
    pytest tests/test_config.py -v
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import src.config as config


# ===========================================================================
# 1. REPRODUCTIBILITÉ
# ===========================================================================

class TestRandomState:

    def test_is_fixed_integer(self):
        assert isinstance(config.RANDOM_STATE, int)
        assert config.RANDOM_STATE == 42


# ===========================================================================
# 2. DATA
# ===========================================================================

class TestData:

    def test_raw_path_built_from_dir_and_filename(self):
        assert config.DATA["raw_path"] == config.DATA["raw_dir"] / config.DATA["raw_filename"]

    def test_processed_path_built_from_dir_and_filename(self):
        assert config.DATA["processed_path"] == (
            config.DATA["processed_dir"] / config.DATA["processed_filename"]
        )

    def test_test_and_val_size_are_valid_fractions(self):
        assert 0.0 < config.DATA["test_size"] < 1.0
        assert 0.0 < config.DATA["val_size"] < 1.0

    def test_test_and_val_size_leave_room_for_train(self):
        # Le train set ne doit jamais être vide ou négatif
        assert config.DATA["test_size"] + config.DATA["val_size"] < 1.0


# ===========================================================================
# 3. PATHS + ensure_dirs()
# ===========================================================================

class TestEnsureDirs:

    def test_creates_all_directory_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config.ensure_dirs()

        for key, path in config.PATHS.items():
            if path.suffix == "":
                assert path.exists() and path.is_dir(), f"PATHS['{key}'] non créé"

        assert config.DATA["raw_dir"].exists()
        assert config.DATA["processed_dir"].exists()

    def test_does_not_create_file_paths_as_directories(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config.ensure_dirs()

        # best_params.json, tuning_report.csv, etc. ont un suffixe -> ce sont
        # des fichiers, ensure_dirs ne doit PAS créer de dossier à leur nom.
        file_paths = [p for p in config.PATHS.values() if p.suffix != ""]
        assert file_paths, "Aucun chemin de fichier trouvé dans PATHS — le test perd son sens"
        for path in file_paths:
            assert not path.exists()  # le fichier lui-même n'est pas créé
            assert not path.is_dir()

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config.ensure_dirs()
        config.ensure_dirs()  # ne doit lever aucune exception au second appel
        assert config.PATHS["models"].exists()

    def test_parent_of_file_paths_is_created(self, tmp_path, monkeypatch):
        # Le dossier parent d'un chemin de fichier (ex: outputs/reports pour
        # best_params.json) doit exister, même si le fichier lui n'existe pas.
        monkeypatch.chdir(tmp_path)
        config.ensure_dirs()
        assert config.PATHS["best_params"].parent.exists()


# ===========================================================================
# 4. FEATURES / TARGET / CLASS_LABELS / POSITIVE_CLASS
# ===========================================================================

class TestFeaturesAndTarget:

    def test_features_list(self):
        assert config.FEATURES == ["ph", "Solids", "Conductivity", "Turbidity"]

    def test_target_name(self):
        assert config.TARGET == "Potability"

    def test_class_labels_match_positive_class_semantics(self):
        # POSITIVE_CLASS doit correspondre à "Non potable" (cas dangereux à détecter)
        assert config.CLASS_LABELS[config.POSITIVE_CLASS] == "Non potable"
        assert config.CLASS_LABELS[1 - config.POSITIVE_CLASS] == "Potable"


# ===========================================================================
# 5. SCORERS PERSONNALISÉS
# ===========================================================================

class TestGMeanScore:

    def test_perfect_prediction_returns_one(self):
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        assert config._gmean_score(y_true, y_pred) == pytest.approx(1.0)

    def test_predicting_all_positive_returns_zero(self):
        # recall classe 1 = 1.0, recall classe 0 = 0.0 -> gmean = 0
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]
        assert config._gmean_score(y_true, y_pred) == pytest.approx(0.0)

    def test_predicting_all_negative_returns_zero(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        assert config._gmean_score(y_true, y_pred) == pytest.approx(0.0)

    def test_known_partial_recalls(self):
        # recall_1 = 4/5 = 0.8 ; recall_0 = 3/4 = 0.75 -> sqrt(0.6)
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 1]
        expected = np.sqrt(0.8 * 0.75)
        assert config._gmean_score(y_true, y_pred) == pytest.approx(expected)

    def test_symmetric_in_class_balance(self):
        # Inverser les deux classes (0<->1) ne doit pas changer le gmean
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 1]
        y_true_inv = [1 - v for v in y_true]
        y_pred_inv = [1 - v for v in y_pred]
        assert config._gmean_score(y_true, y_pred) == pytest.approx(
            config._gmean_score(y_true_inv, y_pred_inv)
        )


class TestScorers:

    def test_gmean_scorer_matches_gmean_score_function(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        model = LogisticRegression().fit(X_train, y_train)

        score_via_scorer = config.GMEAN_SCORER(model, X_test, y_test)
        score_via_function = config._gmean_score(y_test, model.predict(X_test))

        assert score_via_scorer == pytest.approx(score_via_function)

    def test_fbeta_scorer_is_callable_sklearn_scorer(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        model = LogisticRegression().fit(X_train, y_train)
        score = config.FBETA_SCORER(model, X_test, y_test)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# 6. METRICS
# ===========================================================================

class TestMetrics:

    def test_composite_weights_sum_to_one(self):
        total = sum(config.METRICS["composite_weights"].values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_primary_metric_is_in_scoring_dict(self):
        assert config.METRICS["primary"] in config.METRICS["scoring"]

    def test_composite_weight_keys_are_scorable_metrics(self):
        # Chaque métrique pondérée doit être calculable (présente dans scoring
        # ou dérivable, comme pr_auc/mcc qui sont calculés à part dans train_model.py)
        scoring_keys = set(config.METRICS["scoring"].keys())
        weighted_keys = set(config.METRICS["composite_weights"].keys())
        # gmean, fbeta, roc_auc doivent au moins être dans scoring
        assert {"gmean", "fbeta", "roc_auc"}.issubset(scoring_keys)
        assert weighted_keys.issubset(scoring_keys | {"pr_auc", "mcc"})

    def test_secondary_metrics_list_non_empty(self):
        assert len(config.METRICS["secondary"]) > 0


# ===========================================================================
# 7. AUTRES SECTIONS DE CONFIGURATION
# ===========================================================================

class TestRebalancing:

    def test_disabled_by_default(self):
        assert config.REBALANCING["enabled"] is False

    def test_smote_k_neighbors_is_positive(self):
        assert config.REBALANCING["smote_k_neighbors"] > 0


class TestThresholdTuning:

    def test_default_thresholds_grid_matches_docstring(self):
        # "thresholds": None signifie une grille auto np.arange(0.20, 0.80, 0.01)
        assert config.THRESHOLD_TUNING["thresholds"] is None

    def test_optimize_metric_is_valid_choice(self):
        assert config.THRESHOLD_TUNING["optimize_metric"] in {"fbeta", "f1", "recall"}


class TestCV:

    def test_n_splits_is_reasonable(self):
        assert 2 <= config.CV["n_splits"] <= 20

    def test_shuffle_enabled(self):
        assert config.CV["shuffle"] is True


class TestGridSearch:

    def test_cv_splits_smaller_than_outer_cv(self):
        # Le GridSearch interne (tuning) doit utiliser moins de folds que la
        # validation croisée externe (train_model), sans quoi le coût explose.
        assert config.GRID_SEARCH["cv_splits"] <= config.CV["n_splits"]

    def test_scoring_is_gmean_scorer(self):
        assert config.GRID_SEARCH["scoring"] is config.GMEAN_SCORER


class TestModelSelection:

    def test_top_n_is_positive_integer(self):
        assert isinstance(config.MODEL_SELECTION["top_n"], int)
        assert config.MODEL_SELECTION["top_n"] > 0

    def test_min_composite_and_min_roc_auc_are_valid_fractions(self):
        assert 0.0 <= config.MODEL_SELECTION["min_composite"] <= 1.0
        assert 0.0 <= config.MODEL_SELECTION["min_roc_auc"] <= 1.0

    def test_model_format_is_dot_joblib(self):
        assert config.MODEL_SELECTION["model_format"] == ".joblib"


class TestPlot:

    def test_palette_has_enough_colors_for_all_base_models(self):
        # _base_estimators() expose au moins 5 modèles (logreg, svm, rf, et, gb)
        assert len(config.PLOT["palette"]) >= 5

    def test_palette_colors_are_valid_hex(self):
        for color in config.PLOT["palette"]:
            assert color.startswith("#") and len(color) == 7

    def test_dpi_is_positive(self):
        assert config.PLOT["dpi"] > 0


# ===========================================================================
# 8. MODEL_REGISTRY
# ===========================================================================

class TestModelRegistry:

    def test_starts_empty(self):
        # Le registre est rempli dynamiquement par train_model.py — au chargement
        # du module config, il doit être vide.
        assert config.MODEL_REGISTRY == {}
