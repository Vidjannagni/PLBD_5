"""
test_tuning.py
===============
Tests unitaires du module ``src.models_training.tuning`` (diagnostic
immédiat — optimisation des hyperparamètres par Grid Search et tuning
du seuil de décision).

Pour garder la suite rapide, les tests qui exercent réellement
``GridSearchCV`` (run_grid_search, main) utilisent un registre de
modèles "miniature" injecté par monkeypatch, plutôt que les grilles
réelles (qui comptent jusqu'à 324 combinaisons par modèle).

Organisation
------------
    1. _build_model_registry
    2. run_grid_search
    3. tune_threshold
    4. save_results
    5. Visualisations (plot_tuning_summary, plot_param_heatmap)
    6. print_tuning_report
    7. main() — pipeline complet (intégration)

Lancer la suite
----------------
    pytest tests/test_tuning.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.tuning import (
    main,
    plot_param_heatmap,
    plot_tuning_summary,
    print_tuning_report,
    run_grid_search,
    save_results,
    tune_threshold,
)
import src.tuning as tn


# ===========================================================================
# Registre miniature — accélère drastiquement les tests de GridSearchCV
# ===========================================================================

def _tiny_registry():
    return {
        "logreg": {
            "label": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=500),
            "param_grid": {"C": [0.1, 1.0]},
        },
        "rf": {
            "label": "Random Forest",
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {"max_depth": [None, 3], "n_estimators": [10, 20]},
        },
    }


@pytest.fixture
def tiny_registry(monkeypatch):
    monkeypatch.setattr(tn, "_build_model_registry", _tiny_registry)
    monkeypatch.setitem(tn.GRID_SEARCH, "cv_splits", 3)  # accélère encore
    return _tiny_registry()


# ===========================================================================
# 1. _build_model_registry
# ===========================================================================

class TestBuildModelRegistry:

    def test_core_models_always_present(self):
        registry = tn._build_model_registry()
        for key in ["logreg", "svm", "rf", "et", "gb"]:
            assert key in registry
            assert "estimator" in registry[key]
            assert "param_grid" in registry[key]
            assert "label" in registry[key]

    def test_param_grids_are_non_empty(self):
        registry = tn._build_model_registry()
        for key, spec in registry.items():
            grid = spec["param_grid"]
            assert grid, f"Grille vide pour {key}"


# ===========================================================================
# 2. run_grid_search
# ===========================================================================

class TestRunGridSearch:

    def test_returns_best_params_with_expected_structure(self, clf_split, tiny_registry):
        X_train, X_test, y_train, y_test, _ = clf_split
        best_params, full_report = run_grid_search(X_train, y_train)

        assert set(best_params.keys()) == {"logreg", "rf"}
        for key, info in best_params.items():
            assert set(["label", "params", "gmean_cv", "fit_time_s", "threshold"]) <= set(info.keys())
            assert 0.0 <= info["gmean_cv"] <= 1.0
            assert info["threshold"] == 0.5  # pas encore affiné à ce stade
            assert info["fit_time_s"] >= 0.0

    def test_full_report_contains_all_combinations(self, clf_split, tiny_registry):
        X_train, X_test, y_train, y_test, _ = clf_split
        _, full_report = run_grid_search(X_train, y_train)

        assert "model_key" in full_report.columns
        assert "mean_test_score" in full_report.columns
        # logreg : 2 combinaisons, rf : 2×2 = 4 combinaisons
        assert (full_report["model_key"] == "logreg").sum() == 2
        assert (full_report["model_key"] == "rf").sum() == 4

    def test_model_keys_filters_registry(self, clf_split, tiny_registry):
        X_train, X_test, y_train, y_test, _ = clf_split
        best_params, _ = run_grid_search(X_train, y_train, model_keys=["logreg"])
        assert set(best_params.keys()) == {"logreg"}

    def test_unknown_model_key_raises_valueerror(self, clf_split, tiny_registry):
        X_train, X_test, y_train, y_test, _ = clf_split
        with pytest.raises(ValueError, match="Modèles inconnus"):
            run_grid_search(X_train, y_train, model_keys=["modele_fantome"])

    def test_best_params_picks_a_grid_value(self, clf_split, tiny_registry):
        X_train, X_test, y_train, y_test, _ = clf_split
        best_params, _ = run_grid_search(X_train, y_train, model_keys=["logreg"])
        assert best_params["logreg"]["params"]["C"] in [0.1, 1.0]


# ===========================================================================
# 3. tune_threshold
# ===========================================================================

class TestTuneThreshold:

    class _FakeEstimator:
        """Estimateur factice : predict_proba renvoie des probabilités fixes,
        indépendamment de X, pour rendre le test entièrement déterministe."""

        def __init__(self, proba_class1):
            self._proba1 = np.asarray(proba_class1)

        def predict_proba(self, X):
            return np.column_stack([1 - self._proba1, self._proba1])

    def test_picks_threshold_maximizing_fbeta(self, monkeypatch):
        # Calcul à la main (voir docstring du fichier de tests) :
        # f1(0.3) = 0.727  >  f1(0.5) = 0.667  =  f1(0.7) = 0.667
        # -> le seuil optimal attendu est 0.3
        monkeypatch.setitem(tn.THRESHOLD_TUNING, "thresholds", np.array([0.3, 0.5, 0.7]))
        monkeypatch.setitem(tn.THRESHOLD_TUNING, "beta", 1)

        proba = [0.9, 0.8, 0.6, 0.4, 0.65, 0.55, 0.3, 0.1]
        y_val = [1, 1, 1, 1, 0, 0, 0, 0]
        est = self._FakeEstimator(proba)

        best_thresh = tune_threshold(est, X_val=np.zeros((8, 1)), y_val=y_val)
        assert best_thresh == pytest.approx(0.3)

    def test_default_grid_returns_value_in_expected_range(self, clf_split):
        from sklearn.linear_model import LogisticRegression as LR

        X_train, X_test, y_train, y_test, _ = clf_split
        est = LR().fit(X_train, y_train)

        best_thresh = tune_threshold(est, X_test, y_test)
        assert 0.20 <= best_thresh <= 0.80

    def test_returns_python_float(self, clf_split):
        from sklearn.linear_model import LogisticRegression as LR

        X_train, X_test, y_train, y_test, _ = clf_split
        est = LR().fit(X_train, y_train)
        best_thresh = tune_threshold(est, X_test, y_test)
        assert isinstance(best_thresh, float)


# ===========================================================================
# 4. save_results
# ===========================================================================

class TestSaveResults:

    def test_writes_json_and_csv(self, tmp_outputs):
        from config import PATHS

        best_params = {
            "logreg": {"label": "Logistic Regression", "params": {"C": 1.0},
                       "gmean_cv": 0.81, "fit_time_s": 1.2, "threshold": 0.45},
        }
        full_report = pd.DataFrame({"model_key": ["logreg", "logreg"], "mean_test_score": [0.8, 0.81]})

        save_results(best_params, full_report)

        assert PATHS["best_params"].exists()
        assert PATHS["tuning_report"].exists()

        loaded = json.loads(PATHS["best_params"].read_text(encoding="utf-8"))
        assert loaded["logreg"]["params"]["C"] == 1.0

        loaded_csv = pd.read_csv(PATHS["tuning_report"])
        assert len(loaded_csv) == 2


# ===========================================================================
# 5. VISUALISATIONS
# ===========================================================================

class TestVisualisations:

    def test_plot_tuning_summary_creates_file(self, tmp_outputs):
        from config import PATHS

        best_params = {
            "logreg": {"label": "Logistic Regression", "gmean_cv": 0.80, "fit_time_s": 1.1},
            "rf":     {"label": "Random Forest",        "gmean_cv": 0.85, "fit_time_s": 4.3},
        }
        plot_tuning_summary(best_params)
        out = PATHS["figures_tuning"] / "tuning_summary.png"
        assert out.exists() and out.stat().st_size > 0

    def test_plot_param_heatmap_creates_file_with_two_params(self, tmp_outputs):
        from config import PATHS

        full_report = pd.DataFrame({
            "model_key":          ["rf"] * 4,
            "param_max_depth":    [None, None, 3, 3],
            "param_n_estimators": [10, 20, 10, 20],
            "mean_test_score":    [0.7, 0.75, 0.72, 0.78],
        })
        plot_param_heatmap(full_report, "rf")
        out = PATHS["figures_tuning"] / "heatmap_rf.png"
        assert out.exists() and out.stat().st_size > 0

    def test_plot_param_heatmap_skips_when_model_absent(self, tmp_outputs):
        from config import PATHS

        full_report = pd.DataFrame({
            "model_key":       ["rf"],
            "param_max_depth": [3],
            "param_n_estimators": [10],
            "mean_test_score": [0.7],
        })
        plot_param_heatmap(full_report, "logreg")  # absent du report
        out = PATHS["figures_tuning"] / "heatmap_logreg.png"
        assert not out.exists()

    def test_plot_param_heatmap_skips_when_fewer_than_two_params(self, tmp_outputs):
        from config import PATHS

        full_report = pd.DataFrame({
            "model_key": ["logreg", "logreg"],
            "param_C":   [0.1, 1.0],
            "mean_test_score": [0.7, 0.75],
        })
        plot_param_heatmap(full_report, "logreg")  # un seul param_*
        out = PATHS["figures_tuning"] / "heatmap_logreg.png"
        assert not out.exists()

    def test_plot_param_heatmap_uses_own_params_not_other_models(self, tmp_outputs, monkeypatch):
        """
        Test de régression pour le bug trouvé : avec l'ordre réel de
        concaténation (logreg -> svm -> rf), les colonnes param_C et
        param_kernel (héritées de logreg/svm, NaN pour rf) ne doivent
        PAS être choisies à la place de param_max_depth/param_n_estimators.
        """
        captured_pivots = []
        monkeypatch.setattr(
            tn.sns, "heatmap",
            lambda data, **kwargs: captured_pivots.append(data) or tn.plt.gca()
        )

        df_logreg = pd.DataFrame({"model_key": ["logreg"] * 2, "param_C": [0.1, 1.0], "mean_test_score": [0.7, 0.75]})
        df_svm = pd.DataFrame({
            "model_key": ["svm"] * 2, "param_kernel": ["rbf", "linear"],
            "param_C": [1, 10], "param_gamma": ["scale", "auto"], "mean_test_score": [0.6, 0.65],
        })
        df_rf = pd.DataFrame({
            "model_key": ["rf"] * 4,
            "param_max_depth": [None, None, 3, 3],
            "param_n_estimators": [10, 20, 10, 20],
            "mean_test_score": [0.7, 0.75, 0.72, 0.78],
        })
        full_report = pd.concat([df_logreg, df_svm, df_rf], ignore_index=True)

        plot_param_heatmap(full_report, "rf")

        assert len(captured_pivots) == 1
        pivot = captured_pivots[0]
        assert pivot.index.name == "param_max_depth"
        assert pivot.columns.name == "param_n_estimators"

        from config import PATHS

        full_report = pd.DataFrame({
            "model_key": ["logreg", "logreg"],
            "param_C":   [0.1, 1.0],
            "mean_test_score": [0.7, 0.75],
        })
        plot_param_heatmap(full_report, "logreg")  # un seul param_*
        out = PATHS["figures_tuning"] / "heatmap_logreg.png"
        assert not out.exists()


# ===========================================================================
# 6. print_tuning_report
# ===========================================================================

class TestPrintTuningReport:

    def test_prints_model_labels_and_scores(self, capsys):
        best_params = {
            "logreg": {"label": "Logistic Regression", "params": {"C": 1.0},
                       "gmean_cv": 0.80, "fit_time_s": 1.5, "threshold": 0.5},
            "rf":     {"label": "Random Forest", "params": {"n_estimators": 100},
                       "gmean_cv": 0.88, "fit_time_s": 3.2, "threshold": 0.5},
        }
        print_tuning_report(best_params)
        captured = capsys.readouterr().out

        assert "Random Forest" in captured
        assert "Logistic Regression" in captured
        assert "0.88" in captured  # meilleur score affiché

    def test_best_model_ranked_first(self, capsys):
        best_params = {
            "a": {"label": "A", "params": {}, "gmean_cv": 0.60, "fit_time_s": 1.0, "threshold": 0.5},
            "b": {"label": "B", "params": {}, "gmean_cv": 0.90, "fit_time_s": 1.0, "threshold": 0.5},
        }
        print_tuning_report(best_params)
        captured = capsys.readouterr().out
        # "B" (meilleur score) doit apparaître avant "A" dans la sortie
        assert captured.index("[B]") < captured.index("[A]")


# ===========================================================================
# 7. main() — pipeline complet (intégration)
# ===========================================================================

class TestMainPipeline:

    def test_full_pipeline_runs_and_produces_artifacts(
        self, raw_csv_path, tmp_outputs, tiny_registry
    ):
        from config import PATHS

        best_params = main(data_path=raw_csv_path, model_keys=None)

        assert set(best_params.keys()) == {"logreg", "rf"}
        for info in best_params.values():
            assert 0.20 <= info["threshold"] <= 0.80  # affiné par tune_threshold

        assert PATHS["best_params"].exists()
        assert PATHS["tuning_report"].exists()
        assert (PATHS["figures_tuning"] / "tuning_summary.png").exists()
        # rf a 2 hyperparamètres -> heatmap générée ; logreg n'en a qu'un -> pas de heatmap
        assert (PATHS["figures_tuning"] / "heatmap_rf.png").exists()
        assert not (PATHS["figures_tuning"] / "heatmap_logreg.png").exists()  # 1 seul hyperparamètre dans le registre miniature

    def test_threshold_tuning_disabled_keeps_default_half(
        self, raw_csv_path, tmp_outputs, tiny_registry, monkeypatch
    ):
        monkeypatch.setitem(tn.THRESHOLD_TUNING, "enabled", False)
        best_params = main(data_path=raw_csv_path, model_keys=None)
        for info in best_params.values():
            assert info["threshold"] == 0.5

    def test_model_keys_subset_propagated_through_main(
        self, raw_csv_path, tmp_outputs, tiny_registry
    ):
        best_params = main(data_path=raw_csv_path, model_keys=["logreg"])
        assert set(best_params.keys()) == {"logreg"}

    def test_smotetomek_does_not_crash_when_enabled(
        self, raw_csv_path, tmp_outputs, tiny_registry, monkeypatch
    ):
        pytest.importorskip("imblearn")
        monkeypatch.setitem(tn.REBALANCING, "enabled", True)
        best_params = main(data_path=raw_csv_path, model_keys=["logreg"])
        assert "logreg" in best_params

    def test_smotetomek_gracefully_skipped_when_imblearn_missing(
        self, raw_csv_path, tmp_outputs, tiny_registry, monkeypatch
    ):
        import sys
        monkeypatch.setitem(tn.REBALANCING, "enabled", True)
        monkeypatch.setitem(sys.modules, "imblearn.combine", None)
        monkeypatch.setitem(sys.modules, "imblearn.over_sampling", None)

        # Le pipeline doit continuer sans planter (ImportError attrapé + warning)
        best_params = main(data_path=raw_csv_path, model_keys=["logreg"])
        assert "logreg" in best_params
