"""
test_train_model.py
====================
Tests unitaires du module ``src.models_training.train_model``
(diagnostic immédiat — entraînement, validation croisée et évaluation
finale).

Organisation
------------
    1. ThresholdClassifier (wrapper modèle + seuil)
    2. _base_estimators / build_models
    3. cross_validate_models
    4. select_top_models
    5. train_and_evaluate
    6. save_models
    7. Visualisations (plot_*)
    8. write_summary_report
    9. main() — pipeline complet (intégration)

Lancer la suite
----------------
    pytest tests/test_train_model.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.train_model import (
    ThresholdClassifier,
    build_models,
    cross_validate_models,
    save_models,
    select_top_models,
    train_and_evaluate,
    write_summary_report,
    plot_confusion_matrices,
    plot_cv_comparison,
    plot_feature_importance,
    plot_roc_curves,
    main,
)
import src.train_model as tm


# ===========================================================================
# 1. ThresholdClassifier
# ===========================================================================

class TestThresholdClassifier:

    def test_fit_sets_classes(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        clf = ThresholdClassifier(LogisticRegression(), threshold=0.5)
        clf.fit(X_train, y_train)
        assert list(clf.classes_) == [0, 1]

    def test_predict_proba_delegates_to_estimator(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        est = LogisticRegression().fit(X_train, y_train)
        wrapped = ThresholdClassifier(est)
        wrapped.classes_ = est.classes_
        np.testing.assert_array_equal(
            wrapped.predict_proba(X_test), est.predict_proba(X_test)
        )

    def test_predict_uses_threshold_not_default_half(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        est = LogisticRegression().fit(X_train, y_train)
        proba1 = est.predict_proba(X_test)[:, 1]

        # Seuil extrême haut : (presque) tout le monde classé 0
        clf_high = ThresholdClassifier(est, threshold=0.999)
        clf_high.classes_ = est.classes_
        pred_high = clf_high.predict(X_test)

        # Seuil extrême bas : (presque) tout le monde classé 1
        clf_low = ThresholdClassifier(est, threshold=0.001)
        clf_low.classes_ = est.classes_
        pred_low = clf_low.predict(X_test)

        assert pred_high.sum() <= pred_low.sum()
        np.testing.assert_array_equal(pred_high, (proba1 >= 0.999).astype(int))
        np.testing.assert_array_equal(pred_low, (proba1 >= 0.001).astype(int))

    def test_set_params_updates_threshold_and_estimator(self):
        clf = ThresholdClassifier(LogisticRegression(C=1.0), threshold=0.5)
        clf.set_params(threshold=0.3, C=10.0)
        assert clf.threshold == 0.3
        assert clf.estimator.get_params()["C"] == 10.0

    def test_get_params_exposes_estimator_prefixed_keys(self):
        clf = ThresholdClassifier(LogisticRegression(C=2.0), threshold=0.42)
        params = clf.get_params(deep=True)
        assert params["threshold"] == 0.42
        assert params["estimator__C"] == 2.0

    def test_get_params_shallow_excludes_estimator_prefixed_keys(self):
        clf = ThresholdClassifier(LogisticRegression(C=2.0), threshold=0.42)
        params = clf.get_params(deep=False)
        assert "estimator__C" not in params
        assert params["threshold"] == 0.42

    def test_feature_importances_delegates_when_available(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        est = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
        wrapped = ThresholdClassifier(est)
        np.testing.assert_array_equal(wrapped.feature_importances_, est.feature_importances_)

    def test_feature_importances_none_when_unavailable(self):
        wrapped = ThresholdClassifier(LogisticRegression())
        assert wrapped.feature_importances_ is None

    def test_coef_delegates_when_available(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        est = LogisticRegression().fit(X_train, y_train)
        wrapped = ThresholdClassifier(est)
        np.testing.assert_array_equal(wrapped.coef_, est.coef_)

    def test_coef_none_when_unavailable(self, clf_split):
        X_train, X_test, y_train, y_test, _ = clf_split
        est = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
        wrapped = ThresholdClassifier(est)
        assert wrapped.coef_ is None

    def test_repr_contains_class_name_and_threshold(self):
        clf = ThresholdClassifier(LogisticRegression(), threshold=0.314)
        assert "LogisticRegression" in repr(clf)
        assert "0.314" in repr(clf)

    def test_survives_joblib_roundtrip(self, clf_split, tmp_path):
        import joblib

        X_train, X_test, y_train, y_test, _ = clf_split
        clf = ThresholdClassifier(LogisticRegression(), threshold=0.37)
        clf.fit(X_train, y_train)

        path = tmp_path / "model.joblib"
        joblib.dump(clf, path)
        reloaded = joblib.load(path)

        assert reloaded.threshold == 0.37
        np.testing.assert_array_equal(reloaded.predict(X_test), clf.predict(X_test))


# ===========================================================================
# 2. _base_estimators / build_models
# ===========================================================================

class TestBuildModels:

    def test_base_estimators_always_include_core_models(self):
        base = tm._base_estimators()
        for key in ["logreg", "svm", "rf", "et", "gb"]:
            assert key in base

    def test_build_models_without_best_params_returns_base(self):
        models = build_models(best_params_path=None)
        base = tm._base_estimators()
        assert set(models.keys()) == set(base.keys())

    def test_build_models_with_missing_file_returns_base(self, tmp_path):
        models = build_models(best_params_path=tmp_path / "absent.json")
        assert set(models.keys()) == set(tm._base_estimators().keys())

    def test_build_models_applies_valid_params(self, tmp_path):
        best_params = {
            "logreg": {"label": "Logistic Regression", "params": {"C": 7.5}},
        }
        path = tmp_path / "best_params.json"
        path.write_text(json.dumps(best_params))

        models = build_models(best_params_path=path)
        assert models["logreg"].get_params()["C"] == 7.5

    def test_build_models_ignores_unknown_model_keys(self, tmp_path):
        best_params = {"modele_qui_n_existe_pas": {"params": {"C": 1.0}}}
        path = tmp_path / "best_params.json"
        path.write_text(json.dumps(best_params))

        models = build_models(best_params_path=path)
        assert set(models.keys()) == set(tm._base_estimators().keys())

    def test_build_models_falls_back_on_invalid_param(self, tmp_path):
        # "n_inexistant" n'est pas un hyperparamètre valide de LogisticRegression
        best_params = {"logreg": {"params": {"n_inexistant": 123}}}
        path = tmp_path / "best_params.json"
        path.write_text(json.dumps(best_params))

        models = build_models(best_params_path=path)
        # Ne doit pas lever d'exception : le modèle par défaut est conservé
        assert "logreg" in models
        assert isinstance(models["logreg"], LogisticRegression)


# ===========================================================================
# 3. cross_validate_models
# ===========================================================================

class TestCrossValidateModels:

    def test_returns_ranked_dataframe_with_expected_columns(self, clf_split, monkeypatch):
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)  # accélère le test

        models = {
            "logreg": LogisticRegression(),
            "rf": RandomForestClassifier(n_estimators=20, random_state=42),
        }
        cv_results = cross_validate_models(models, X_train, y_train)

        assert set(cv_results["model_key"]) == {"logreg", "rf"}
        assert list(cv_results["rank"]) == [1, 2] or list(cv_results["rank"]) == [1, 2]
        assert cv_results["roc_auc_mean"].is_monotonic_decreasing
        for col in ["roc_auc_mean", "roc_auc_std", "f1_mean", "gmean_mean"]:
            assert col in cv_results.columns

    def test_failing_model_is_excluded_without_crashing(self, clf_split, monkeypatch):
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)

        class BrokenEstimator:
            """Lève systématiquement une exception au fit — simule un modèle cassé."""

            def fit(self, X, y):
                raise ValueError("modèle volontairement cassé pour le test")

            def get_params(self, deep=True):
                return {}

            def set_params(self, **kw):
                return self

        models = {
            "logreg": LogisticRegression(),
            "broken": BrokenEstimator(),
        }
        cv_results = cross_validate_models(models, X_train, y_train)

        assert "broken" not in set(cv_results["model_key"])
        assert "logreg" in set(cv_results["model_key"])

    def test_rank_starts_at_one(self, clf_split, monkeypatch):
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)
        models = {"logreg": LogisticRegression()}
        cv_results = cross_validate_models(models, X_train, y_train)
        assert cv_results["rank"].iloc[0] == 1


# ===========================================================================
# 4. select_top_models
# ===========================================================================

class TestSelectTopModels:

    @staticmethod
    def _fake_cv_results(rows):
        df = pd.DataFrame(rows)
        df = df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def test_filters_below_min_roc_auc(self, monkeypatch):
        monkeypatch.setitem(tm.MODEL_SELECTION, "min_roc_auc", 0.70)
        cv_results = self._fake_cv_results([
            {"model_key": "good", "roc_auc_mean": 0.85},
            {"model_key": "bad",  "roc_auc_mean": 0.55},
        ])
        models = {"good": "modele_good", "bad": "modele_bad"}

        top = select_top_models(cv_results, models, top_n=2)
        assert set(top.keys()) == {"good"}

    def test_fallback_to_top_n_when_all_below_threshold(self, monkeypatch):
        monkeypatch.setitem(tm.MODEL_SELECTION, "min_roc_auc", 0.95)
        cv_results = self._fake_cv_results([
            {"model_key": "a", "roc_auc_mean": 0.60},
            {"model_key": "b", "roc_auc_mean": 0.55},
        ])
        models = {"a": "modele_a", "b": "modele_b"}

        top = select_top_models(cv_results, models, top_n=2)
        # Aucun modèle ne passe le seuil -> fallback sur le top-n brut
        assert set(top.keys()) == {"a", "b"}

    def test_respects_top_n_limit(self, monkeypatch):
        monkeypatch.setitem(tm.MODEL_SELECTION, "min_roc_auc", 0.0)
        cv_results = self._fake_cv_results([
            {"model_key": "a", "roc_auc_mean": 0.90},
            {"model_key": "b", "roc_auc_mean": 0.85},
            {"model_key": "c", "roc_auc_mean": 0.80},
        ])
        models = {"a": "A", "b": "B", "c": "C"}

        top = select_top_models(cv_results, models, top_n=2)
        assert set(top.keys()) == {"a", "b"}


# ===========================================================================
# 5. train_and_evaluate
# ===========================================================================

class TestTrainAndEvaluate:

    def test_eval_report_columns_and_ranges(self, clf_split, monkeypatch):
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)

        top_models = {"logreg": LogisticRegression()}
        cv_results = cross_validate_models(top_models, X_train, y_train)

        eval_report, fitted_models = train_and_evaluate(
            top_models, X_train, X_test, y_train, y_test, cv_results,
            best_params_thresholds={"logreg": 0.4},
        )

        assert "logreg" in fitted_models
        assert isinstance(fitted_models["logreg"], ThresholdClassifier)
        assert fitted_models["logreg"].threshold == 0.4

        row = eval_report.iloc[0]
        for col in ["roc_auc_test", "pr_auc_test", "gmean_test", "f1_test", "accuracy_test"]:
            assert 0.0 <= row[col] <= 1.0
        assert row["threshold"] == pytest.approx(0.4)

    def test_default_threshold_is_half_when_not_provided(self, clf_split, monkeypatch):
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)

        top_models = {"logreg": LogisticRegression()}
        cv_results = cross_validate_models(top_models, X_train, y_train)

        eval_report, fitted_models = train_and_evaluate(
            top_models, X_train, X_test, y_train, y_test, cv_results,
            best_params_thresholds=None,
        )
        assert fitted_models["logreg"].threshold == 0.5
        assert eval_report.iloc[0]["threshold"] == pytest.approx(0.5)

    def test_predict_consistency_with_threshold(self, clf_split, monkeypatch):
        """Vérifie que y_pred utilisé pour les métriques test correspond
        exactement au seuil intégré au ThresholdClassifier (pas de seuil 0.5
        caché ailleurs)."""
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)

        top_models = {"logreg": LogisticRegression()}
        cv_results = cross_validate_models(top_models, X_train, y_train)
        eval_report, fitted_models = train_and_evaluate(
            top_models, X_train, X_test, y_train, y_test, cv_results,
            best_params_thresholds={"logreg": 0.3},
        )
        from sklearn.metrics import recall_score
        model = fitted_models["logreg"]
        y_pred_manual = (model.predict_proba(X_test)[:, 1] >= 0.3).astype(int)
        expected_recall = recall_score(y_test, y_pred_manual, zero_division=0)
        assert eval_report.iloc[0]["recall_test"] == pytest.approx(expected_recall, abs=1e-5)


# ===========================================================================
# 6. save_models
# ===========================================================================

class TestSaveModels:

    def test_saves_scaler_and_ranked_models(self, clf_split, tmp_outputs):
        from config import PATHS

        X_train, X_test, y_train, y_test, scaler = clf_split
        fitted = {
            "logreg": ThresholdClassifier(LogisticRegression().fit(X_train, y_train)),
            "rf":     ThresholdClassifier(RandomForestClassifier(n_estimators=10).fit(X_train, y_train)),
        }
        eval_report = pd.DataFrame({"model_key": ["rf", "logreg"]})  # rf classé #1

        save_models(fitted, scaler, eval_report)

        assert (PATHS["models"] / "scaler.joblib").exists()
        assert (PATHS["models"] / "model_1_rf.joblib").exists()
        assert (PATHS["models"] / "model_2_logreg.joblib").exists()

    def test_skips_models_absent_from_fitted_dict(self, clf_split, tmp_outputs):
        from config import PATHS

        X_train, X_test, y_train, y_test, scaler = clf_split
        fitted = {"logreg": ThresholdClassifier(LogisticRegression().fit(X_train, y_train))}
        eval_report = pd.DataFrame({"model_key": ["rf", "logreg"]})  # "rf" absent de fitted

        save_models(fitted, scaler, eval_report)  # ne doit pas lever d'exception

        assert not (PATHS["models"] / "model_1_rf.joblib").exists()
        assert (PATHS["models"] / "model_2_logreg.joblib").exists()


# ===========================================================================
# 7. VISUALISATIONS
# ===========================================================================

class TestVisualisations:

    def test_plot_cv_comparison_creates_file(self, clf_split, tmp_outputs, monkeypatch):
        from config import PATHS
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)
        cv_results = cross_validate_models({"logreg": LogisticRegression()}, X_train, y_train)
        plot_cv_comparison(cv_results)
        out = PATHS["figures_eval"] / "cv_comparison.png"
        assert out.exists() and out.stat().st_size > 0

    @pytest.fixture
    def fitted_for_plots(self, clf_split):
        X_train, X_test, y_train, y_test, scaler = clf_split
        fitted = {
            "logreg": ThresholdClassifier(LogisticRegression().fit(X_train, y_train), threshold=0.5),
            "rf":     ThresholdClassifier(
                RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train),
                threshold=0.5,
            ),
        }
        eval_report = pd.DataFrame({
            "model_key": ["logreg", "rf"],
            "rank":      [1, 2],
        })
        return fitted, X_test, y_test, eval_report

    def test_plot_roc_curves_creates_file(self, fitted_for_plots, tmp_outputs):
        from config import PATHS
        fitted, X_test, y_test, eval_report = fitted_for_plots
        plot_roc_curves(fitted, X_test, y_test, eval_report)
        out = PATHS["figures_eval"] / "roc_curves.png"
        assert out.exists() and out.stat().st_size > 0

    def test_plot_confusion_matrices_creates_file(self, fitted_for_plots, tmp_outputs):
        from config import PATHS
        fitted, X_test, y_test, _ = fitted_for_plots
        plot_confusion_matrices(fitted, X_test, y_test, thresholds=None)
        out = PATHS["figures_eval"] / "confusion_matrices.png"
        assert out.exists() and out.stat().st_size > 0

    def test_plot_feature_importance_creates_file_for_tree_models(self, fitted_for_plots, tmp_outputs):
        from config import PATHS, FEATURES
        fitted, X_test, y_test, _ = fitted_for_plots
        plot_feature_importance(fitted, FEATURES)
        out = PATHS["figures_eval"] / "feature_importance.png"
        assert out.exists() and out.stat().st_size > 0

    def test_plot_confusion_matrices_hides_empty_axes(self, clf_split, tmp_outputs):
        """Avec 4 modèles et ncols=3, 2 axes doivent rester vides et être masqués (ligne 678)."""
        from config import PATHS
        X_train, X_test, y_train, y_test, _ = clf_split
        fitted = {
            f"m{i}": ThresholdClassifier(LogisticRegression().fit(X_train, y_train))
            for i in range(4)
        }
        plot_confusion_matrices(fitted, X_test, y_test)
        out = PATHS["figures_eval"] / "confusion_matrices.png"
        assert out.exists() and out.stat().st_size > 0

    def test_plot_feature_importance_skips_when_no_model_exposes_it(self, clf_split, tmp_outputs):
        """SVC(kernel='rbf') n'expose ni feature_importances_ ni coef_."""
        from config import PATHS, FEATURES
        from sklearn.svm import SVC

        X_train, X_test, y_train, y_test, _ = clf_split
        fitted = {"svm_rbf": ThresholdClassifier(SVC(kernel="rbf", probability=True).fit(X_train, y_train))}

        plot_feature_importance(fitted, FEATURES)
        out = PATHS["figures_eval"] / "feature_importance.png"
        assert not out.exists()


# ===========================================================================
# 7b. RÉÉQUILIBRAGE SMOTETomek (via main())
# ===========================================================================

class TestRebalancingBranch:

    def test_smotetomek_applied_when_enabled_and_available(
        self, raw_csv_path, tmp_outputs, monkeypatch
    ):
        pytest.importorskip("imblearn")
        monkeypatch.setitem(tm.CV, "n_splits", 3)
        monkeypatch.setitem(tm.REBALANCING, "enabled", True)
        monkeypatch.setattr(
            tm, "build_models",
            lambda best_params_path=None: {"logreg": LogisticRegression()},
        )
        # Ne doit pas lever d'exception ; SMOTETomek doit s'exécuter normalement
        result = main(data_path=raw_csv_path, best_params_path=None, top_n=1)
        assert "logreg" in result["fitted_models"]

    def test_smotetomek_gracefully_skipped_when_imblearn_missing(
        self, raw_csv_path, tmp_outputs, monkeypatch
    ):
        import sys
        monkeypatch.setitem(tm.CV, "n_splits", 3)
        monkeypatch.setitem(tm.REBALANCING, "enabled", True)
        monkeypatch.setattr(
            tm, "build_models",
            lambda best_params_path=None: {"logreg": LogisticRegression()},
        )
        # Simule l'absence d'imbalanced-learn même s'il est installé dans le sandbox
        monkeypatch.setitem(sys.modules, "imblearn.combine", None)
        monkeypatch.setitem(sys.modules, "imblearn.over_sampling", None)

        # Le pipeline doit continuer sans planter (ImportError attrapé + warning)
        result = main(data_path=raw_csv_path, best_params_path=None, top_n=1)
        assert "logreg" in result["fitted_models"]


# ===========================================================================
# 7c. CHARGEMENT DES SEUILS DEPUIS best_params.json (via main())
# ===========================================================================

class TestMainThresholdLoading:

    def test_thresholds_from_best_params_are_applied(
        self, raw_csv_path, tmp_outputs, monkeypatch
    ):
        from config import PATHS

        monkeypatch.setitem(tm.CV, "n_splits", 3)
        monkeypatch.setattr(
            tm, "build_models",
            lambda best_params_path=None: {"logreg": LogisticRegression()},
        )

        best_params = {"logreg": {"label": "Logistic Regression", "params": {}, "threshold": 0.27}}
        PATHS["best_params"].write_text(json.dumps(best_params), encoding="utf-8")

        result = main(data_path=raw_csv_path, best_params_path=PATHS["best_params"], top_n=1)
        assert result["fitted_models"]["logreg"].threshold == pytest.approx(0.27)


# ===========================================================================
# 8. write_summary_report
# ===========================================================================

class TestWriteSummaryReport:

    def test_writes_txt_and_csv_without_best_params(self, clf_split, tmp_outputs, monkeypatch):
        from config import PATHS
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)

        top_models = {"logreg": LogisticRegression()}
        cv_results = cross_validate_models(top_models, X_train, y_train)
        eval_report, _ = train_and_evaluate(
            top_models, X_train, X_test, y_train, y_test, cv_results,
        )

        write_summary_report(cv_results, eval_report, PATHS["best_params"])

        assert PATHS["summary_txt"].exists()
        assert PATHS["eval_report"].exists()
        content = PATHS["summary_txt"].read_text(encoding="utf-8")
        assert "RAPPORT D'ÉVALUATION" in content

    def test_includes_best_params_section_when_file_exists(self, clf_split, tmp_outputs, monkeypatch):
        from config import PATHS
        X_train, X_test, y_train, y_test, _ = clf_split
        monkeypatch.setitem(tm.CV, "n_splits", 3)

        best_params = {"logreg": {"label": "Logistic Regression", "params": {"C": 1.0}, "roc_auc_cv": 0.9}}
        PATHS["best_params"].write_text(json.dumps(best_params), encoding="utf-8")

        top_models = {"logreg": LogisticRegression()}
        cv_results = cross_validate_models(top_models, X_train, y_train)
        eval_report, _ = train_and_evaluate(
            top_models, X_train, X_test, y_train, y_test, cv_results,
        )

        write_summary_report(cv_results, eval_report, PATHS["best_params"])
        content = PATHS["summary_txt"].read_text(encoding="utf-8")
        assert "MEILLEURS HYPERPARAMÈTRES" in content
        assert "LOGREG" in content


# ===========================================================================
# 9. main() — pipeline complet (intégration)
# ===========================================================================

class TestMainPipeline:

    def test_full_pipeline_runs_and_produces_artifacts(
        self, raw_csv_path, tmp_outputs, monkeypatch
    ):
        from config import PATHS

        # Accélère le test : CV réduite + seulement 2 modèles légers
        monkeypatch.setitem(tm.CV, "n_splits", 3)
        monkeypatch.setattr(
            tm, "build_models",
            lambda best_params_path=None: {
                "logreg": LogisticRegression(),
                "rf":     RandomForestClassifier(n_estimators=15, random_state=42),
            },
        )

        result = main(data_path=raw_csv_path, best_params_path=None, top_n=2)

        assert set(result["fitted_models"].keys()) <= {"logreg", "rf"}
        assert "composite_score" in result["eval_report"].columns
        assert list(result["eval_report"]["rank"]) == list(
            range(1, len(result["eval_report"]) + 1)
        )
        # Classé par composite_score décroissant
        assert result["eval_report"]["composite_score"].is_monotonic_decreasing

        assert (PATHS["models"] / "scaler.joblib").exists()
        assert PATHS["summary_txt"].exists()
        assert PATHS["eval_report"].exists()
        assert (PATHS["figures_eval"] / "roc_curves.png").exists()

    def test_warns_and_uses_defaults_without_best_params_json(
        self, raw_csv_path, tmp_outputs, monkeypatch, caplog
    ):
        monkeypatch.setitem(tm.CV, "n_splits", 3)
        monkeypatch.setattr(
            tm, "build_models",
            lambda best_params_path=None: {"logreg": LogisticRegression()},
        )
        result = main(data_path=raw_csv_path, best_params_path=None, top_n=1)
        assert "logreg" in result["fitted_models"]
