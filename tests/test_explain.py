"""SHAP explainability checks.

Local wrapper classes mimic the contract shapes of ``GBMForecaster`` (one HGB)
and ``ConformalForecaster`` (lower/upper/median HGBs, median is the point model)
so ``explain.py`` is exercised without importing ``surecast.models`` (built in
parallel). A guarded test also runs against the real classes when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

from surecast.explain import global_importance, local_explanation


class GBMLike:
    """Single-tree point model wrapper (mimics GBMForecaster)."""

    def __init__(self, random_state: int = 0) -> None:
        self.model = HistGradientBoostingRegressor(random_state=random_state, max_iter=60)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GBMLike:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X))


class ConformalLike:
    """Three-tree wrapper; ``predict`` returns the median model (mimics CQR)."""

    def __init__(self, random_state: int = 0) -> None:
        self._lower = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.05, random_state=random_state, max_iter=60
        )
        self._upper = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.95, random_state=random_state, max_iter=60
        )
        self._median = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.5, random_state=random_state, max_iter=60
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConformalLike:
        self._lower.fit(X, y)
        self._upper.fit(X, y)
        self._median.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._median.predict(X))


@pytest.fixture
def fitted_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 400
    data = {
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": rng.normal(size=n),
    }
    X = pd.DataFrame(data)
    y = pd.Series(3.0 * X["a"] - 2.0 * X["b"] + 0.5 * X["c"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_global_importance_covers_all_features(fitted_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = fitted_data
    model = GBMLike().fit(X, y)
    imp = global_importance(model, X)
    assert isinstance(imp, pd.Series)
    assert set(imp.index) == set(X.columns)
    # non-negative mean-abs values
    assert (imp.values >= 0).all()
    # sorted descending
    assert list(imp.values) == sorted(imp.values, reverse=True)


def test_global_importance_ranks_signal_over_noise(
    fitted_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = fitted_data
    model = GBMLike().fit(X, y)
    imp = global_importance(model, X)
    # "a" and "b" drive y; "d" is pure noise and should rank below them.
    assert imp["a"] > imp["d"]
    assert imp["b"] > imp["d"]


def test_global_importance_respects_max_samples(
    fitted_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = fitted_data
    model = GBMLike().fit(X, y)
    # should not raise and still cover all features with a small cap
    imp = global_importance(model, X, max_samples=50)
    assert set(imp.index) == set(X.columns)


def test_local_explanation_additivity_gbm(fitted_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = fitted_data
    model = GBMLike().fit(X, y)
    row = 7
    local = local_explanation(model, X, row=row)
    assert set(local.index) == set(X.columns)

    base = float(np.asarray(shap.TreeExplainer(model.model).expected_value).ravel()[0])
    reconstructed = base + float(local.sum())
    assert reconstructed == pytest.approx(float(model.predict(X.iloc[[row]])[0]), abs=1e-4)


def test_point_estimator_selected_for_conformal_like(
    fitted_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = fitted_data
    model = ConformalLike().fit(X, y)
    row = 11
    local = local_explanation(model, X, row=row)

    # explanation must correspond to the MEDIAN (point) estimator, not lower/upper.
    base = float(np.asarray(shap.TreeExplainer(model._median).expected_value).ravel()[0])
    reconstructed = base + float(local.sum())
    assert reconstructed == pytest.approx(float(model.predict(X.iloc[[row]])[0]), abs=1e-4)

    # sanity: reconstructing against the lower model would NOT match the point forecast
    base_lo = float(np.asarray(shap.TreeExplainer(model._lower).expected_value).ravel()[0])
    lower_pred = float(model._lower.predict(X.iloc[[row]])[0])
    assert base_lo + float(local.sum()) != pytest.approx(lower_pred, abs=1e-4) or (
        lower_pred == pytest.approx(float(model.predict(X.iloc[[row]])[0]), abs=1e-4)
    )


def test_real_models_if_available(fitted_data: tuple[pd.DataFrame, pd.Series]) -> None:
    models = pytest.importorskip("surecast.models")
    X, y = fitted_data
    gbm_cls = getattr(models, "GBMForecaster", None)
    if gbm_cls is None:
        pytest.skip("GBMForecaster not present")
    model = gbm_cls().fit(X, y)
    imp = global_importance(model, X)
    assert set(imp.index) == set(X.columns)
    assert (imp.values >= 0).all()
    local = local_explanation(model, X, row=3)
    assert set(local.index) == set(X.columns)
