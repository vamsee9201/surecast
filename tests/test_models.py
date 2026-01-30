"""Tests for surecast.models.

Focus: the seasonal-naive baseline behaves, the GBM point model fits/predicts,
and — the crux of the project — the ConformalForecaster produces *calibrated*
intervals (empirical coverage on a held-out set ≈ nominal 1 - alpha) while
never training its quantile models on the calibration rows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from surecast.models import (
    ConformalForecaster,
    GBMForecaster,
    SeasonalNaiveForecaster,
)


def _make_regression(
    n: int, n_features: int = 4, noise: float = 3.0, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """iid synthetic regression: y = X @ w + intercept + Gaussian(noise)."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=cols)
    w = rng.normal(size=n_features)
    y = pd.Series(X.to_numpy() @ w + 10.0 + rng.normal(scale=noise, size=n), name="cnt")
    return X, y


# --------------------------------------------------------------------------- #
# SeasonalNaiveForecaster
# --------------------------------------------------------------------------- #
def test_seasonal_naive_returns_lag_168() -> None:
    X = pd.DataFrame({"lag_168": [1.0, 2.0, 3.0], "other": [9.0, 9.0, 9.0]})
    y = pd.Series([0.0, 0.0, 0.0])
    model = SeasonalNaiveForecaster().fit(X, y)
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    np.testing.assert_array_equal(pred, np.array([1.0, 2.0, 3.0]))


def test_seasonal_naive_fit_is_noop_and_returns_self() -> None:
    model = SeasonalNaiveForecaster()
    assert model.fit(pd.DataFrame({"lag_168": [1.0]}), pd.Series([1.0])) is model


def test_seasonal_naive_raises_without_lag_168() -> None:
    X = pd.DataFrame({"lag_24": [1.0, 2.0]})
    with pytest.raises(KeyError):
        SeasonalNaiveForecaster().predict(X)


# --------------------------------------------------------------------------- #
# GBMForecaster
# --------------------------------------------------------------------------- #
def test_gbm_fit_predict_shape() -> None:
    X, y = _make_regression(300, seed=1)
    model = GBMForecaster(random_state=0).fit(X, y)
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(X),)


def test_gbm_forwards_kwargs() -> None:
    model = GBMForecaster(random_state=0, max_iter=7)
    assert model.model.max_iter == 7
    assert model.model.get_params()["loss"] == "squared_error"


# --------------------------------------------------------------------------- #
# ConformalForecaster — construction / validation
# --------------------------------------------------------------------------- #
def test_conformal_quantile_levels() -> None:
    m = ConformalForecaster(alpha=0.1)
    assert m.q_lo == pytest.approx(0.05)
    assert m.q_hi == pytest.approx(0.95)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_conformal_rejects_bad_alpha(bad_alpha: float) -> None:
    with pytest.raises(ValueError):
        ConformalForecaster(alpha=bad_alpha)


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.2, 2.0])
def test_conformal_rejects_bad_calib_fraction(bad_frac: float) -> None:
    with pytest.raises(ValueError):
        ConformalForecaster(calib_fraction=bad_frac)


def test_conformal_raises_when_too_few_rows() -> None:
    X, y = _make_regression(3, seed=2)
    # calib_fraction 0.2 of 3 rows floors to 0 calibration rows -> invalid.
    with pytest.raises(ValueError):
        ConformalForecaster(alpha=0.1, calib_fraction=0.2).fit(X, y)


# --------------------------------------------------------------------------- #
# ConformalForecaster — the conformal-correction math
# --------------------------------------------------------------------------- #
def test_conformal_quantile_order_statistic() -> None:
    # n=10, alpha=0.1 -> k = ceil(11 * 0.9) = ceil(9.9) = 10 -> the max.
    scores = np.arange(1.0, 11.0)  # 1..10
    q = ConformalForecaster._conformal_quantile(scores, alpha=0.1)
    assert q == pytest.approx(10.0)
    assert math.ceil((10 + 1) * (1 - 0.1)) == 10


def test_conformal_quantile_infinite_when_k_exceeds_n() -> None:
    # n=5, alpha=0.1 -> k = ceil(6*0.9)=ceil(5.4)=6 > 5 -> +inf.
    scores = np.arange(1.0, 6.0)
    q = ConformalForecaster._conformal_quantile(scores, alpha=0.1)
    assert math.isinf(q)


# --------------------------------------------------------------------------- #
# ConformalForecaster — leakage guard: calibration excluded from training
# --------------------------------------------------------------------------- #
def test_calibration_rows_excluded_from_quantile_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spy on HistGradientBoostingRegressor.fit; assert every quantile model is
    trained on exactly ``n - n_calib`` rows (the calibration tail is held out).
    """
    fit_sizes: list[int] = []
    original_fit = HistGradientBoostingRegressor.fit

    def spy_fit(self: HistGradientBoostingRegressor, X: pd.DataFrame, y: pd.Series, **kw: object):  # type: ignore[no-untyped-def]
        fit_sizes.append(len(X))
        return original_fit(self, X, y, **kw)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy_fit)

    n = 500
    calib_fraction = 0.2
    X, y = _make_regression(n, seed=3)
    ConformalForecaster(alpha=0.1, calib_fraction=calib_fraction).fit(X, y)

    n_calib = math.floor(n * calib_fraction)
    expected_train = n - n_calib
    # three quantile models (lower, upper, median), each on the train split only.
    assert len(fit_sizes) == 3
    assert all(size == expected_train for size in fit_sizes)
    assert expected_train < n  # sanity: calibration really was withheld


# --------------------------------------------------------------------------- #
# ConformalForecaster — calibrated coverage on a held-out set
# --------------------------------------------------------------------------- #
def test_conformal_coverage_matches_nominal_on_holdout() -> None:
    """Empirical coverage on data the model never saw ≈ nominal 1 - alpha."""
    alpha = 0.1
    n_fit, n_test = 3000, 1500
    X, y = _make_regression(n_fit + n_test, n_features=4, noise=3.0, seed=42)

    X_fit, y_fit = X.iloc[:n_fit], y.iloc[:n_fit]
    X_test, y_test = X.iloc[n_fit:], y.iloc[n_fit:]

    model = ConformalForecaster(alpha=alpha, calib_fraction=0.3, random_state=0).fit(X_fit, y_fit)
    lower, upper = model.predict_interval(X_test)

    inside = (y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)
    coverage = float(inside.mean())
    assert coverage == pytest.approx(1 - alpha, abs=0.07)


def test_conformal_predict_returns_point_forecast() -> None:
    X, y = _make_regression(400, seed=5)
    model = ConformalForecaster(alpha=0.1, calib_fraction=0.25).fit(X, y)
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(X),)


def test_conformal_interval_nonneg_and_ordered() -> None:
    X, y = _make_regression(400, seed=6)
    model = ConformalForecaster(alpha=0.1, calib_fraction=0.25).fit(X, y)
    lower, upper = model.predict_interval(X)
    assert lower.shape == upper.shape == (len(X),)
    assert np.all(lower >= 0.0)  # clipped at zero (demand is non-negative)
    assert np.all(upper >= 0.0)
    assert np.all(upper >= lower)  # correction >= 0 keeps ordering


def test_conformal_correction_is_nonnegative() -> None:
    X, y = _make_regression(600, seed=7)
    model = ConformalForecaster(alpha=0.1, calib_fraction=0.2).fit(X, y)
    assert model.correction_ >= 0.0
    assert math.isfinite(model.correction_)
