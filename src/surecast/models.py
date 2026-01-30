"""Forecasting models: a seasonal-naive baseline, a gradient-boosted point
model, and a Conformalized Quantile Regression (CQR) model that produces
*calibrated* prediction intervals.

The CQR implementation follows Romano, Patterson & Candès (2019),
"Conformalized Quantile Regression". Two quantile regressors are trained at the
lower/upper nominal quantiles, a held-out calibration set is used to compute
conformity scores, and the interval is widened by a single conformal
correction so that the finite-sample marginal coverage is at least ``1 - alpha``.

Anti-leakage rule for this module: the calibration set is a *time-ordered*
tail of the training data and is **never** used to fit the quantile models.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from surecast.schema import LAG_FEATURES


@runtime_checkable
class Forecaster(Protocol):
    """Minimal point-forecaster interface used across the project."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaiveForecaster:
    """Baseline: predict the value from the same hour one week ago.

    The forecast is simply the ``lag_168`` feature (168 h = 7 days). ``fit`` is
    a stateless no-op; the model carries no learned parameters.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaiveForecaster:
        """No-op: the seasonal-naive baseline is stateless. Returns ``self``."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the ``lag_168`` column as the point forecast.

        Raises:
            KeyError: if ``lag_168`` is not present in ``X``.
        """
        if "lag_168" not in X.columns:
            raise KeyError(
                "SeasonalNaiveForecaster requires the 'lag_168' feature column "
                f"(one of {LAG_FEATURES}); it is absent from X."
            )
        return X["lag_168"].to_numpy(dtype=float)


class GBMForecaster:
    """HistGradientBoostingRegressor point model (squared-error loss)."""

    def __init__(self, random_state: int = 0, **hgb_kwargs: object) -> None:
        """Create the underlying regressor.

        Args:
            random_state: seed for reproducibility.
            **hgb_kwargs: extra keyword args forwarded to
                :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
        """
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(
            loss="squared_error", random_state=random_state, **hgb_kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GBMForecaster:
        """Fit the point model on ``(X, y)``. Returns ``self``."""
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the point forecast as a numpy array."""
        return np.asarray(self.model.predict(X), dtype=float)


class ConformalForecaster:
    """Conformalized Quantile Regression for calibrated prediction intervals.

    Wraps three :class:`HistGradientBoostingRegressor` quantile models — lower
    (``alpha/2``), upper (``1 - alpha/2``) and median (``0.5``, the point
    forecast) — and calibrates the interval width on a time-ordered held-out
    calibration set so that empirical coverage matches the nominal ``1 - alpha``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        calib_fraction: float = 0.2,
        random_state: int = 0,
    ) -> None:
        """Configure the conformal forecaster.

        Args:
            alpha: miscoverage rate; target coverage is ``1 - alpha`` (default
                0.1 → 90%). Lower quantile is ``alpha/2``, upper ``1 - alpha/2``.
            calib_fraction: fraction of the (time-ordered) training rows held
                out as the calibration set. Must be in ``(0, 1)``.
            random_state: seed for the underlying quantile regressors.

        Raises:
            ValueError: if ``alpha`` or ``calib_fraction`` are out of range.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        if not 0.0 < calib_fraction < 1.0:
            raise ValueError(f"calib_fraction must be in (0, 1); got {calib_fraction}.")
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.random_state = random_state
        self.q_lo = alpha / 2.0
        self.q_hi = 1.0 - alpha / 2.0

        self.lower_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=self.q_lo, random_state=random_state
        )
        self.upper_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=self.q_hi, random_state=random_state
        )
        self.median_model = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.5, random_state=random_state
        )
        self.correction_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConformalForecaster:
        """Fit the quantile models and compute the conformal correction.

        The last ``calib_fraction`` of rows (``X`` is assumed pre-sorted by
        timestamp) form the calibration set; the earlier rows train the three
        quantile models. Conformity scores
        ``E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))`` are computed on the
        calibration set, and the conformal correction is stored as the
        ``ceil((n_calib + 1)(1 - alpha)) / n_calib`` empirical quantile of the
        ``E_i``. The calibration set is never used to train the models.

        Raises:
            ValueError: if there are too few rows to form both a non-empty
                training set and a non-empty calibration set.
        """
        n = len(X)
        n_calib = math.floor(n * self.calib_fraction)
        if n_calib < 1 or n - n_calib < 1:
            raise ValueError(
                f"Not enough rows ({n}) for a train/calibration split with "
                f"calib_fraction={self.calib_fraction} (need >=1 row on each side)."
            )

        # Time-ordered split: earlier rows train, the tail calibrates.
        X_train = X.iloc[: n - n_calib]
        y_train = y.iloc[: n - n_calib]
        X_calib = X.iloc[n - n_calib :]
        y_calib = y.iloc[n - n_calib :].to_numpy(dtype=float)

        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)
        self.median_model.fit(X_train, y_train)

        # CQR conformity scores on the calibration set (models never saw it).
        lo_pred = np.asarray(self.lower_model.predict(X_calib), dtype=float)
        hi_pred = np.asarray(self.upper_model.predict(X_calib), dtype=float)
        scores = np.maximum(lo_pred - y_calib, y_calib - hi_pred)

        self.correction_ = self._conformal_quantile(scores, self.alpha)
        return self

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """The ``ceil((n+1)(1-alpha))/n`` empirical quantile of ``scores``.

        This is the finite-sample conformal correction: the ``k``-th smallest
        score where ``k = ceil((n + 1)(1 - alpha))``. If ``k > n`` (too little
        calibration data for the requested coverage) the correction is
        ``+inf``, yielding an unbounded — and therefore trivially valid —
        interval.
        """
        n = scores.shape[0]
        k = math.ceil((n + 1) * (1.0 - alpha))
        if k > n:
            return float("inf")
        ordered = np.sort(scores)
        return float(ordered[k - 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the median-quantile point forecast as a numpy array."""
        return np.asarray(self.median_model.predict(X), dtype=float)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated ``(lower, upper)`` bounds for each row of ``X``.

        The raw quantile predictions are widened by the conformal correction:
        ``lower = q_lo(x) - correction``, ``upper = q_hi(x) + correction``.
        Both bounds are clipped at 0 because demand is non-negative.
        """
        lo_pred = np.asarray(self.lower_model.predict(X), dtype=float)
        hi_pred = np.asarray(self.upper_model.predict(X), dtype=float)
        lower = np.clip(lo_pred - self.correction_, 0.0, None)
        upper = np.clip(hi_pred + self.correction_, 0.0, None)
        return lower, upper
