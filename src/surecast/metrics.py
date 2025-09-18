"""Point and interval evaluation metrics.

Every function accepts array-likes (lists, numpy arrays, pandas Series) and
returns a plain Python ``float``. Metrics are pure and stateless.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "coverage",
    "mae",
    "mean_interval_width",
    "pinball_loss",
    "rmse",
    "smape",
]


def _as_1d(a: ArrayLike) -> np.ndarray:
    """Coerce an array-like to a 1-D float ``ndarray``."""
    arr = np.asarray(a, dtype=float).ravel()
    return arr


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric mean absolute percentage error, in percent.

    Defined as ``mean(2*|y-yhat| / (|y| + |yhat|)) * 100``. When both the
    actual and the prediction are zero the per-element term is defined as 0
    (no division by zero).
    """
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    numerator = 2.0 * np.abs(yt - yp)
    denominator = np.abs(yt) + np.abs(yp)
    # Where denominator == 0, both values are 0 -> perfect agreement -> term 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denominator == 0.0, 0.0, numerator / denominator)
    return float(np.mean(terms) * 100.0)


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of ``y_true`` falling within the closed interval ``[lower, upper]``.

    Endpoints are inclusive.
    """
    yt = _as_1d(y_true)
    lo = _as_1d(lower)
    up = _as_1d(upper)
    inside = (yt >= lo) & (yt <= up)
    return float(np.mean(inside))


def mean_interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean width ``upper - lower`` of the prediction intervals."""
    lo = _as_1d(lower)
    up = _as_1d(upper)
    return float(np.mean(up - lo))


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    """Average pinball (quantile) loss at ``quantile`` in ``(0, 1)``.

    For a target quantile ``q``, the per-element loss is
    ``max(q * (y - yhat), (q - 1) * (y - yhat))``.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    diff = yt - yp
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))
