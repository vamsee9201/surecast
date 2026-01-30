"""SHAP explainability over the underlying tree point model.

Both :class:`~surecast.models.GBMForecaster` and
:class:`~surecast.models.ConformalForecaster` wrap one or more
:class:`sklearn.ensemble.HistGradientBoostingRegressor` estimators. These
helpers locate the *point* estimator (the one whose predictions equal the
forecaster's ``predict`` output) and run a SHAP ``TreeExplainer`` on it, so the
explanations correspond exactly to the model's point forecast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

if TYPE_CHECKING:
    from surecast.models import ConformalForecaster, GBMForecaster

__all__ = ["global_importance", "local_explanation"]


def _collect_hgb(obj: object, _seen: set[int] | None = None) -> list[HistGradientBoostingRegressor]:
    """Recursively gather fitted ``HistGradientBoostingRegressor`` instances.

    Walks instance attributes and simple containers (list/tuple/dict values) one
    level of nesting at a time, guarding against cycles. Order is deterministic:
    attribute-declaration order, then container order.
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return []
    _seen.add(id(obj))

    found: list[HistGradientBoostingRegressor] = []
    if isinstance(obj, HistGradientBoostingRegressor):
        found.append(obj)
        return found

    values: list[Any] = []
    if isinstance(obj, dict):
        values = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        values = list(obj)
    elif hasattr(obj, "__dict__"):
        values = list(vars(obj).values())

    for value in values:
        if isinstance(value, HistGradientBoostingRegressor):
            found.append(value)
        elif isinstance(value, (dict, list, tuple)) or hasattr(value, "__dict__"):
            found.extend(_collect_hgb(value, _seen))
    return found


def _point_estimator(
    model: GBMForecaster | ConformalForecaster,
    X: pd.DataFrame,
) -> HistGradientBoostingRegressor:
    """Return the wrapped HGB whose predictions match ``model.predict``.

    For a plain GBM there is a single estimator. For the conformal wrapper there
    are lower/upper/median estimators; the point model is the one reproducing
    ``model.predict(X)``. Falls back to the first estimator found if none match
    (should not happen for contract-conforming models).
    """
    candidates = _collect_hgb(model)
    if not candidates:
        raise TypeError(
            f"no HistGradientBoostingRegressor found inside {type(model).__name__}"
        )

    target = np.asarray(model.predict(X), dtype=float)
    for est in candidates:
        try:
            pred = np.asarray(est.predict(X), dtype=float)
        except Exception:
            continue
        if pred.shape == target.shape and np.allclose(pred, target, rtol=1e-6, atol=1e-8):
            return est
    return candidates[0]


def _feature_names(X: pd.DataFrame) -> list[str]:
    return [str(c) for c in X.columns]


def global_importance(
    model: GBMForecaster | ConformalForecaster,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted descending.

    Parameters
    ----------
    model:
        A fitted forecaster wrapping a tree point model.
    X:
        Feature frame to explain. If it has more than ``max_samples`` rows, the
        first ``max_samples`` are used (contiguous, preserving time order).
    max_samples:
        Cap on rows sent through the explainer for speed.

    Returns
    -------
    pd.Series
        Index = feature names, values = ``mean(|shap|)``, sorted descending.
    """
    estimator = _point_estimator(model, X)
    X_sample = X.iloc[:max_samples] if len(X) > max_samples else X

    explainer = shap.TreeExplainer(estimator)
    shap_values = np.asarray(explainer.shap_values(X_sample))
    mean_abs = np.abs(shap_values).mean(axis=0)

    series = pd.Series(mean_abs, index=_feature_names(X_sample), name="mean_abs_shap")
    return series.sort_values(ascending=False)


def local_explanation(
    model: GBMForecaster | ConformalForecaster,
    X: pd.DataFrame,
    row: int,
    max_samples: int = 500,
) -> pd.Series:
    """Signed SHAP values for a single row.

    The returned values satisfy the SHAP additivity property:
    ``base_value + sum(shap) == point_estimator.predict(row)``.

    Parameters
    ----------
    model:
        A fitted forecaster wrapping a tree point model.
    X:
        Feature frame; ``row`` indexes it positionally (``iloc``).
    row:
        Positional index of the row to explain.
    max_samples:
        Accepted for signature symmetry; TreeExplainer explains the single row
        exactly and needs no background sample.

    Returns
    -------
    pd.Series
        Index = feature names, values = signed SHAP contributions.
    """
    _ = max_samples  # exact per-row tree explanation needs no background sample
    estimator = _point_estimator(model, X)
    x_row = X.iloc[[row]]

    explainer = shap.TreeExplainer(estimator)
    shap_values = np.asarray(explainer.shap_values(x_row))[0]

    return pd.Series(shap_values, index=_feature_names(X), name="shap")
