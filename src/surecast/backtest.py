"""Rolling-origin (expanding-window) backtesting for forecasters.

Uses :class:`sklearn.model_selection.TimeSeriesSplit` so that, in every fold,
the train indices strictly precede the test indices — the anti-leakage
guarantee for time-series evaluation. A *fresh* forecaster is built from a
factory for each fold; nothing fit on a fold ever sees that fold's test rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from surecast import metrics

if TYPE_CHECKING:
    from surecast.models import Forecaster

__all__ = ["BacktestReport", "FoldResult", "rolling_backtest"]


@dataclass
class FoldResult:
    """Predictions and ground truth for one model on one backtest fold."""

    model: str
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None


@dataclass
class BacktestReport:
    """Aggregated backtest output.

    ``summary`` maps each model name to its cross-fold mean metrics. Interval
    models additionally carry ``coverage`` and ``width`` keys.
    """

    per_fold: list[FoldResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)


def rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    forecasters: dict[str, Callable[[], Forecaster]],
    n_splits: int = 5,
    alpha: float = 0.1,
) -> BacktestReport:
    """Expanding-window rolling-origin cross-validation.

    Parameters
    ----------
    X, y:
        Feature matrix and target, assumed sorted ascending by timestamp and
        index-aligned.
    forecasters:
        Mapping of model name to a zero-argument factory returning a fresh
        (unfit) forecaster. A new instance is built for every fold so no state
        leaks across folds.
    n_splits:
        Number of expanding-window folds (``TimeSeriesSplit``).
    alpha:
        Passed through to interval-producing forecasters at construction time
        only implicitly (the factory owns its own ``alpha``); retained here so
        callers can document the nominal ``1 - alpha`` target coverage. Not used
        to build forecasters — the factory is the single source of truth.

    Returns
    -------
    BacktestReport
        Per-fold predictions plus cross-fold mean metrics per model.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if not forecasters:
        raise ValueError("forecasters mapping is empty")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    y_values = np.asarray(y, dtype=float)

    per_fold: list[FoldResult] = []
    # Accumulate per-fold metric dicts per model, averaged at the end.
    accum: dict[str, list[dict[str, float]]] = {name: [] for name in forecasters}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        # Leakage guard: TimeSeriesSplit guarantees this, assert it defensively.
        if train_idx.max() >= test_idx.min():
            raise AssertionError(
                f"fold {fold_idx}: train index {train_idx.max()} not before "
                f"test index {test_idx.min()}"
            )

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y_values[test_idx]

        for name, factory in forecasters.items():
            model = factory()
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=float)

            lower: np.ndarray | None = None
            upper: np.ndarray | None = None
            fold_metrics: dict[str, float] = {
                "mae": metrics.mae(y_test, y_pred),
                "rmse": metrics.rmse(y_test, y_pred),
                "smape": metrics.smape(y_test, y_pred),
            }

            predict_interval = getattr(model, "predict_interval", None)
            if callable(predict_interval):
                lo, up = predict_interval(X_test)
                lower = np.asarray(lo, dtype=float)
                upper = np.asarray(up, dtype=float)
                fold_metrics["coverage"] = metrics.coverage(y_test, lower, upper)
                fold_metrics["width"] = metrics.mean_interval_width(lower, upper)

            per_fold.append(
                FoldResult(
                    model=name,
                    fold=fold_idx,
                    y_true=y_test.copy(),
                    y_pred=y_pred,
                    lower=lower,
                    upper=upper,
                )
            )
            accum[name].append(fold_metrics)

    summary: dict[str, dict[str, float]] = {}
    for name, fold_dicts in accum.items():
        keys = fold_dicts[0].keys()
        summary[name] = {
            key: float(np.mean([fd[key] for fd in fold_dicts])) for key in keys
        }

    return BacktestReport(per_fold=per_fold, summary=summary)
