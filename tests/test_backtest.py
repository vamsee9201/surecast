"""Backtest leakage and aggregation checks with trivial deterministic models.

These tests deliberately define their own tiny forecasters so the backtest logic
is exercised without depending on ``surecast.models`` (built in parallel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surecast.backtest import BacktestReport, FoldResult, rolling_backtest


class RecordingMean:
    """Predicts the training mean; records the index ranges it was fit/asked on."""

    def __init__(self) -> None:
        self.train_index: pd.Index | None = None
        self.test_index: pd.Index | None = None
        self._mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingMean:
        self.train_index = X.index
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.test_index = X.index
        return np.full(len(X), self._mean, dtype=float)


class ConstantInterval:
    """Point = a constant, with a fixed-width symmetric interval."""

    def __init__(self, half_width: float = 5.0) -> None:
        self.half_width = half_width
        self._c = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConstantInterval:
        self._c = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._c, dtype=float)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        point = self.predict(X)
        return point - self.half_width, point + self.half_width


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2011-01-01", periods=60, freq="h", name="ts")
    X = pd.DataFrame({"f": np.arange(60, dtype=float)}, index=idx)
    y = pd.Series(np.arange(60, dtype=float) * 2.0, index=idx, name="cnt")
    return X, y


def test_produces_n_splits_folds_per_model(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    report = rolling_backtest(
        X, y, {"mean": RecordingMean, "interval": ConstantInterval}, n_splits=5
    )
    assert isinstance(report, BacktestReport)
    # 5 folds * 2 models
    assert len(report.per_fold) == 10
    for fr in report.per_fold:
        assert isinstance(fr, FoldResult)
        assert len(fr.y_true) == len(fr.y_pred)


def test_train_indices_strictly_precede_test(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    created: list[RecordingMean] = []

    def factory() -> RecordingMean:
        inst = RecordingMean()
        created.append(inst)
        return inst

    rolling_backtest(X, y, {"m": factory}, n_splits=4)

    assert len(created) == 4  # one fresh model per fold
    for inst in created:
        assert inst.train_index is not None and inst.test_index is not None
        # every training timestamp is strictly before every test timestamp
        assert inst.train_index.max() < inst.test_index.min()
        # no overlap between train and test rows
        assert len(inst.train_index.intersection(inst.test_index)) == 0


def test_summary_keys_point_vs_interval(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    report = rolling_backtest(
        X, y, {"mean": RecordingMean, "interval": ConstantInterval}, n_splits=5
    )
    point_keys = {"mae", "rmse", "smape"}
    # point-only model: exactly the point metrics, no interval metrics
    assert set(report.summary["mean"].keys()) == point_keys
    # interval model: point metrics plus coverage + width
    assert set(report.summary["interval"].keys()) == point_keys | {"coverage", "width"}
    # interval width is exactly the fixed 2*half_width
    assert report.summary["interval"]["width"] == pytest.approx(10.0)


def test_summary_values_are_floats(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    report = rolling_backtest(X, y, {"mean": RecordingMean}, n_splits=3)
    for v in report.summary["mean"].values():
        assert isinstance(v, float)


def test_fresh_model_per_fold_no_state_leak(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    seen_means: list[float] = []

    class SpyMean(RecordingMean):
        def fit(self, X: pd.DataFrame, y: pd.Series) -> SpyMean:
            super().fit(X, y)
            seen_means.append(self._mean)
            return self

    rolling_backtest(X, y, {"s": SpyMean}, n_splits=4)
    # expanding window -> training means differ across folds (monotone y)
    assert len(seen_means) == 4
    assert len(set(seen_means)) == 4


def test_length_mismatch_raises() -> None:
    X = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        rolling_backtest(X, y, {"m": RecordingMean})


def test_empty_forecasters_raises(data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = data
    with pytest.raises(ValueError):
        rolling_backtest(X, y, {})
