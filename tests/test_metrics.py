"""Hand-computed checks for the evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from surecast.metrics import (
    coverage,
    mae,
    mean_interval_width,
    pinball_loss,
    rmse,
    smape,
)


def test_mae_hand_computed() -> None:
    # |1-1| + |2-4| + |3-1| = 0 + 2 + 2 = 4, /3
    assert mae([1, 2, 3], [1, 4, 1]) == pytest.approx(4 / 3)


def test_rmse_hand_computed() -> None:
    # errors 0, 2, -2 -> squared 0,4,4 -> mean 8/3 -> sqrt
    assert rmse([1, 2, 3], [1, 4, 1]) == pytest.approx(math.sqrt(8 / 3))


def test_metrics_accept_pandas_and_lists() -> None:
    yt = pd.Series([10.0, 20.0])
    yp = np.array([12.0, 18.0])
    assert mae(yt, yp) == pytest.approx(2.0)
    assert isinstance(mae(yt, yp), float)


def test_smape_hand_computed() -> None:
    # y=100, yhat=110 -> 2*10/210 = 0.095238...; y=50,yhat=50 -> 0
    val = smape([100.0, 50.0], [110.0, 50.0])
    expected = ((2 * 10 / 210) + 0.0) / 2 * 100
    assert val == pytest.approx(expected)


def test_smape_both_zero_no_div_by_zero() -> None:
    # Both zero everywhere -> perfect agreement -> 0, and no warning/nan.
    with np.errstate(all="raise"):
        val = smape([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert val == 0.0


def test_smape_mixed_zero_and_nonzero() -> None:
    # one exact-zero pair (term 0) + one where y=0,yhat=4 -> 2*4/4 = 2
    val = smape([0.0, 0.0], [0.0, 4.0])
    assert val == pytest.approx((0.0 + 2.0) / 2 * 100)


def test_coverage_boundaries_inclusive() -> None:
    # y exactly on lower and upper endpoints must count as covered.
    y = [0.0, 5.0, 10.0, 11.0]
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [10.0, 10.0, 10.0, 10.0]
    # covered: 0(==lower), 5(inside), 10(==upper); not covered: 11
    assert coverage(y, lower, upper) == pytest.approx(3 / 4)


def test_coverage_all_and_none() -> None:
    assert coverage([1, 2, 3], [0, 0, 0], [5, 5, 5]) == 1.0
    assert coverage([9, 9, 9], [0, 0, 0], [5, 5, 5]) == 0.0


def test_mean_interval_width() -> None:
    assert mean_interval_width([0.0, 1.0], [2.0, 5.0]) == pytest.approx((2 + 4) / 2)


def test_pinball_loss_hand_computed() -> None:
    # quantile 0.5 -> pinball = 0.5 * |diff| = 0.5 * MAE
    yt = [1.0, 2.0, 3.0]
    yp = [1.0, 4.0, 1.0]
    assert pinball_loss(yt, yp, 0.5) == pytest.approx(0.5 * mae(yt, yp))


def test_pinball_loss_asymmetric() -> None:
    # under-prediction at high quantile is penalised more.
    # y=10, yhat=8, q=0.9: diff=2 -> max(0.9*2, -0.1*2)=1.8
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)
    # over-prediction: y=8, yhat=10, q=0.9: diff=-2 -> max(-1.8, 0.2)=0.2
    assert pinball_loss([8.0], [10.0], 0.9) == pytest.approx(0.2)


def test_pinball_loss_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.0)
