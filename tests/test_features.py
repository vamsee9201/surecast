"""Tests for surecast.features.build_features.

The leakage guard is the most important test in the project: it must genuinely
catch a target lag / rolling feature that peeks at cnt inside the forbidden
``(t - 24, t]`` window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surecast import schema
from surecast.features import HORIZON, build_features

_N_HOURS = 400  # > 168 warm-up, leaves a healthy usable tail


def _make_frame(cnt: np.ndarray) -> pd.DataFrame:
    """A gap-free hourly frame in load_hourly's output shape, given a cnt array."""
    index = pd.date_range("2011-01-01", periods=len(cnt), freq="h", name=schema.TS)
    n = len(cnt)
    return pd.DataFrame(
        {
            schema.TARGET: cnt.astype("float64"),
            "temp": np.linspace(0.1, 0.9, n),
            "atemp": np.linspace(0.1, 0.9, n),
            "hum": np.full(n, 0.5),
            "windspeed": np.full(n, 0.2),
            "weathersit": pd.array((np.arange(n) % 3 + 1), dtype="Int64"),
            "workingday": pd.array(np.where(index.dayofweek < 5, 1, 0), dtype="Int64"),
            "holiday": pd.array(np.zeros(n, dtype=int), dtype="Int64"),
            "season": pd.array(np.ones(n, dtype=int), dtype="Int64"),
        },
        index=index,
    )


def _unique_cnt() -> np.ndarray:
    """Strictly increasing, all-distinct cnt so any wrong lag is detectable."""
    return (np.arange(_N_HOURS, dtype="float64") + 1.0) * 3.0 + 100.0


def test_columns_and_order() -> None:
    x, _ = build_features(_make_frame(_unique_cnt()))
    assert list(x.columns) == schema.FEATURES


def test_no_nans_and_aligned_index() -> None:
    x, y = build_features(_make_frame(_unique_cnt()))
    assert not x.isna().to_numpy().any()
    assert not y.isna().to_numpy().any()
    assert x.index.equals(y.index)
    # Warm-up (first 168h) and nothing else is dropped.
    assert len(x) == _N_HOURS - 168


def test_lags_equal_hand_computed_past_values() -> None:
    """Every lag column equals cnt exactly 24/25/48/168 hours earlier."""
    cnt = _unique_cnt()
    frame = _make_frame(cnt)
    x, _ = build_features(frame)

    pos = 200  # a usable timestamp well past warm-up
    ts = frame.index[pos]
    assert x.loc[ts, "lag_24"] == cnt[pos - 24]
    assert x.loc[ts, "lag_25"] == cnt[pos - 25]
    assert x.loc[ts, "lag_48"] == cnt[pos - 48]
    assert x.loc[ts, "lag_168"] == cnt[pos - 168]


def test_rolling_equals_window_ending_at_t_minus_24() -> None:
    """roll_mean/std_24 summarise cnt[t-47 .. t-24] (24 values, ending at t-24)."""
    cnt = _unique_cnt()
    frame = _make_frame(cnt)
    x, _ = build_features(frame)

    pos = 200
    ts = frame.index[pos]
    window = cnt[pos - 47 : pos - 23]  # inclusive of pos-47 .. pos-24 -> 24 values
    assert window.shape == (24,)
    assert x.loc[ts, "roll_mean_24"] == pytest.approx(window.mean())
    assert x.loc[ts, "roll_std_24"] == pytest.approx(window.std(ddof=1))


def test_leakage_guard_perturbation() -> None:
    """A single spike in cnt[p] must NOT change any feature row at t in [p, p+23].

    Those rows may legitimately reference cnt only up to t-24 <= p-1, so cnt[p]
    is off-limits for them. If a feature used cnt[t-1..t-23] (or cnt[t] itself),
    perturbing cnt[p] would leak into at least one of these rows and this test
    would fail. This is what makes a lag<24 mistake genuinely detectable.
    """
    base_cnt = _unique_cnt()
    x_base, _ = build_features(_make_frame(base_cnt))

    p = 250
    spiked = base_cnt.copy()
    spiked[p] += 1_000_000.0  # unmistakable spike
    x_spiked, _ = build_features(_make_frame(spiked))

    index = pd.date_range("2011-01-01", periods=_N_HOURS, freq="h", name=schema.TS)
    forbidden_window = index[p : p + HORIZON]  # t in [p, p+23]

    common = x_base.index.intersection(forbidden_window)
    assert len(common) == HORIZON  # all 24 rows are present in the usable range
    pd.testing.assert_frame_equal(x_base.loc[common], x_spiked.loc[common])

    # Sanity: the spike DOES propagate to the first legitimately-affected row
    # (t = p + 24, whose lag_24 is exactly cnt[p]).
    first_affected = index[p + HORIZON]
    assert x_base.loc[first_affected, "lag_24"] != x_spiked.loc[first_affected, "lag_24"]


def test_leakage_guard_detects_a_broken_implementation() -> None:
    """Meta-test: the perturbation guard fails against a lag<24 implementation.

    We emulate a buggy feature builder (lag_1 = cnt.shift(1)) and confirm the
    same window comparison the guard uses would raise — proving the guard has
    teeth rather than passing vacuously.
    """
    base_cnt = _unique_cnt()
    index = pd.date_range("2011-01-01", periods=_N_HOURS, freq="h", name=schema.TS)

    def buggy_leaky_feature(cnt: np.ndarray) -> pd.Series:
        # lag_1 peeks at cnt[t-1]: inside the forbidden window -> leakage.
        return pd.Series(cnt, index=index).shift(1)

    p = 250
    spiked = base_cnt.copy()
    spiked[p] += 1_000_000.0

    base_feat = buggy_leaky_feature(base_cnt)
    spiked_feat = buggy_leaky_feature(spiked)

    forbidden_window = index[p : p + HORIZON]
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(
            base_feat.loc[forbidden_window], spiked_feat.loc[forbidden_window]
        )


def test_non_datetime_index_raises() -> None:
    frame = _make_frame(_unique_cnt()).reset_index(drop=True)
    with pytest.raises(TypeError):
        build_features(frame)
