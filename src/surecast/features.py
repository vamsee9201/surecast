"""Feature engineering for surecast, with a hard anti-leakage guarantee.

Every engineered feature at timestamp ``t`` is knowable at ``t - HORIZON``:

* **calendar** features are deterministic functions of the timestamp (known
  arbitrarily far ahead);
* **weather** features are treated as available from a weather forecast at
  prediction time (a documented framing assumption);
* **target lags** reference ``cnt`` only at ``t - 24``, ``t - 25``, ``t - 48``
  and ``t - 168`` — all at least :data:`HORIZON` hours in the past;
* **rolling** statistics summarise a 24-hour window that *ends at* ``t - 24``
  (``cnt.shift(24).rolling(24)``), so they never touch ``cnt[t-1 .. t-23]``.

No feature may reference ``cnt`` inside the forbidden ``(t-24, t]`` window.
Because :func:`surecast.data.load_hourly` reindexes onto a gap-free hourly
grid, a positional ``shift(k)`` is exactly a ``k``-hour lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surecast import schema

HORIZON = 24  # hours ahead; the minimum admissible target lag

# Rolling window length (hours) for the shifted rolling statistics.
_ROLL_WINDOW = 24

# Lag depths (hours) for the target-lag features. Every value is >= HORIZON.
_LAGS = {"lag_24": 24, "lag_25": 25, "lag_48": 48, "lag_168": 168}


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the leakage-safe feature matrix and aligned target.

    Args:
        df: Frame from :func:`surecast.data.load_hourly` (complete hourly index,
            ``cnt`` target plus raw weather/calendar columns).

    Returns:
        ``(X, y)`` where ``X`` has columns :data:`schema.FEATURES` (float64) and
        ``y`` is ``cnt``. Rows with any ``NaN`` feature or ``NaN`` target — from
        reindex gaps or lag/rolling warm-up — are dropped; ``X`` and ``y`` share
        an identical ``DatetimeIndex``.

    Raises:
        TypeError: If ``df`` is not indexed by a ``DatetimeIndex``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_features expects a DatetimeIndex (from load_hourly).")

    index = df.index
    cnt = df[schema.TARGET].astype("float64")

    features: dict[str, pd.Series] = {}

    # --- Calendar (known arbitrarily far ahead) -------------------------------
    hour = index.hour.to_numpy()
    features["hour"] = pd.Series(hour, index=index, dtype="float64")
    features["dayofweek"] = pd.Series(index.dayofweek.to_numpy(), index=index, dtype="float64")
    features["month"] = pd.Series(index.month.to_numpy(), index=index, dtype="float64")
    features["is_weekend"] = pd.Series(
        (index.dayofweek.to_numpy() >= 5).astype("float64"), index=index
    )
    features["workingday"] = df["workingday"].astype("float64")
    features["holiday"] = df["holiday"].astype("float64")
    features["hour_sin"] = pd.Series(np.sin(2.0 * np.pi * hour / 24.0), index=index)
    features["hour_cos"] = pd.Series(np.cos(2.0 * np.pi * hour / 24.0), index=index)

    # --- Weather (assumed forecast at prediction time) ------------------------
    for column in schema.WEATHER_FEATURES:
        features[column] = df[column].astype("float64")

    # --- Target lags (all >= HORIZON hours) -----------------------------------
    for name, depth in _LAGS.items():
        assert depth >= HORIZON, f"lag {name} of {depth}h violates the >= {HORIZON}h rule"
        features[name] = cnt.shift(depth)

    # --- Rolling over the PAST, window ending at t - HORIZON ------------------
    # cnt.shift(HORIZON) moves the window's right edge to t - HORIZON, so the
    # 24h window covers cnt[t-47 .. t-24] and never cnt[t-1 .. t-23].
    shifted = cnt.shift(HORIZON)
    features["roll_mean_24"] = shifted.rolling(_ROLL_WINDOW).mean()
    features["roll_std_24"] = shifted.rolling(_ROLL_WINDOW).std()

    x = pd.DataFrame(features, index=index)[schema.FEATURES].astype("float64")
    y = cnt

    mask = x.notna().all(axis=1) & y.notna()
    x = x.loc[mask]
    y = y.loc[mask]

    return x, y
