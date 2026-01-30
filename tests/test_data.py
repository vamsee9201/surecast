"""Tests for surecast.data.load_hourly (no network — synthetic CSV fixtures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from surecast import schema
from surecast.data import load_hourly

# Raw UCI hour.csv column order.
_RAW_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]

# Hours deliberately omitted from day 2 to create reindex gaps.
_MISSING_HOURS = {5, 6, 7}


def _write_fixture(path: Path) -> int:
    """Write a 2-day hour.csv with three missing hours; return expected row count."""
    rows = []
    instant = 1
    for day, date in enumerate(["2011-01-01", "2011-01-02"]):
        for hr in range(24):
            if day == 1 and hr in _MISSING_HOURS:
                continue  # leave a gap for reindex to fill
            rows.append(
                {
                    "instant": instant,
                    "dteday": date,
                    "season": 1,
                    "yr": 0,
                    "mnth": 1,
                    "hr": hr,
                    "holiday": 0,
                    "weekday": 6 - day,
                    "workingday": day,  # 0 then 1, so we can check the value survives
                    "weathersit": 1 + (hr % 3),
                    "temp": 0.2 + hr * 0.01,
                    "atemp": 0.25 + hr * 0.01,
                    "hum": 0.5,
                    "windspeed": 0.1,
                    "casual": hr,
                    "registered": hr * 2,
                    "cnt": 100 + instant,
                }
            )
            instant += 1
    frame = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    frame.to_csv(path, index=False)
    return 48  # full hourly grid from 2011-01-01 00:00 to 2011-01-02 23:00


def test_complete_hourly_index_with_gaps_filled(tmp_path: Path) -> None:
    csv_path = tmp_path / "hour.csv"
    expected_len = _write_fixture(csv_path)

    df = load_hourly(csv_path)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == schema.TS
    assert len(df) == expected_len
    # A complete grid: consecutive timestamps are exactly one hour apart.
    deltas = df.index.to_series().diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(hours=1)]
    # Full contiguous range, endpoints inclusive.
    assert df.index[0] == pd.Timestamp("2011-01-01 00:00")
    assert df.index[-1] == pd.Timestamp("2011-01-02 23:00")


def test_gap_rows_are_nan(tmp_path: Path) -> None:
    csv_path = tmp_path / "hour.csv"
    _write_fixture(csv_path)

    df = load_hourly(csv_path)

    gap_ts = [pd.Timestamp(f"2011-01-02 0{h}:00") for h in sorted(_MISSING_HOURS)]
    for ts in gap_ts:
        assert pd.isna(df.loc[ts, schema.TARGET])
        for column in ["temp", "atemp", "hum", "windspeed"]:
            assert pd.isna(df.loc[ts, column])
        # Nullable-int columns carry <NA> on gap rows.
        assert pd.isna(df.loc[ts, "weathersit"])

    # Non-gap rows retain their values.
    present = pd.Timestamp("2011-01-02 04:00")
    assert not pd.isna(df.loc[present, schema.TARGET])


def test_dtypes(tmp_path: Path) -> None:
    csv_path = tmp_path / "hour.csv"
    _write_fixture(csv_path)

    df = load_hourly(csv_path)

    assert pd.api.types.is_float_dtype(df[schema.TARGET])
    for column in ["temp", "atemp", "hum", "windspeed"]:
        assert pd.api.types.is_float_dtype(df[column])
    for column in ["weathersit", "workingday", "holiday", "season"]:
        assert pd.api.types.is_integer_dtype(df[column])


def test_sorted_and_unique(tmp_path: Path) -> None:
    csv_path = tmp_path / "hour.csv"
    _write_fixture(csv_path)

    df = load_hourly(csv_path)

    assert df.index.is_monotonic_increasing
    assert df.index.is_unique


def test_duplicate_timestamps_dropped(tmp_path: Path) -> None:
    csv_path = tmp_path / "hour.csv"
    _write_fixture(csv_path)
    # Append a duplicate of the very first row (same dteday+hr).
    raw = pd.read_csv(csv_path)
    dup = raw.iloc[[0]].copy()
    dup["cnt"] = 999999
    pd.concat([raw, dup], ignore_index=True).to_csv(csv_path, index=False)

    df = load_hourly(csv_path)

    assert df.index.is_unique
    # First occurrence kept, not the appended duplicate.
    assert df.loc[pd.Timestamp("2011-01-01 00:00"), schema.TARGET] != 999999


def test_expected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "hour.csv"
    _write_fixture(csv_path)

    df = load_hourly(csv_path)

    expected = {
        schema.TARGET,
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "weathersit",
        "workingday",
        "holiday",
        "season",
    }
    assert set(df.columns) == expected


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hourly(tmp_path / "does_not_exist.csv")
