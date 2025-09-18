"""Data acquisition and loading for surecast.

Downloads the UCI Bike Sharing dataset and loads ``hour.csv`` into a frame
indexed by a *complete* hourly ``DatetimeIndex``. The raw series is missing
~141 hours; :func:`load_hourly` reindexes onto a gap-free hourly grid so that
positional shifts in :mod:`surecast.features` correspond exactly to wall-clock
hours (a lag of ``shift(24)`` is genuinely 24 hours, never fewer).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from surecast import schema

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

# Integer-valued columns kept from hour.csv. Stored as pandas nullable Int64 so
# that reindex-inserted gap rows can carry <NA> while the columns remain an
# integer dtype (a plain numpy int cannot hold NaN).
_INT_COLUMNS = ["weathersit", "workingday", "holiday", "season"]

# Float-valued weather columns; gap rows carry NaN.
_FLOAT_COLUMNS = ["temp", "atemp", "hum", "windspeed"]

# Every column returned by load_hourly, in a stable order.
_KEPT_COLUMNS = [schema.TARGET, *_FLOAT_COLUMNS, *_INT_COLUMNS]


def download_raw(dest_dir: Path) -> Path:
    """Download the dataset zip and extract ``hour.csv`` into ``dest_dir``.

    Skips the network entirely if ``dest_dir/hour.csv`` already exists.

    Args:
        dest_dir: Directory that will hold ``hour.csv`` (created if missing).

    Returns:
        Path to the extracted ``hour.csv``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "hour.csv"
    if target.exists():
        return target

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(DATA_URL)
        response.raise_for_status()

    with (
        zipfile.ZipFile(io.BytesIO(response.content)) as archive,
        archive.open("hour.csv") as source,
    ):
        target.write_bytes(source.read())

    return target


def load_hourly(csv_path: Path) -> pd.DataFrame:
    """Load ``hour.csv`` onto a complete, gap-free hourly grid.

    Builds a tz-naive hourly ``DatetimeIndex`` named :data:`schema.TS` from the
    ``dteday`` date and ``hr`` hour columns, sorts ascending, drops any
    duplicate timestamps, then reindexes onto the full hourly range between the
    first and last observed hours. Hours missing from the raw file become rows
    with ``NaN`` target and ``NaN`` weather.

    Args:
        csv_path: Path to a ``hour.csv`` in the UCI schema.

    Returns:
        Frame indexed by ``ts`` with columns ``cnt`` (float), ``temp``,
        ``atemp``, ``hum``, ``windspeed`` (float) and ``weathersit``,
        ``workingday``, ``holiday``, ``season`` (nullable ``Int64``).
    """
    raw = pd.read_csv(csv_path)

    ts = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"].astype(int), unit="h")
    frame = raw.copy()
    frame[schema.TS] = ts
    frame = frame.set_index(schema.TS)

    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.sort_index()

    frame = frame[_KEPT_COLUMNS]

    # Cast before reindex so real rows carry the right dtype; reindex then only
    # introduces NA values into already-nullable columns.
    frame[schema.TARGET] = frame[schema.TARGET].astype("float64")
    for column in _FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in _INT_COLUMNS:
        frame[column] = frame[column].astype("Int64")

    full_range = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name=schema.TS)
    frame = frame.reindex(full_range)

    return frame
