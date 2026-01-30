"""Streamlit dashboard for surecast.

Run with::

    streamlit run src/surecast/dashboard.py

It loads the cached dataset and the ``artifacts/metrics.json`` produced by
``surecast backtest`` and renders:

1. Recent actuals + day-ahead forecast with the calibrated interval band.
2. The backtest summary table.
3. A coverage-vs-nominal calibration readout.
4. The global SHAP importances bar chart.

**Import safety:** there are no top-level Streamlit calls. Everything runs
inside :func:`main`, which Streamlit invokes because it executes this file as
``__main__``. This lets the module be imported (e.g. in a smoke test) without a
running server and without side effects. The sibling pipeline modules are
imported lazily inside the cached helpers so importing this module never depends
on them being present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    import pandas as pd

# Repo-root-relative defaults. dashboard.py lives at src/surecast/dashboard.py,
# so parents[2] is the project root.
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _ROOT / "data"
DEFAULT_ARTIFACTS_DIR = _ROOT / "artifacts"

# How many trailing hours of history to draw on the forecast chart.
RECENT_WINDOW_HOURS = 24 * 7
# How many hours of day-ahead forecast to show.
FORECAST_HORIZON = 24


# --------------------------------------------------------------------------- #
# cached data loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_metrics(artifacts_dir: str) -> dict[str, Any] | None:
    """Load ``metrics.json`` from ``artifacts_dir``; ``None`` if absent."""
    path = Path(artifacts_dir) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_dataset(data_dir: str) -> pd.DataFrame | None:
    """Load the cached hourly frame; ``None`` if the CSV is missing."""
    from surecast import data as data_mod

    csv_path = Path(data_dir) / "hour.csv"
    if not csv_path.exists():
        return None
    return data_mod.load_hourly(csv_path)


@st.cache_data(show_spinner=False)
def build_xy(data_dir: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Build the ``(X, y)`` design matrix from the cached dataset."""
    from surecast import features as features_mod

    df = load_dataset(data_dir)
    if df is None:
        return None
    return features_mod.build_features(df)


@st.cache_data(show_spinner=False)
def fit_conformal(data_dir: str, alpha: float) -> Any | None:
    """Fit a :class:`ConformalForecaster` on all available rows for the demo."""
    from surecast import models as models_mod

    xy = build_xy(data_dir)
    if xy is None:
        return None
    X, y = xy
    model = models_mod.ConformalForecaster(alpha=alpha, random_state=0)
    model.fit(X, y)
    return model


@st.cache_data(show_spinner=False)
def global_shap(data_dir: str, alpha: float) -> pd.Series | None:
    """Return mean(|SHAP|) importances for the fitted conformal point model."""
    from surecast import explain as explain_mod

    model = fit_conformal(data_dir, alpha)
    xy = build_xy(data_dir)
    if model is None or xy is None:
        return None
    X, _ = xy
    return explain_mod.global_importance(model, X)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def _render_forecast(data_dir: str, alpha: float) -> None:
    st.subheader("Recent actuals + day-ahead forecast")
    xy = build_xy(data_dir)
    model = fit_conformal(data_dir, alpha)
    if xy is None or model is None:
        st.info("No cached dataset. Run `surecast fetch` then `surecast backtest`.")
        return

    import matplotlib.pyplot as plt

    X, y = xy
    recent = y.iloc[-RECENT_WINDOW_HOURS:]
    x_fore = X.iloc[-FORECAST_HORIZON:]
    point = model.predict(x_fore)
    lower, upper = model.predict_interval(x_fore)
    fore_idx = x_fore.index

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(recent.index, recent.to_numpy(), color="#3b6ea5", lw=1.6, label="actual")
    ax.plot(fore_idx, point, color="#e07b39", lw=2.0, label="forecast")
    ax.fill_between(
        fore_idx, lower, upper, color="#e07b39", alpha=0.22,
        label=f"{(1 - alpha) * 100:.0f}% calibrated interval",
    )
    ax.axvline(fore_idx[0], color="#999", ls="--", lw=0.8)
    ax.set_ylabel("rentals / hour")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Shaded band is the calibrated {(1 - alpha) * 100:.0f}% prediction interval "
        "(conformalized quantile regression). Dashed line marks the forecast origin."
    )


def _render_backtest_table(metrics: dict[str, Any] | None) -> None:
    st.subheader("Backtest summary")
    if not metrics or "models" not in metrics:
        st.info("No `metrics.json`. Run `surecast backtest`.")
        return

    import pandas as pd

    table = pd.DataFrame(metrics["models"]).T
    st.dataframe(table)


def _render_calibration(metrics: dict[str, Any] | None) -> None:
    st.subheader("Calibration: coverage vs nominal")
    if not metrics or "models" not in metrics:
        st.info("No `metrics.json`.")
        return

    target = metrics.get("target_coverage")
    conformal = metrics["models"].get("Conformal", {})
    achieved = conformal.get("coverage")
    if achieved is None or target is None:
        st.info("Conformal coverage not present in metrics.json.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Nominal", f"{target:.2%}")
    col2.metric("Empirical", f"{achieved:.2%}", delta=f"{achieved - target:+.2%}")
    width = conformal.get("width")
    col3.metric("Mean width", "—" if width is None else f"{width:.1f}")


def _render_shap(data_dir: str, alpha: float) -> None:
    st.subheader("Global SHAP importances")
    importances = global_shap(data_dir, alpha)
    if importances is None:
        st.info("No cached dataset for SHAP.")
        return
    st.bar_chart(importances)


# --------------------------------------------------------------------------- #
# entry point (Streamlit executes this file as __main__)
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the full dashboard. Called only when Streamlit runs this file."""
    st.set_page_config(page_title="surecast", layout="wide")
    st.title("surecast — day-ahead demand forecasting")

    with st.sidebar:
        st.header("Settings")
        data_dir = st.text_input("Data dir", str(DEFAULT_DATA_DIR))
        artifacts_dir = st.text_input("Artifacts dir", str(DEFAULT_ARTIFACTS_DIR))
        alpha = st.slider("alpha (1 - coverage)", 0.01, 0.5, 0.1, 0.01)

    metrics = load_metrics(artifacts_dir)

    _render_forecast(data_dir, alpha)
    st.divider()
    _render_backtest_table(metrics)
    st.divider()
    _render_calibration(metrics)
    st.divider()
    _render_shap(data_dir, alpha)


if __name__ == "__main__":
    main()
