"""Generate the README figures from real data + the fitted model.

Reproducible: `uv run python scripts/make_figures.py` writes
docs/img/forecast.png and docs/img/shap.png. No hand-edited images.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from surecast.data import download_raw, load_hourly
from surecast.explain import global_importance
from surecast.features import build_features
from surecast.models import ConformalForecaster

OUT = Path("docs/img")
DATA = Path("data")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv = download_raw(DATA)
    X, y = build_features(load_hourly(csv))

    # Fit on all but the last 24h; forecast that final day.
    horizon = 24
    model = ConformalForecaster(alpha=0.1, random_state=0).fit(X.iloc[:-horizon], y.iloc[:-horizon])
    x_fore = X.iloc[-horizon:]
    point = model.predict(x_fore)
    lower, upper = model.predict_interval(x_fore)

    recent = y.iloc[-(horizon * 5):]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(recent.index, recent.to_numpy(), color="#3b6ea5", lw=1.6, label="actual")
    ax.plot(x_fore.index, point, color="#e07b39", lw=2.0, label="forecast (median)")
    ax.fill_between(
        x_fore.index, lower, upper, color="#e07b39", alpha=0.22,
        label="90% calibrated interval",
    )
    ax.axvline(x_fore.index[0], color="#999", ls="--", lw=0.8)
    ax.set_ylabel("rentals / hour")
    ax.set_title("Day-ahead forecast with conformalized 90% prediction interval")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "forecast.png", dpi=130)
    plt.close(fig)

    imp = global_importance(model, X.iloc[-2000:]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.to_numpy(), color="#3b6ea5")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(OUT / "shap.png", dpi=130)
    plt.close(fig)

    print(f"wrote {OUT}/forecast.png and {OUT}/shap.png")


if __name__ == "__main__":
    main()
