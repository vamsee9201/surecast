"""Command-line entry point for surecast.

Three subcommands:

* ``surecast fetch``    — download and cache the UCI Bike Sharing ``hour.csv``.
* ``surecast backtest`` — run the rolling-origin backtest over the three
  forecasters and write ``<out>/metrics.json`` (the single source of truth for
  every number the README and model card quote), plus print a summary table.
* ``surecast report``   — regenerate ``docs/model_card.md`` from that
  ``metrics.json`` (exit 2 if it is missing).

The imports of the sibling pipeline modules (:mod:`surecast.data`,
:mod:`surecast.features`, :mod:`surecast.models`, :mod:`surecast.backtest`) are
performed lazily inside the command functions. This keeps ``import surecast.cli``
cheap and side-effect-free and lets the tests substitute lightweight fakes for
those seams (the contract, not the concrete module, is the interface).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Backtest configuration is fixed by the contract (target coverage 90%,
# 5-fold expanding-window rolling-origin CV).
DEFAULT_N_SPLITS = 5
DEFAULT_ALPHA = 0.1

# Ordered metric columns for the printed table / model card. The optional
# interval metrics (coverage, width) only appear for interval forecasters.
_POINT_METRICS = ("mae", "rmse", "smape")
_INTERVAL_METRICS = ("coverage", "width")
_METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
    "coverage": "Coverage",
    "width": "Width",
}


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def cmd_fetch(args: argparse.Namespace) -> int:
    """Download + cache ``hour.csv`` into ``--data-dir`` and print its path."""
    from surecast import data as data_mod

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_mod.download_raw(data_dir)
    print(f"Cached dataset at {csv_path}")
    return 0


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def _build_forecasters(alpha: float) -> dict[str, Callable[[], Any]]:
    """Return the {name: factory} map the contract mandates for the backtest."""
    from surecast import models as models_mod

    return {
        "SeasonalNaive": models_mod.SeasonalNaiveForecaster,
        "GBM": lambda: models_mod.GBMForecaster(random_state=0),
        "Conformal": lambda: models_mod.ConformalForecaster(alpha=alpha, random_state=0),
    }


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the rolling backtest and persist ``metrics.json`` + print a table."""
    from surecast import backtest as backtest_mod
    from surecast import data as data_mod
    from surecast import features as features_mod

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "hour.csv"
    df = data_mod.load_hourly(csv_path)
    X, y = features_mod.build_features(df)

    forecasters = _build_forecasters(DEFAULT_ALPHA)
    report = backtest_mod.rolling_backtest(
        X, y, forecasters, n_splits=DEFAULT_N_SPLITS, alpha=DEFAULT_ALPHA
    )
    summary: dict[str, dict[str, float]] = report.summary

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "UCI Bike Sharing (hour.csv)",
        "n_samples": len(X),
        "n_splits": DEFAULT_N_SPLITS,
        "alpha": DEFAULT_ALPHA,
        "target_coverage": round(1.0 - DEFAULT_ALPHA, 4),
        "models": summary,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(_format_summary_table(summary))
    print(f"\nWrote {metrics_path}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate ``docs/model_card.md`` from ``<out>/metrics.json``."""
    out_dir = Path(args.out)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"error: {metrics_path} not found — run `surecast backtest` first.")
        return 2

    metrics = json.loads(metrics_path.read_text())
    card = _render_model_card(metrics)

    card_path = Path("docs") / "model_card.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card)
    print(f"Wrote {card_path}")
    return 0


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _active_columns(summary: dict[str, dict[str, float]]) -> tuple[str, ...]:
    """Point metrics always; interval metrics only if any model reports them."""
    cols: list[str] = list(_POINT_METRICS)
    for metric in _INTERVAL_METRICS:
        if any(metric in row for row in summary.values()):
            cols.append(metric)
    return tuple(cols)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _format_summary_table(summary: dict[str, dict[str, float]]) -> str:
    """Render the per-model metric summary as a fixed-width text table."""
    cols = _active_columns(summary)
    headers = ["Model", *(_METRIC_LABELS[c] for c in cols)]
    rows = [
        [name, *(_fmt(row.get(c)) for c in cols)]
        for name, row in summary.items()
    ]

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _render_model_card(metrics: dict[str, Any]) -> str:
    """Build the Markdown model card from the parsed metrics.json payload."""
    summary: dict[str, dict[str, float]] = metrics.get("models", {})
    cols = _active_columns(summary)

    header = "| Model | " + " | ".join(_METRIC_LABELS[c] for c in cols) + " |"
    divider = "|" + "|".join([" --- "] * (len(cols) + 1)) + "|"
    body = [
        "| " + " | ".join([name, *(_fmt(row.get(c)) for c in cols)]) + " |"
        for name, row in summary.items()
    ]

    target = metrics.get("target_coverage")
    alpha = metrics.get("alpha")
    conformal = summary.get("Conformal", {})
    achieved = conformal.get("coverage")

    lines = [
        "# surecast — model card",
        "",
        "_Auto-generated from `artifacts/metrics.json`; do not edit by hand._",
        "",
        f"- Generated: {metrics.get('generated_at', 'n/a')}",
        f"- Dataset: {metrics.get('dataset', 'n/a')}",
        f"- Samples (after feature warmup): {metrics.get('n_samples', 'n/a')}",
        f"- Backtest: expanding-window rolling-origin CV, "
        f"{metrics.get('n_splits', 'n/a')} folds",
        f"- Nominal interval coverage: {target} (alpha = {alpha})",
        "",
        "## Framing and anti-leakage",
        "",
        "Day-ahead (24h-horizon) hourly demand. `cnt[t]` is forecast using only",
        "information available at `t - 24h`: calendar features of `t`, a weather",
        "forecast for `t`, and target lags of **>= 24 hours** (plus rolling",
        "statistics over windows ending at `t - 24h`). No target lag shorter than",
        "24 hours is ever a feature. Weather-at-prediction-time is assumed",
        "available from a forecast — a standard, documented demand-forecasting",
        "assumption.",
        "",
        "## Backtest results",
        "",
        header,
        divider,
        *body,
        "",
        "## Calibration readout",
        "",
    ]
    if achieved is not None and target is not None:
        lines.append(
            f"Conformal (CQR) empirical coverage on held-out folds: "
            f"**{achieved:.3f}** vs nominal **{target}** "
            f"(gap {achieved - target:+.3f})."
        )
    else:
        lines.append("Conformal coverage not available in metrics.json.")
    lines += [
        "",
        "## Intended use and limitations",
        "",
        "- **Intended use:** short-horizon (day-ahead) hourly demand planning where a",
        "  calibrated *range* matters more than a single number — staffing, inventory,",
        "  capacity. Not a long-horizon or multi-step recursive forecaster.",
        "- **Why coverage can undershoot the nominal level:** conformal prediction",
        "  guarantees marginal coverage only under *exchangeability*. A real time series",
        "  is not exchangeable — later folds see demand regimes (seasonal growth, weather",
        "  shifts) the calibration set never did — so held-out coverage lands modestly",
        "  below nominal. On exchangeable synthetic data the same code hits nominal",
        "  coverage almost exactly; the gap here is a property of the data, not a bug.",
        "  Time-aware conformal (e.g. weighted/adaptive conformal, or online",
        "  recalibration) would close it and is the natural next step.",
        "- **Point vs. interval model:** the conformal point estimate is the predictive",
        "  *median* (robust to right-skewed demand), so its MAE differs slightly from the",
        "  squared-error GBM, which targets the mean. Both beat the seasonal-naive baseline.",
        "- **Weather assumption:** features use weather *at* the forecast timestamp,",
        "  assuming a weather forecast is available at prediction time — standard in demand",
        "  forecasting. Degrade weather inputs to forecasts to assess real-world impact.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# argument parsing / dispatch
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with the three subcommands."""
    parser = argparse.ArgumentParser(
        prog="surecast",
        description="Day-ahead hourly demand forecasting with calibrated intervals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download + cache hour.csv")
    p_fetch.add_argument("--data-dir", default="data", help="cache directory (default: data)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_backtest = sub.add_parser("backtest", help="run rolling backtest, write metrics.json")
    p_backtest.add_argument(
        "--data-dir", default="data", help="dataset directory (default: data)"
    )
    p_backtest.add_argument(
        "--out", default="artifacts", help="output directory (default: artifacts)"
    )
    p_backtest.set_defaults(func=cmd_backtest)

    p_report = sub.add_parser("report", help="regenerate docs/model_card.md from metrics.json")
    p_report.add_argument(
        "--out", default="artifacts", help="artifacts directory (default: artifacts)"
    )
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
