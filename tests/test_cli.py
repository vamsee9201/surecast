"""Tests for the surecast CLI (Agent D).

The sibling pipeline modules (``surecast.data`` / ``features`` / ``models`` /
``backtest``) are built concurrently by other agents; the *contract* is the
interface. These tests substitute lightweight fakes for those seams so the CLI
orchestration — argument parsing, metrics.json emission, model-card rendering,
and exit codes — is verified in isolation and without any network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import surecast
from surecast import cli


def _install_fake(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> ModuleType:
    """Register a fake ``surecast.<name>`` module for the duration of a test."""
    module = ModuleType(f"surecast.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, f"surecast.{name}", module)
    monkeypatch.setattr(surecast, name, module, raising=False)
    return module


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def test_fetch_uses_download_raw_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, Path] = {}

    def fake_download_raw(dest_dir: Path) -> Path:
        calls["dest"] = Path(dest_dir)
        out = Path(dest_dir) / "hour.csv"
        out.write_text("instant,cnt\n1,16\n")
        return out

    _install_fake(monkeypatch, "data", download_raw=fake_download_raw)

    data_dir = tmp_path / "data"
    rc = cli.main(["fetch", "--data-dir", str(data_dir)])

    assert rc == 0
    assert calls["dest"] == data_dir
    assert (data_dir / "hour.csv").exists()
    assert "Cached dataset" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def _fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Wire fake data/features/models/backtest modules; return captured args."""
    captured: dict[str, Any] = {}

    X = pd.DataFrame({"lag_168": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="cnt")

    _install_fake(
        monkeypatch,
        "data",
        load_hourly=lambda csv_path: captured.setdefault("csv_path", Path(csv_path))
        or pd.DataFrame({"cnt": [1.0]}),
    )
    _install_fake(monkeypatch, "features", build_features=lambda df: (X, y))
    _install_fake(
        monkeypatch,
        "models",
        SeasonalNaiveForecaster=object,
        GBMForecaster=lambda **kw: object(),
        ConformalForecaster=lambda **kw: object(),
    )

    summary = {
        "SeasonalNaive": {"mae": 50.0, "rmse": 70.0, "smape": 30.0},
        "GBM": {"mae": 40.0, "rmse": 55.0, "smape": 25.0},
        "Conformal": {
            "mae": 42.0,
            "rmse": 57.0,
            "smape": 26.0,
            "coverage": 0.9,
            "width": 120.0,
        },
    }

    def fake_rolling_backtest(
        X_: pd.DataFrame,
        y_: pd.Series,
        forecasters: dict[str, Any],
        n_splits: int = 5,
        alpha: float = 0.1,
    ) -> SimpleNamespace:
        captured["forecaster_names"] = list(forecasters)
        captured["n_splits"] = n_splits
        captured["alpha"] = alpha
        return SimpleNamespace(summary=summary, per_fold=[])

    _install_fake(monkeypatch, "backtest", rolling_backtest=fake_rolling_backtest)
    captured["summary"] = summary
    return captured


def test_backtest_writes_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = _fake_pipeline(monkeypatch)
    out_dir = tmp_path / "artifacts"

    rc = cli.main(
        ["backtest", "--data-dir", str(tmp_path / "data"), "--out", str(out_dir)]
    )
    assert rc == 0

    # The three contract forecasters were requested.
    assert captured["forecaster_names"] == ["SeasonalNaive", "GBM", "Conformal"]
    assert captured["n_splits"] == 5
    assert captured["alpha"] == pytest.approx(0.1)
    # Reads <data-dir>/hour.csv, not something else.
    assert captured["csv_path"] == tmp_path / "data" / "hour.csv"

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())

    assert set(metrics["models"]) == {"SeasonalNaive", "GBM", "Conformal"}
    assert metrics["models"] == captured["summary"]
    assert metrics["n_splits"] == 5
    assert metrics["alpha"] == pytest.approx(0.1)
    assert metrics["target_coverage"] == pytest.approx(0.9)
    assert metrics["n_samples"] == 4

    out = capsys.readouterr().out
    assert "SeasonalNaive" in out and "Coverage" in out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _write_metrics(artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-07-12T00:00:00+00:00",
        "dataset": "UCI Bike Sharing (hour.csv)",
        "n_samples": 17000,
        "n_splits": 5,
        "alpha": 0.1,
        "target_coverage": 0.9,
        "models": {
            "SeasonalNaive": {"mae": 50.0, "rmse": 70.0, "smape": 30.0},
            "GBM": {"mae": 40.0, "rmse": 55.0, "smape": 25.0},
            "Conformal": {
                "mae": 42.0,
                "rmse": 57.0,
                "smape": 26.0,
                "coverage": 0.902,
                "width": 118.0,
            },
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(payload))


def test_report_builds_model_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_metrics(tmp_path / "artifacts")

    rc = cli.main(["report", "--out", "artifacts"])
    assert rc == 0

    card_path = tmp_path / "docs" / "model_card.md"
    assert card_path.exists()
    card = card_path.read_text()
    assert "# surecast — model card" in card
    for model in ("SeasonalNaive", "GBM", "Conformal"):
        assert model in card
    # Calibration readout quotes empirical vs nominal coverage.
    assert "0.902" in card
    assert "Coverage" in card


def test_report_missing_metrics_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # No artifacts/metrics.json written.
    rc = cli.main(["report", "--out", "artifacts"])
    assert rc == 2
    assert not (tmp_path / "docs" / "model_card.md").exists()


# --------------------------------------------------------------------------- #
# dashboard import safety
# --------------------------------------------------------------------------- #
def test_dashboard_imports_cleanly() -> None:
    import importlib

    module = importlib.import_module("surecast.dashboard")
    # Import must not have triggered Streamlit rendering; main is callable.
    assert hasattr(module, "main")
    assert callable(module.main)
