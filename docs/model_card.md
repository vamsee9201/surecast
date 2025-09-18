# surecast — model card

_Auto-generated from `artifacts/metrics.json`; do not edit by hand._

- Generated: 2026-07-11T22:19:02.184649+00:00
- Dataset: UCI Bike Sharing (hour.csv)
- Samples (after feature warmup): 15439
- Backtest: expanding-window rolling-origin CV, 5 folds
- Nominal interval coverage: 0.9 (alpha = 0.1)

## Framing and anti-leakage

Day-ahead (24h-horizon) hourly demand. `cnt[t]` is forecast using only
information available at `t - 24h`: calendar features of `t`, a weather
forecast for `t`, and target lags of **>= 24 hours** (plus rolling
statistics over windows ending at `t - 24h`). No target lag shorter than
24 hours is ever a feature. Weather-at-prediction-time is assumed
available from a forecast — a standard, documented demand-forecasting
assumption.

## Backtest results

| Model | MAE | RMSE | sMAPE | Coverage | Width |
| --- | --- | --- | --- | --- | --- |
| Conformal | 49.039 | 78.458 | 32.068 | 0.875 | 201.783 |
| GBM | 42.673 | 67.412 | 31.016 | — | — |
| SeasonalNaive | 55.707 | 94.911 | 34.971 | — | — |

## Calibration readout

Conformal (CQR) empirical coverage on held-out folds: **0.875** vs nominal **0.9** (gap -0.025).

## Intended use and limitations

- **Intended use:** short-horizon (day-ahead) hourly demand planning where a
  calibrated *range* matters more than a single number — staffing, inventory,
  capacity. Not a long-horizon or multi-step recursive forecaster.
- **Why coverage can undershoot the nominal level:** conformal prediction
  guarantees marginal coverage only under *exchangeability*. A real time series
  is not exchangeable — later folds see demand regimes (seasonal growth, weather
  shifts) the calibration set never did — so held-out coverage lands modestly
  below nominal. On exchangeable synthetic data the same code hits nominal
  coverage almost exactly; the gap here is a property of the data, not a bug.
  Time-aware conformal (e.g. weighted/adaptive conformal, or online
  recalibration) would close it and is the natural next step.
- **Point vs. interval model:** the conformal point estimate is the predictive
  *median* (robust to right-skewed demand), so its MAE differs slightly from the
  squared-error GBM, which targets the mean. Both beat the seasonal-naive baseline.
- **Weather assumption:** features use weather *at* the forecast timestamp,
  assuming a weather forecast is available at prediction time — standard in demand
  forecasting. Degrade weather inputs to forecasts to assess real-world impact.
