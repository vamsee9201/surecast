# surecast — module contract (source of truth for parallel implementation)

Day-ahead (24h-horizon) hourly bike-demand forecasting with **calibrated**
prediction intervals (conformalized quantile regression) and SHAP explainability,
served through a Streamlit dashboard.

Dataset: UCI Bike Sharing (`hour.csv`, 17,379 rows, 2011–2012). Target: `cnt`
(total rentals that hour). The raw series has ~141 missing hours — the loader
reindexes to a complete hourly grid.

**Framing (binding):** we forecast `cnt[t]` using only information available at
`t - 24h`, EXCEPT weather, which we assume is available from a weather forecast
at prediction time (a standard, documented assumption in demand forecasting —
state it in the model card). Concretely: calendar features of `t`, weather of
`t`, and **target lags of ≥ 24 hours**. No target lag < 24h may ever be a
feature — that is the core anti-leakage rule.

Conventions: Python 3.12, ruff line-length 100 (`E,F,I,UP,B,SIM,RUF`), mypy
`check_untyped_defs`, plain pytest. Type-hint public functions. Deterministic:
any model takes `random_state: int = 0`.

## src/surecast/schema.py  (ALREADY WRITTEN — do not modify)

Column-name constants and the feature list, imported by everyone. See the file.

## src/surecast/data.py  (Agent A)

```python
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

def download_raw(dest_dir: Path) -> Path
    # Download the zip, extract hour.csv to dest_dir/hour.csv, return that path.
    # Skip download if the file already exists. Uses httpx (follow_redirects).

def load_hourly(csv_path: Path) -> pd.DataFrame
    # Read hour.csv; build a tz-naive hourly DatetimeIndex named "ts" from
    # (dteday + hr); REINDEX to a complete hourly range (fill gaps). Returns a
    # frame indexed by ts with columns: cnt (float, gaps -> NaN, dropped later
    # in features), temp, atemp, hum, windspeed, weathersit (int),
    # workingday (int 0/1), holiday (int 0/1), season (int). Sorted ascending,
    # no duplicate timestamps. Missing-hour rows have NaN cnt and NaN weather.
```

Tests `tests/test_data.py`: use a small synthetic hour.csv fixture (write a
DataFrame to CSV in tmp_path — DO NOT hit the network in tests). Assert the
index is a complete hourly range with gaps inserted, dtypes, sorted, cnt NaN on
inserted gap rows.

## src/surecast/features.py  (Agent A)

```python
HORIZON = 24  # hours

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Input: the frame from load_hourly. Output: (X, y) aligned on ts.
    # y = df["cnt"]. X columns (see schema.FEATURES), ALL usable at t-24h:
    #   calendar (known ahead): hour, dayofweek, month, is_weekend, workingday,
    #     holiday, plus cyclical hour_sin/hour_cos.
    #   weather (assumed forecast): temp, hum, windspeed, weathersit.
    #   target lags: lag_24, lag_25, lag_48, lag_168  (all >= 24h).
    #   rolling over PAST, shifted by >= 24h: roll_mean_24 = mean of cnt over the
    #     24h window ENDING at t-24 (i.e. cnt.shift(24).rolling(24).mean());
    #     roll_std_24 likewise. NO feature may reference cnt at t-1..t-23.
    # Rows with any NaN feature or NaN y (from gaps / warmup) are DROPPED, and X
    # and y stay index-aligned. Return X, y with identical DatetimeIndex.
```

Tests `tests/test_features.py`: **leakage guard** — assert every lag/rolling
column at a given ts equals the hand-computed value from cnt at ts-24 or
earlier, and construct a series where cnt[t] is unique so any accidental use of
cnt[t..t-23] would be detectable; assert no NaNs remain; assert X.index == y.index.

## src/surecast/models.py  (Agent B)

```python
class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...            # point forecast

class SeasonalNaiveForecaster:
    # Baseline: predict = X["lag_168"] (same hour last week). fit() is a no-op
    # (stateless). predict returns the lag_168 column as a numpy array. Raises
    # if lag_168 is absent.

class GBMForecaster:
    # sklearn HistGradientBoostingRegressor point model (squared_error).
    # __init__(random_state=0, **hgb_kwargs). Standard fit/predict.

class ConformalForecaster:
    # Conformalized Quantile Regression (Romano et al. 2019) for CALIBRATED
    # intervals. Wraps two HistGradientBoostingRegressor quantile models
    # (loss="quantile") at the lower/upper nominal quantiles plus a median point
    # model.
    def __init__(self, alpha: float = 0.1, calib_fraction: float = 0.2, random_state: int = 0)
        # target coverage = 1 - alpha (default 90%). lower q = alpha/2, upper = 1-alpha/2.
    def fit(self, X, y) -> "ConformalForecaster":
        # TIME-ORDERED split: last calib_fraction of rows (X is pre-sorted by ts)
        # is the calibration set, the rest trains the quantile models. Then CQR:
        # conformity score E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)) on calib;
        # store the ceil((n_calib+1)(1-alpha))/n_calib empirical quantile as the
        # conformal correction. The calibration set MUST NOT be used to train the
        # quantile models (that is the whole point).
    def predict(self, X) -> np.ndarray            # median-quantile point forecast
    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        # returns (lower, upper) = (q_lo - correction, q_hi + correction),
        # both clipped at 0 (demand is non-negative).
```

Tests `tests/test_models.py`: SeasonalNaive returns lag_168; GBM fits/predicts
shape; ConformalForecaster on synthetic data where y = f(x)+noise achieves
empirical coverage ON A HELD-OUT SET within a tolerance of the nominal 1-alpha
(e.g. 0.90 ± 0.07 for n≥2000), and that the calibration rows are excluded from
quantile-model training (check via a spy/monkeypatch on fit sizes). No network,
small n, fast.

## src/surecast/metrics.py  (Agent C)

```python
def mae(y_true, y_pred) -> float
def rmse(y_true, y_pred) -> float
def smape(y_true, y_pred) -> float        # symmetric MAPE in %, 0 when both 0, no div-by-0
def coverage(y_true, lower, upper) -> float          # fraction of y in [lower, upper]
def mean_interval_width(lower, upper) -> float
def pinball_loss(y_true, y_pred, quantile) -> float  # for quantile calibration
```
All accept array-likes, return plain floats. Tests `tests/test_metrics.py`:
hand-computed values incl. smape both-zero edge case and coverage boundaries
(inclusive endpoints).

## src/surecast/backtest.py  (Agent C)

```python
@dataclass
class FoldResult:
    model: str
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None

@dataclass
class BacktestReport:
    per_fold: list[FoldResult]
    summary: dict[str, dict[str, float]]   # {model_name: {mae, rmse, smape, coverage?, width?}}

def rolling_backtest(
    X: pd.DataFrame, y: pd.Series,
    forecasters: dict[str, Callable[[], Forecaster]],
    n_splits: int = 5, alpha: float = 0.1,
) -> BacktestReport:
    # EXPANDING-window rolling-origin CV (sklearn TimeSeriesSplit). For each fold:
    # fit a FRESH forecaster (from the factory) on the train slice, predict the
    # test slice. If the forecaster has predict_interval, collect intervals and
    # include coverage + mean width in its summary. Point metrics (mae/rmse/smape)
    # for all. summary aggregates across folds (mean). NEVER fit on test data.
```
Tests `tests/test_backtest.py`: with a trivial deterministic forecaster, assert
each fold's train indices strictly precede its test indices (no leakage), n_splits
folds produced, summary keys present, interval metrics only for interval models.

## src/surecast/explain.py  (Agent C)

```python
def global_importance(model: GBMForecaster | ConformalForecaster, X: pd.DataFrame, max_samples: int = 500) -> pd.Series:
    # SHAP TreeExplainer on the underlying HGB point model; return mean(|shap|)
    # per feature as a Series indexed by feature name, sorted descending.
def local_explanation(model, X: pd.DataFrame, row: int, max_samples: int = 500) -> pd.Series:
    # signed SHAP values for one row, Series by feature.
```
Tests `tests/test_explain.py`: importances Series covers all feature columns,
non-negative, sorted; local sums (+ base value) ≈ model prediction within tol.
Small synthetic fit, fast.

## src/surecast/cli.py + src/surecast/dashboard.py  (Agent D)

cli.py (argparse, `main(argv=None) -> int`):
```
surecast fetch [--data-dir data]                       # download + cache hour.csv
surecast backtest [--data-dir data] [--out artifacts]  # run rolling_backtest over
    # SeasonalNaive, GBM, Conformal; write artifacts/metrics.json (the numbers the
    # README/model card quote) and print a summary table. Exit 0.
surecast report [--out artifacts]                      # regenerate docs/model_card.md
    # from artifacts/metrics.json (fail with exit 2 if metrics.json missing).
```
dashboard.py: a Streamlit app (`streamlit run src/surecast/dashboard.py`) that
loads cached data + artifacts and shows: (1) recent actuals + forecast with the
calibrated interval band, (2) the backtest summary table, (3) a coverage-vs-nominal
calibration readout, (4) global SHAP importances bar chart. Guard all heavy work
behind `@st.cache_data`. dashboard.py must import cleanly without a running server
(no top-level Streamlit calls outside a `def main()` / `if __name__`-style guard
that Streamlit invokes) so it can be smoke-imported in a test.

Tests `tests/test_cli.py`: `fetch` monkeypatched to a fixture (no network);
`backtest` on a tiny cached CSV writes metrics.json with the expected model keys;
`report` builds model_card.md and exits 2 when metrics.json absent; importing
dashboard does not error.

## Boundaries
- Agent A: data.py, features.py, tests/test_data.py, tests/test_features.py
- Agent B: models.py, tests/test_models.py
- Agent C: metrics.py, backtest.py, explain.py, tests/test_metrics.py, test_backtest.py, test_explain.py
- Agent D: cli.py, dashboard.py, tests/test_cli.py
- Nobody edits pyproject.toml or schema.py or another agent's files.
