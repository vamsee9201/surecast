# Review process & findings

surecast's eight modules were built in parallel by four agents against
`docs/CONTRACT.md`, integrated, then put through three adversarial reviewers,
each required to **reproduce** any defect (run code and data, not just read) before
reporting it. The three mandates were the real risks of a forecasting project:

1. **Leakage & temporal correctness** — the top risk.
2. **Conformal coverage & metric correctness** — is the uncertainty actually calibrated?
3. **General correctness & robustness.**

## Outcome: no defects survived reproduction

Unlike a typical review, this one came back clean — which for a leakage-prone
forecasting pipeline is the point, not an anticlimax. What each reviewer
*reproduced* (so the clean result is evidence, not assertion):

**Leakage (reproduced on the real 15,439-row dataset):**
- Spiking `cnt[p]` changes **zero** feature rows in the forbidden window `[p, p+23]`; the effect first appears at `t = p+24` via `lag_24`, exactly as intended.
- A `fit` spy confirms the conformal quantile models train on only `n − n_calib` rows — the calibration tail is never seen during quantile training.
- All five `TimeSeriesSplit` folds satisfy `train.max() < test.min()`; a fresh forecaster is built per fold (no state bleed).

**Calibration (reproduced):**
- The CQR conformity score `E_i = max(q_lo − y, y − q_hi)` and correction quantile `k = ⌈(n+1)(1−α)⌉` match Romano et al. (2019) for finite-sample validity.
- Empirical coverage over 40 seeds (fit n=3000, test n=1500) = **0.900** on exchangeable data, and 0.901 on a heteroskedastic demand-like generator — nominal, confirming the math.
- `pinball_loss` matches `sklearn.metrics.mean_pinball_loss` exactly at q ∈ {0.05, 0.5, 0.95}; `smape` handles the both-zero case; `coverage` is inclusive on both endpoints.

**Correctness (reproduced):**
- SHAP local additivity: `base + Σ shap − predict = 0.000000` for both the GBM and the conformal median model.
- CLI exit codes and `metrics.json` schema verified; the dashboard imports with zero top-level Streamlit side effects.
- Determinism: predictions, conformal correction, and intervals are bit-identical across independent fits at `random_state=0`.

## Note on the run

The parallel build needed **no** integration fixes — the contract was tight
enough that the four independently-built module sets composed on the first try
(66 tests green, a real backtest where the ML models beat the seasonal-naive
baseline by ~23% MAE). Every result above was re-verified by hand outside the
swarm before shipping.

The single honest caveat the review surfaced — conformal coverage of 0.875 on
the real backtest vs. 0.90 nominal — is a property of temporal
non-exchangeability, not a code defect (see the calibration section of the
[model card](model_card.md)), and it is documented rather than hidden.
