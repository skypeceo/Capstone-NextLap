# Hito 2 — Executed experiments, error analysis and interpretation

IIT414W · NextLap capstone · Benjamin Levit, Martin Olivares
Repository and exact commit: see `README.md` · Run record: `outputs/hito2_run.json`

## 0. Hypothesis carried from Hito 1

Recent pace variability (`recent_std_s`, the standard deviation of up to k clean laps through t, within driver/race/stint, plus a `short_history` flag) adds information not captured by `recent_mean_s`. No change to the hypothesis or to the planned search (k ∈ {3, 5}) since Hito 1. The checking procedure was extended with two checks not listed in Hito 1 (last observation used, and reproduction of the official `recent_mean_s`); the original checks are kept and reported in Section 1.


## 1. Evaluation setup and availability at t (C1)

| Item | Evidence |
| --- | --- |
| Split | Train R1–7 (5,772 pairs), validation R8–9 (1,808: R8 1,066, R9 742); printed and asserted by `run_hito2.py` |
| Training-only fitting | `run_hito2.py`, block "3. Fit (train only)": `make_model('OLS', ...).fit(train[...], train[TARGET])`. Imputers, scaler, one-hot encoder and regression are fitted on training rows only; validation is only passed to `predict` |
| Feature construction | Built by `build_features.py` from the nine raw race files, not from the filtered clean tables. Only clean laps (same rule as `build_pairs`) enter the history. Laps are grouped by season/race/driver/stint, and a backward rolling window of the last k laps up to and including lap t gives the standard deviation. It is built before the retrospective clean-pair filter, so no t+1 information is used |
| Short-history rule | Fewer than 2 clean laps in the window → `recent_std_s = 0` and `short_history = 1`. Coverage k = 3: train 456, validation 148 (equal to the rows with `recent_count = 1`); k = 5: train 366, validation 108. No rows dropped |
| Common row identities | 7,580 pair_ids written (5,772 + 1,808): 0 missing, 0 duplicated, 0 NaN (`feature_checks.json` → `integrity`). Joins in `run_hito2.py` are one-to-one and keep the row counts; validation IDs identical to the frozen Hito 1 list |

**Availability checks written in Hito 1, executed** (`outputs/feature_checks.json`):

| Check | Failure condition (declared in Hito 1) | Observed |
| --- | --- | --- |
| Time rule: delete all laps after t and recompute (200 pairs, seed 414) | Any value changes | 0 values changed |
| Last observation used ≤ end of lap t | Any pair uses a later lap | 0 pairs (k = 3 and k = 5) |
| Stint boundaries: first lap of each stint has no history | Window crosses a stint | 0 first laps with history (k = 3 and k = 5) |
| Timeline matches the official one: our rolling mean reproduces the released `recent_mean_s` | Difference > 1e-9 s | max difference ≈ 1.4e-14 s over 7,580 pairs |
| Manual check of one pair | Hand-computed std ≠ CSV value | `2026-R8-1-L10`: laps 8, 9, 10 (all clean, stint 1) = 73.025, 73.013, 73.197 s; sample std computed by hand in Excel (`=STDEV.S`) = 0.1029 s; `features_pace.csv` = 0.1029 s → match |

The last two checks were not listed in Hito 1. We added them to show that our timeline is the same as the official one and that a value can be reproduced by hand; the original checks are kept.

## 2. Executed comparison and ablation (C2, C3)

| Version | Model and inputs | Validation n | MAE (s) | Output or run |
| --- | --- | --- | --- | --- |
| OLS reference | Six published inputs | 1,808 | 0.3867 | `outputs/comparison.csv`, `outputs/validation_predictions.csv` column `OLS reference = matched control` |
| Candidate | OLS + `recent_std_s_k3` + `short_history_k3` | 1,808 | 0.3697 | same files, column `OLS + pace variability k=3` |
| Matched control | Same OLS, same Pipeline, training rows and validation IDs, with the two added inputs removed and refitted | 1,808 | 0.3867 | **same run as the OLS reference** |

**Control MAE − candidate MAE = 0.3867 − 0.3697 = +0.0170 s** (positive: the candidate has lower MAE on these 1,808 rows, ≈4.4% of the control MAE).

Other trials on the same rows and protocol: k = 5 → MAE 0.3831 s (difference +0.0036 s). Persistence, for context: 0.4093 s. Both trials are logged in `TRIAL_LOG.csv`.

**Reading.** With everything else fixed (OLS, preprocessing fitted on training only, the same validation `pair_id`s), adding the variability of the last 3 clean laps lowers the validation MAE by 0.017 s. This supports the hypothesis that recent stability carries information that the recent mean alone does not.
**Decision.** Keep the change with k = 3. It beats both the control and k = 5 overall, and it improves both races (Section 3), while k = 5 is worse than the control in R8.
**Limit of the claim.** The gain is measured only on clean same-stint pairs of two races (R8–R9). k was chosen on this same validation set, so this is selection evidence, not an independent estimate; the reserved final data were not used. It does not show that variability causes slower laps, and it does not hold for every subgroup (Section 3).


## 3. Errors by context (C4)

| Race | n | MAE control (s) | MAE candidate (s) | Control − candidate (s) |
| --- | --- | --- | --- | --- |
| R8 Austria | 1,066 | 0.3736 | 0.3588 | +0.0148 |
| R9 Great Britain | 742 | 0.4056 | 0.3854 | +0.0201 |
| Pooled (n-weighted) | 1,808 | 0.3867 | 0.3697 | +0.0170 |

Pooled values are the row-count-weighted means of the race values, e.g. candidate (0.3588 × 1,066 + 0.3854 × 742) / 1,808 = 0.3697 s, equal to the pooled MAE in `comparison.csv` (checked by an assert in `run_hito2.py`). The race pattern agrees with the pooled result: the candidate improves in both races, slightly more in R9, the harder race. For k = 5 the races disagree (R8 −0.0019 s, R9 +0.0116 s), which is one reason to keep k = 3.

**Inspected validation case.** `2026-R9-23-L6` (selection rule: largest absolute error of the candidate on validation). Observed 105.128 s, candidate prediction 96.874 s, absolute error |105.128 − 96.874| = 8.254 s. The matched control predicted 97.303 s (error 7.825 s), so on this lap the candidate is 0.43 s worse.

**Diagnosis.** This lap was about 8 s slower than anything the inputs through lap t suggest. No input observable at t (pace, variability, tyres) anticipates a sudden event in lap t+1, and the added feature does not help on such outliers. A second limitation appears in `outputs/mae_by_recent_count.csv`:

| `recent_count` | n | MAE control (s) | MAE candidate (s) | Control − candidate (s) |
| --- | --- | --- | --- | --- |
| 1 (short history, std = 0 + flag) | 148 | 0.3959 | 0.4534 | −0.0575 |
| 2 | 173 | 0.4233 | 0.4310 | −0.0077 |
| 3 (full window) | 1,487 | 0.3815 | 0.3543 | +0.0272 |

The whole gain comes from laps with a full 3-lap window. On short histories the candidate is worse, so our short-history rule (std = 0 plus a single flag) does not work well for them.

**Next check.** Test a short-history rule decided before looking at the result again: for rows with `short_history_k3 = 1`, use the matched control's prediction (a declared fallback), and compare by `recent_count` with n on the same rows. If we try it, it will be logged as a new trial and reported as selection evidence on validation. For outliers like `2026-R9-23-L6`, a next check is whether such laps coincide with track-status changes or rain at t+1, which would mean they are outside what any input at t can predict.

## 4. Local SHAP explanation (C5)

| Item | Value |
| --- | --- |
| Explained model | OLS + pace variability k = 3, the version kept after the ablation (Section 2) |
| Explainer and background | Exact linear SHAP for the fitted OLS: contribution = coefficient × (encoded value − mean over all 5,772 training rows); one-hot columns summed into their original input (`shap_encoded_to_original.csv`). Cross-checked with `shap.LinearExplainer` 0.54.1 using all training rows as background: max difference 0 s (`hito2_run.json`) |
| Row | `pair_id` 2026-R9-23-L6 (validation, R9; the same case inspected in Section 3) |
| Predicted time | 96.874 s |
| Base value | 86.734 s (model prediction at the training mean) |

| Input | Contribution (s) |
| --- | --- |
| `recent_mean_s` | +6.366 |
| `current_lap_s` | +4.103 |
| `recent_std_s_k3` | −0.342 |
| `short_history_k3` | +0.043 |
| `current_compound` | +0.043 |
| `current_lap_number` | −0.033 |
| `current_team` | −0.027 |
| `current_tyre_life` | −0.013 |
| **Base + contributions** | 86.734 + 10.140 = 96.874 s (equal to the prediction within 1e-6 s) |

Source: `outputs/shap_local_case.json`.

**Meaning.** Most of the prediction comes from pace: `recent_mean_s` and `current_lap_s` raise it by about 10.5 s above the base, because Great Britain laps (≈95 s) are slower than the training average. The new input `recent_std_s_k3` lowers this prediction by 0.342 s relative to the base. In this lap the standard deviation of the last 3 clean laps was 0.98 s, more than twice the training average (0.39 s), and the fitted model associates high recent variability with a slightly faster next lap (an isolated slow lap tends not to repeat). This is an association inside the model, not a cause of the lap time.

**Observed error (separate).** Observed 105.128 s − predicted 96.874 s → absolute error 8.254 s. The real lap was much slower than any input at t suggested, so on this lap the variability contribution moved the prediction in the wrong direction (Section 3 discusses this case).

## 5. Global SHAP summary (C6)

**Sample:** all 1,808 validation rows (R8–R9). **Model:** OLS + pace variability k = 3 (the model kept after the ablation). **Explainer:** exact linear SHAP for the fitted OLS, contribution = coefficient × (encoded value − mean over all 5,772 training rows), cross-checked with `shap.LinearExplainer` using all training rows as background (`outputs/hito2_run.json` → `shap.cross_check`). **Aggregation:** mean absolute contribution in seconds per original input; one-hot columns of `current_compound` and `current_team` are summed into their original input (`outputs/shap_encoded_to_original.csv`).

| Original input | Mean \|SHAP\| (s) |
| --- | --- |
| `recent_mean_s` | 7.405 |
| `current_lap_s` | 4.273 |
| `recent_std_s_k3` | 0.155 |
| `short_history_k3` | 0.080 |
| `current_team` | 0.064 |
| `current_compound` | 0.031 |
| `current_lap_number` | 0.022 |
| `current_tyre_life` | 0.008 |

**What it suggests.** The two pace inputs dominate. Their large contributions mostly move the prediction from the training average (base 86.73 s) to the pace level of each circuit (Austria ≈ 73 s, Great Britain ≈ 95 s). Among the remaining inputs, the new variability feature has the largest average contribution (≈0.15 s), ahead of team, compound and tyre life.
**Scope.** This ranking describes how this fitted model builds its predictions on these 1,808 validation rows. It is not a causal effect of any input, it may differ in other races or populations, and a large contribution is not proof of lower error. The evidence that the new feature improves accuracy comes from the ablation (Section 2), not from SHAP.


## 6. Reproduction (C7)
See `README.md`: data location and hashes, environment (`requirements.txt`), the two commands (`python Hito2/build_features.py`, then `python Hito2/run_hito2.py --explain k3`), and the map from each section of this report to the script and saved output that generate it. The run record (commit at run time, command, environment, input hashes, SHAP method and cross-check) is saved in `outputs/hito2_run.json`.

## 7. Each partner's explanation (C8)
**Benjamin Levit.** Decision: building the feature from the raw laps before the clean-pair filter / the std = 0 + flag rule.
Evidence: the truncation check changed 0 of 200 values / my hand calculation for 2026-R8-1-L10 (0.1029 s) matched the CSV.
Limitation or next check: on short histories (recent_count = 1, n = 148) the candidate is 0.058 s worse, so the flag rule should be revised.
AI: Claude helped with the feature code and how verified it with hand calculation and reading the truncation check.

**Martin Olivares.** I decided to keep k = 3 rather than k = 5, and to treat the OLS reference as the matched control so that only the added information changes. The ablation supports this: the control − candidate difference is +0.0170 s with k = 3 versus +0.0036 s with k = 5, and k = 3 improves both races (R8 +0.0148 s, R9 +0.0201 s) while k = 5 is worse in R8. A limitation is that the gain comes only from laps with a full 3-lap history: with `recent_count = 1` (n = 148) the candidate is 0.058 s worse, so the next check is a declared fallback to the control for short histories. AI (Claude) wrote the experiment script; I verified it by reproducing the Hito 1 reference MAE exactly, checking the n-weighted pooling, and by detecting and fixing a SHAP background mismatch (0.435 s → 0 s).
