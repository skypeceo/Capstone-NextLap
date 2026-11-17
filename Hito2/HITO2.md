# Hito 2 — Executed experiments, error analysis and interpretation

IIT414W · NextLap capstone · Benjamin Levit, Martin Olivares
Repository and exact commit: see `README.md` · Run record: `outputs/hito2_run.json`


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


## 3. Errors by context (C4)


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


## 6. Reproduction (C7)

## 7. Each partner's explanation (C8)

