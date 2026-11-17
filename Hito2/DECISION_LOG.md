# Decision log — NextLap capstone

Members: Benjamin Levit, Martin Olivares
Hito 1 log (as submitted): `Hito1/DECISION_LOG.md`. This file continues it for Hito 2.

## Hito 1 summary (carried forward)

- Reference: OLS with the 6 released inputs (MAE 0.3867 s on 1,808 validation pairs), reproduced in our own run.
- Primary hypothesis: recent pace variability (`recent_std_s` + `short_history` flag), search k ∈ {3, 5}, OLS only, same rows.
- Revision after class peer review: std = 0 **plus** a `short_history` flag for short histories.

## Hito 2 decisions

| Date | Decision | Evidence | Who |
| --- | --- | --- | --- |
| 2026-11-16 | Keep the Hito 1 hypothesis and search unchanged (k ∈ {3, 5}, OLS) | The train-only exploration (E0 in `TRIAL_LOG.csv`) also pointed to variability; no reason to change the plan | Both |
| 2026-11-16 | Build the feature from the raw laps, within driver/race/stint, before the clean-pair filter | Required by the brief; checked by the truncation test (0 values changed) and by reproducing the official `recent_mean_s` (max diff ≈ 1.4e-14 s) — `outputs/feature_checks.json` | Benjamin |
| 2026-11-16 | Use the OLS reference as the matched control (same run) | Candidate = same OLS, Pipeline, training rows and validation IDs plus the two added inputs; the brief allows the reference to serve as the control in this case | Martin |
| 2026-11-16 | **Keep the change with k = 3; discard k = 5** | k = 3: control − candidate = +0.0170 s overall, +0.0148 s R8, +0.0201 s R9. k = 5: +0.0036 s overall but −0.0019 s in R8 (`outputs/comparison.csv`, `outputs/mae_by_race.csv`) | Martin |
| 2026-11-16 | Explain the kept model (k = 3) with SHAP; local case = largest-error validation row `2026-R9-23-L6` | Declared selection rule; same case used in the error analysis (Sections 3 and 4) | Both |
| 2026-11-16 | Do **not** test the short-history fallback on validation now; record it as the next check | The candidate is worse for `recent_count = 1` (n = 148, −0.0575 s), but testing more variants on validation would be an unplanned search | Martin |

## Trials

All trials, including unsuccessful ones, are in `TRIAL_LOG.csv`: T0 (reference / control), E0 (train-only exploration by AI), X0 (code test with random values by AI), T1 (k = 3, kept), T2 (k = 5, discarded).

## Changes after checking (procedure revisions)

| Check | Earlier entry | Revised entry / reason |
| --- | --- | --- |
| SHAP cross-check with the `shap` library gave max difference 0.435 s against our exact linear decomposition | `shap.LinearExplainer(reg, Z_tr)` | The library subsamples 100 background rows by default. We now pass a masker with all 5,772 training rows; difference 0 s. Earlier entry kept here; outputs regenerated (`outputs/hito2_run.json` → `shap.cross_check`) |
| Availability checks listed in Hito 1 | Time rule, stint boundaries, same rows, missing data | Kept, and two checks added: last observation used ≤ end of lap t, and reproduction of the official `recent_mean_s`. A manual check of one pair was also added (`2026-R8-1-L10`) |

## AI use (Hito 2)

We used Claude (Anthropic) on 16–17 Nov 2026.

| Use | Representative prompt (translated) | Accepted / changed | How we verified it |
| --- | --- | --- | --- |
| Wrote `build_features.py` (feature + checks) | "How would we build the feature, with all its steps?" | Accepted; read block by block by Benjamin | Truncation test 0 changes; reproduces official `recent_mean_s`; manual std of `2026-R8-1-L10` (73.025, 73.013, 73.197 s → 0.1029 s) equals the CSV value |
| Wrote `run_hito2.py` (comparison, ablation, errors, SHAP) | - | Accepted; SHAP background later corrected | Reference MAE identical to Hito 1 (0.3867 s); asserts on rounds, IDs, n-weighted pooling and base + contributions = prediction; `shap` library cross-check = 0 s |
| Ran the first validation comparison (T1, T2) and a code test with random values (X0) while testing the scripts | - | Results then reproduced by our own run | Same MAE in our run (`outputs/comparison.csv`) |
| Train-only exploration of candidate features on 9 Nov (E0) | "With what we have now, is it possible to improve the MAE?" | Not adopted as evidence; plan unchanged | Logged in `TRIAL_LOG.csv` |


## Contributions (Hito 2)

| Member | What they did | Decision they can explain |
| --- | --- | --- |
| Benjamin Levit | Ran `build_features.py` and `run_hito2.py` on his computer, manual check of one pair, HITO2.md sections 1 and 4, his explanation | Building the feature before the clean-pair filter; the short-history rule; reading the local SHAP case |
| Martin Olivares | Own run of both scripts, HITO2.md sections 0, 2, 3, 5, 6, README, `TRIAL_LOG.csv`, this log, his explanation | Keeping k = 3 and using the reference as the matched control; diagnosis by `recent_count` |

## Next check (before the final report, 22 Nov)

- Short-history fallback: for `short_history_k3 = 1`, use the control prediction; compare by `recent_count` with n on the same rows; log as a new trial and report as selection evidence on validation.
- Check whether large-error laps such as `2026-R9-23-L6` coincide with a track-status change at t+1.
