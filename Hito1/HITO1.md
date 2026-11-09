# Hito 1 — Problem framing, decision value and experiment plan


## 1. Problem and decision value

| Item | Definition |
| --- | --- |
| Target and unit | `next_lap_s`: duration in seconds of the same driver's next lap (t+1). |
| Prediction time | At the end of lap t, using only information observed through lap t. |
| Intended reader | The race engineer on the pit wall. |
| Decision informed | Whether the driver's pace is expected to stay stable or drop. If the actual lap is much slower than predicted, the engineer can flag it early and investigate a possible cause (tyre degradation, a car issue) before any pit discussion. |
| What it cannot establish | When the car should pit, how many positions a strategy gains, or any causal effect of an input. A one-lap time prediction does not decide strategy. |
| Conditional-population limit | Results only hold for **clean same-stint pairs**. `clean_pair` is defined retrospectively with t+1 information (no pit, yellow/red flag or deleted lap at t+1; same stint and compound), so it cannot identify in advance which future laps the model applies to. Pit, out and flag laps are not covered, and the clean-pair MAE is not evidence of performance on all laps or in a live race. |


## 2. Reproduced reference

| Item | Value |
| --- | --- |
| Data version | `IIT414W_Capstone_NextLap_StudentData_v1` (SHA-256 in `outputs/reference_run.json`, matching the course manifest) |
| Split | Train: rounds 1–7 of 2026, 5,772 clean pairs. Validation: rounds 8 (Austria, 1,066) and 9 (Great Britain, 742), 1,808 clean pairs. `pair_id` unique and disjoint (asserted). |
| Inputs | `current_lap_s`, `recent_mean_s`, `current_lap_number`, `current_tyre_life` (numeric); `current_compound`, `current_team` (categorical) |
| Fit/transform | Median imputer + StandardScaler (numeric) and most-frequent imputer + OneHotEncoder (categorical) inside an sklearn Pipeline fitted on training rows only; validation is only transformed. The median baseline is also computed on training. |
| Command | `python run_reference.py` (see `README.md`) |
| Evidence | Own run: `outputs/reference_metrics.csv`, `outputs/validation_reference_predictions.csv` (per `pair_id`), `outputs/validation_pair_ids.txt`, `outputs/reference_run.json` |

MAE in seconds on the same validation rows (own run, 2026-11-09, Python 3.13.5, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1; identical to the instructor-supplied outputs, which used numpy 2.0.2, pandas 2.3.1, scikit-learn 1.8.0):

| Model | All validation (n = 1,808) | Round 8 (n = 1,066) | Round 9 (n = 742) |
| --- | --- | --- | --- |
| Training median | 11.376 | 11.716 | 10.888 |
| Persistence (`next = current_lap_s`) | 0.409 | 0.401 | 0.422 |
| OLS reference | **0.387** | 0.374 | 0.406 |

**Comparison.** The training median fails (≈11.4 s) because lap times range from 73 to 109 s between circuits and one constant ignores the circuit. Persistence is already strong because consecutive clean laps of the same driver in the same stint are very similar. OLS improves on persistence by only ≈0.022 s (≈5.5%), in both rounds, so most information is already in the current lap and the margin for any added feature is small. Round 9 is harder than round 8 for every model, so we always report MAE by race.

## 3. Primary hypothesis

| Item | Definition |
| --- | --- |
| Change | Add **recent pace variability** `recent_std_s`: the standard deviation of up to the last 3 clean lap times through lap t, plus a binary flag `short_history`. The six reference inputs stay unchanged. |
| What differs from the reference | `recent_mean_s` summarises the *level* of recent pace, but not its *stability*. No reference input measures how much the recent laps vary. |
| Mechanism | A driver whose recent laps vary (traffic, degradation, pushing or saving tyres) is less predictable by the mean alone. With the variability, the model can weigh `current_lap_s` and `recent_mean_s` differently for stable and unstable drivers. |
| Source | Released timing snapshots (`data/race_2026_r0*_laps_v2.csv`, FastF1 official timing, SHA-256 in the manifest). No external source, so no external fallback is needed. |
| Availability at t | Lap t has just ended, so its time is known. Only laps with `end_time_s` ≤ the `end_time_s` of lap t are used. Computed within season/round/driver/stint (it never crosses drivers, races or tyre stints), using only clean laps (the same `current_clean` rule as `build_pairs`), on the original lap timeline **before** retrospective clean-pair filtering. |
| Short-history rule | The std needs at least 2 clean laps. With fewer, set `recent_std_s = 0` and `short_history = 1` (otherwise 0). This affects the rows with `recent_count = 1`: 456 in training and 148 in validation. The flag keeps "no history" separate from "perfectly stable". No rows are dropped. |
| Inadmissible inputs | `next_lap_s` and any `next_*` column (target / future information); `clean_pair` (uses t+1). |
| Split (unchanged) | Train on rounds 1–7 (5,772 pairs); validate on rounds 8 and 9 together (1,808 pairs); reserved final data not used. |

## 4. Experiment plan

**Fixed controls.** Everything except the added information stays the same as in Section 2:

| Control | Fixed value |
| --- | --- |
| Target and metric | `next_lap_s`; MAE in seconds |
| Rows | The same 5,772 training and 1,808 validation `pair_id`s (asserted against `outputs/validation_pair_ids.txt`) |
| Split | By race: train rounds 1–7, validation rounds 8–9; validation is never added to training |
| Preprocessing | The same `make_model` Pipeline (imputers, scaler, one-hot), fitted on training only |
| Model | OLS |
| Seed | `RANDOM_SEED = 414` |

**Reference, extension and ablation.**

| Experiment | Information | Model | Rows | Metric | Isolates |
| --- | --- | --- | --- | --- | --- |
| A. Reference | 6 reference inputs | OLS | Common rows | MAE overall + by race, with n | Baseline |
| B. Extension | 6 inputs + `recent_std_s` + `short_history` | OLS | Same rows | Same | Reference + variability |
| C. Without added information | B with `recent_std_s` and `short_history` removed | OLS | Same rows | Same | B − C = contribution of variability |

With OLS, C equals A. Only one thing changes between B and C. If we later try another model family (e.g. Ridge), we also run that same model without the feature, so a model gain is not attributed to the feature.

**Bounded search.** Window k ∈ {3, 5} (k = 3 matches `recent_mean_s`). That makes 2 trials in total, all logged in `TRIAL_LOG.csv`, including unsuccessful ones. No other settings are tuned.

**Selection vs. final evaluation.** Validation (R8–R9) is used only to choose k, and it is reported as selection evidence, not as final performance. The instructor's reserved final data is not accessed and is not used to choose features, settings or models. Prior access: none.

**Checks written before implementing (run in Hito 2).**

| Check | Evidence | Failure condition |
| --- | --- | --- |
| Time rule | Recompute the feature after deleting all laps after t from the raw data | Any pair whose value changes |
| Boundaries | Group by season/round/driver/stint; inspect laps around a pit stop | A window mixing two stints or drivers |
| Same rows | Assert the validation `pair_id` set equals `validation_pair_ids.txt`; re-run the reference | n ≠ 1,808, a missing or duplicated ID, or reference MAE ≠ 0.3867 s |
| Missing data | Count `short_history = 1` in train (expected 456) and validation (expected 148); check all predictions are finite | Different counts, NaN predictions or dropped rows |

**What would support or contradict the hypothesis.**
- Support: B has a lower MAE than A overall **and** in both R8 and R9.
- Contradict: no gain, a gain in only one race, or a gain below ≈0.005 s (too small to tell from noise with only 2 validation races).
- A negative result will be reported as such; it is still a valid outcome.

**Planned robustness checks.** MAE by validation race with n (R8 n = 1,066, R9 n = 742), and MAE by `recent_count` (1, 2, 3) with n, for A and B. This checks whether any gain comes from drivers with full history or is hidden by the global MAE.

## 5. Traceability and next steps

See `DECISION_LOG.md` for decisions, evidence, contributions and AI use.

| Item | Content |
| --- | --- |
| Revision after feedback | Added the `short_history` flag after the class peer review (originally std = 0 only). Our first draft of sections 1 and 3 (pit-stop prediction; tyre columns already in the reference) was revised to the `next_lap_s` framing and the pace-variability hypothesis. |
| Before Hito 2 | Benjamin Levit: implement `recent_std_s` + `short_history` inside the `build_pairs` logic; run the time-rule and boundary checks. Martin Olivares: run experiments A/B/C, MAE by race and by `recent_count` with n, and update `TRIAL_LOG.csv`. |
| Known blocker | Confirm with Ángel (12 Nov) that std = 0 + flag is an acceptable short-history rule and that k = 5 can be computed from the as-of history view. |
| Possible second step | Tyre life × compound interaction or traffic (gap to the car ahead from `end_time_s`), tested separately so it does not mix with this hypothesis. |

