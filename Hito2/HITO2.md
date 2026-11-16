# Hito 2 — Executed experiments, error analysis and interpretation

IIT414W · NextLap capstone · Benjamin Levit, Martin Olivares
Repository and exact commit: see `README.md` · Run record: `outputs/hito2_run.json`

<!-- How to fill: replace every <...>. Copy numbers ONLY from Hito2/outputs/ (our own run).
     Owner of each section in the comment under its heading. -->

## 0. Hypothesis carried from Hito 1
<!-- OWNER: Martin. 2-3 lines. Any change since Hito 1 and its reason (or "no change"). -->

Recent pace variability (`recent_std_s`, std of up to k clean laps through t, within driver/race/stint, plus `short_history`) adds information not captured by `recent_mean_s`. <No change to the hypothesis / Change: ... because ...>. Planned search k ∈ {3, 5}, as declared in Hito 1.

## 1. Evaluation setup and availability at t (C1)
<!-- OWNER: Benjamin. Source: build_features.py, outputs/feature_checks.json, hito2_run.json -->

| Item | Evidence |
| --- | --- |
| Split | Train R1–7 (5,772 pairs), validation R8–9 (1,808: R8 1,066, R9 742); printed and asserted by `run_hito2.py` |
| Training-only fitting | <where in run_hito2.py the Pipeline is fitted on train only> |
| Feature construction | <1–3 sentences: raw laps, clean-lap rule, group by driver/race/stint, backward rolling window including t, before the clean-pair filter> |
| Short-history rule | <fewer than 2 clean laps -> std = 0, short_history = 1; coverage k=3: train <456>, validation <148>; k=5: train <...>, validation <...>> |
| Common row identities | <7,580 pair_ids, 0 missing, 0 duplicated, 0 NaN; validation IDs identical to Hito 1 list> |

**Availability checks written in Hito 1, executed** (`outputs/feature_checks.json`):

| Check | Failure condition (declared in Hito 1) | Observed |
| --- | --- | --- |
| Time rule: delete all laps after t and recompute (200 pairs, seed 414) | Any value changes | <values_changed = ...> |
| Last observation used ≤ end of lap t | Any pair uses a later lap | <...> |
| Stint boundaries: first lap of each stint has no history | Window crosses a stint | <...> |
| Timeline matches the official one (rolling mean reproduces `recent_mean_s`) | Difference > 1e-9 s | <max diff = ...> |
| Manual check of one pair | Hand-computed std ≠ CSV value | <pair_id, 3 lap times, hand std, CSV std> |

## 2. Executed comparison and ablation (C2, C3)
<!-- OWNER: Martin. Source: outputs/comparison.csv, outputs/validation_predictions.csv -->

| Version | Model and inputs | Validation n | MAE (s) | Output or run |
| --- | --- | --- | --- | --- |
| OLS reference | Six published inputs | 1,808 | <...> | `comparison.csv`, column `OLS reference = matched control` |
| Candidate | OLS + `recent_std_s_k3` + `short_history_k3` | 1,808 | <...> | `comparison.csv` |
| Matched control | Same OLS, same procedure and rows, change removed (**same run as the reference**) | 1,808 | <...> | shared run |

Control MAE − candidate MAE = <...> s (positive = lower candidate MAE on these rows).

Other trials (all on the same rows): k = 5 -> MAE <...> s, difference <...> s. <Persistence <...> s for context.>

**Reading.** <What this supports about the hypothesis, in one or two sentences.>
**Decision.** <Keep / discard the change, and why, citing the numbers.>
**Limit of the claim.** <Only clean same-stint pairs, only R8–R9, k chosen on validation (selection evidence), not the reserved test.>

## 3. Errors by context (C4)
<!-- OWNER: Martin. Source: outputs/mae_by_race.csv, shap_local_case.json, mae_by_recent_count.csv -->

| Race | n | MAE control (s) | MAE candidate (s) | Control − candidate (s) |
| --- | --- | --- | --- | --- |
| R8 Austria | 1,066 | <...> | <...> | <...> |
| R9 Great Britain | 742 | <...> | <...> | <...> |
| Pooled (n-weighted) | 1,808 | <...> | <...> | <...> |

<Does the race pattern agree with the pooled result? 1–2 sentences.>

**Inspected validation case.** `pair_id` <...> (selection rule: <...>): observed <...> s, candidate prediction <...> s, absolute error <...> s; matched control prediction <...> s, absolute error <...> s.

**Diagnosis and next check.** <Concrete limitation suggested by the race table and the case, and a check that could investigate it, e.g. MAE by recent_count (outputs/mae_by_recent_count.csv) with n.>

## 4. Local SHAP explanation (C5)
<!-- OWNER: Benjamin. Source: outputs/shap_local_case.json, hito2_run.json -->

| Item | Value |
| --- | --- |
| Explained model | <OLS + pace variability k=3> — why: <chosen model after the ablation> |
| Explainer and background | Exact linear SHAP (coef × (x − training mean)), background = all 5,772 training rows; cross-checked with `shap.LinearExplainer` <version, max diff> |
| Row | `pair_id` <...> |
| Predicted time | <...> s |
| Base value | <...> s |

| Input | Contribution (s) |
| --- | --- |
| <input> | <+/-...> |
| <input> | <+/-...> |
| ... | ... |
| **Base + contributions** | <...> s (= prediction within 1e-6 s) |

**Meaning.** <Interpret one contribution relative to the base in seconds, e.g. `recent_std_s_k3` moves this prediction by <+x> s above the base. It is a model association, not a cause.>
**Observed error (separate).** Observed <...> s − prediction <...> s → absolute error <...> s.

## 5. Global SHAP summary (C6)
<!-- OWNER: Martin. Source: outputs/shap_global_mean_abs.csv, shap_encoded_to_original.csv -->

Sample: all 1,808 validation rows (R8–R9). Aggregation: mean absolute SHAP contribution (s) per original input; one-hot columns summed into their original input (`shap_encoded_to_original.csv`). Explainer and background as in Section 4.

| Original input | Mean \|SHAP\| (s) |
| --- | --- |
| <...> | <...> |

**What it suggests.** <...>
**Scope.** <Within this sample and model only; a large mean |SHAP| for recent_mean_s mostly reflects moving from the training average to each circuit's pace level; not a causal effect nor proof of lower error (that comes from the ablation).>

## 6. Reproduction (C7)
<!-- OWNER: Martin -->

See `README.md`: data location and hashes, environment (`requirements.txt`), commands, and the map from each script to each output.

## 7. Each partner's explanation (C8)
<!-- Each writes their own, ~1 minute to read, in their own words. -->

**Benjamin Levit.** <Decision I contributed to> · <result or check that supports it> · <one limitation or next check> · <AI use and how I verified it>.

**Martin Olivares.** <Decision I contributed to> · <result or check that supports it> · <one limitation or next check> · <AI use and how I verified it>.
