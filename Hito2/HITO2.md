# Hito 2 — Executed experiments, error analysis and interpretation

IIT414W · NextLap capstone · Benjamin Levit, Martin Olivares
Repository and exact commit: see `README.md` · Run record: `outputs/hito2_run.json`


## 1. Evaluation setup and availability at t (C1)

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


## 3. Errors by context (C4)


## 4. Local SHAP explanation (C5)

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


## 6. Reproduction (C7)

## 7. Each partner's explanation (C8)

