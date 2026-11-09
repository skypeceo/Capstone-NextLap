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

## 5. Traceability and next steps

