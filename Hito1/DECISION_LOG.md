# Decision log 

Members: Benjamin Levit, Martin Olivares

## Decisions

| Date | Decision | Evidence | Who |
| --- | --- | --- | --- | 
| 2026-11-06 | Use OLS with the 6 released inputs as the fixed reference | Own run: OLS 0.387 s vs. persistence 0.409 s on the same 1,808 rows, better in R8 and R9 | <Name> |
| 2026-11-06 | Primary direction: recent pace history (variability) | Uses only released data; inspectable time rule; no external source needed. Weather was not loaded (`weather_loaded: false` in the manifest), so it would require an extra source and alignment. | Both |
| 2026-11-06 | Reject `next_lap_s` as an input (worksheet proposal) | It is the target itself: leakage, not available at t | <Name> |
| 2026-11-06 | Window k = 3 as the default | Matches `recent_mean_s` (`history_window = 3` in `build_pairs`), so both features describe the same laps | Martin |
| 2026-11-06 | Keep OLS as the only model for the comparison | The rubric and bonus rules require the same model with and without the feature, so a gain comes from the information | Benjamin |


## Changes after feedback

| Feedback | Earlier entry | Revised entry / reason |
| --- | --- | --- |
| Class peer review (06 Nov): std = 0 mixes "stable" and "no history" | Short history → std = 0 | Std = 0 **plus** `short_history` flag. Original entry kept above. |
| Internal review (08 Nov) of the first draft of sections 1 and 3 | Section 1 framed the task as predicting rival pit stops / the race winner; section 3 proposed adding tyre life, compound and stint, evaluated on round 8 only | Section 1 rewritten around `next_lap_s` and the race engineer's decision (the brief states lap time does not establish pit timing). Section 3 changed to pace variability, because tyre life and compound are already reference inputs, and the validation population is rounds 8 and 9 together (1,808 pairs). Tyre × compound interaction kept as a possible later step. |

## AI use

We used Claude on 06–09 Nov 2026.

| Use | Adopted? | How we checked it |
| --- | --- | --- |
| Analysis of the brief, rubric and data package | Partly: used as a checklist | Compared against the brief and rubric text |
| Reproduction script `run_reference.py` (adapted from the course helper `probar_referencia_v1.py`) | Yes | Ran it ourselves (Python 3.13.5, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1); MAE identical to the supplied `reference_metrics.csv`; asserts on counts, rounds and pair_ids pass |
| Choice of direction | Suggestion considered; decision taken by the team | Reasons recorded in the decisions table |

## Contributions

| Member | What they did | Can explain |
| --- | --- | --- |
| Martin Olivares| Repository, own reference run, `run_reference.py`, README, Reproduced reference section | Fit/transform separation, counts, row identity, MAE table |
| Benajmin Levit | Problem framing, hypothesis, experiment plan, trial log | Decision value, clean-pair limit, availability rule, ablation, support/refutation criteria |
| Both | Cross-review of each other's parts; decision log | Every decision above |

## Next steps (before Hito 2, 19 Nov)

- Benajmin Levit: implement `recent_std_s` + `short_history`; time-rule and boundary checks.
- Martin Olivares: experiments A/B/C, MAE by race and by `recent_count` with n, update `TRIAL_LOG.csv`.
