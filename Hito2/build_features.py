"""Hito 2 - build the pace-variability feature (recent_std_s + short_history) and check it.

Hypothesis (Hito 1): the variability of recent clean laps adds information that
recent_mean_s (level only) does not capture.

Rule: for the pair (lap t -> lap t+1), recent_std_s_k = standard deviation of the
clean lap times among the last k laps up to and INCLUDING lap t, within the same
season/round/driver/stint. Built on the ORIGINAL lap timeline, BEFORE the
retrospective clean-pair filter. Fewer than 2 clean laps -> std = 0, short_history = 1.

Output: Hito2/outputs/features_pace.csv (one row per train/validation pair_id) and
        Hito2/outputs/feature_checks.json (the checks written in Hito 1, executed).

Usage (from the repository root, base_NextLap/ next to Hito1/ and Hito2/):
    python Hito2/build_features.py
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'base_NextLap'
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import load_snapshots, KEYS  # noqa: E402

RANDOM_SEED = 414
WINDOWS = (3, 5)
STINT_KEYS = KEYS + ['stint']          # season, round, driver_number, stint
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------------ STEP 1: original laps
# load_snapshots reads the 9 raw races and verifies their SHA-256 against the manifest.
raw, manifest = load_snapshots(BASE)


def add_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature on a lap table. Uses only each row's own lap and EARLIER laps."""
    d = laps.sort_values(KEYS + ['lap_number']).reset_index(drop=True).copy()

    # STEP 2: same 'clean lap' rule as build_pairs (all observed by the end of that lap)
    d['clean'] = (d.lap_number.gt(1) & d.lap_time_s.gt(0) & d.is_accurate
                  & ~d.pit_in & ~d.pit_out & ~d.deleted & d.track_status.eq('1'))
    # A non-clean lap does not enter the history (it becomes NaN)
    d['history_value'] = d.lap_time_s.where(d.clean)
    d['history_end'] = d.end_time_s.where(d.clean)

    # STEP 3 + 4: rolling window inside driver/race/stint. rolling() only looks BACKWARD,
    # so the window for lap t contains laps t-k+1 ... t and never t+1.
    g = d.groupby(STINT_KEYS, dropna=False, sort=False)
    for k in WINDOWS:
        std = g['history_value'].transform(lambda s: s.rolling(k, min_periods=2).std())
        # STEP 5: short-history rule
        d[f'short_history_k{k}'] = std.isna().astype(int)
        d[f'recent_std_s_k{k}'] = std.fillna(0.0)
        # for the time check: latest end time actually used by the window
        d[f'last_end_used_k{k}'] = g['history_end'].transform(lambda s: s.rolling(k, min_periods=1).max())

    # STEP 6: same pair_id format as build_pairs (season-Rround-driver-Llap of lap t)
    d['pair_id'] = (d.season.astype(str) + '-R' + d['round'].astype(str) + '-'
                    + d.driver_number.astype(str) + '-L' + d.lap_number.astype(int).astype(str))
    return d


feat = add_features(raw)

# ------------------------------------------------------------------ STEP 7: keep train/validation pairs
train_ids = pd.read_csv(BASE / 'train_clean_v1.csv', usecols=['pair_id', 'recent_count'])
val_ids = pd.read_csv(BASE / 'validation_clean_v1.csv', usecols=['pair_id', 'recent_count'])
wanted = pd.concat([train_ids.assign(split='train'), val_ids.assign(split='validation')], ignore_index=True)

cols = ['pair_id'] + [c for k in WINDOWS for c in (f'recent_std_s_k{k}', f'short_history_k{k}')]
assert feat.pair_id.is_unique, 'duplicated pair_id in raw laps'
out = wanted.merge(feat, on='pair_id', how='left', validate='one_to_one', indicator=True)
out[cols].to_csv(OUT / 'features_pace.csv', index=False)

# ------------------------------------------------------------------ CHECKS (written in Hito 1)
checks = {}

# (a) Integrity: every train/validation pair found once, no NaN
checks['integrity'] = {
    'rows_written': len(out), 'expected': 5772 + 1808,
    'pairs_not_found_in_raw': int((out._merge != 'both').sum()),
    'duplicated_pair_id': int(out.pair_id.duplicated().sum()),
    'nan_values': int(out[cols[1:]].isna().sum().sum()),
}
assert checks['integrity']['rows_written'] == 7580
assert checks['integrity']['pairs_not_found_in_raw'] == 0
assert checks['integrity']['duplicated_pair_id'] == 0 and checks['integrity']['nan_values'] == 0

# (b) Last observation used: the window never uses a lap that ends after lap t
for k in WINDOWS:
    used = out[f'last_end_used_k{k}']
    late = int((used > out.end_time_s + 1e-9).sum())
    checks[f'last_observation_k{k}'] = {'pairs_using_a_lap_after_t': late}
    assert late == 0

# (c) Time rule: delete every lap after t and recompute -> value must not change
sample = out.sample(n=200, random_state=RANDOM_SEED)
changed = 0
for _, row in sample.iterrows():
    # this driver's laps in this race, keeping only laps up to t (everything after t deleted)
    truncated = raw[(raw['round'] == row['round']) & (raw.driver_number == row.driver_number)
                    & (raw.lap_number <= row.lap_number)]
    again = add_features(truncated).set_index('pair_id').loc[row.pair_id]
    for k in WINDOWS:
        if not np.isclose(again[f'recent_std_s_k{k}'], row[f'recent_std_s_k{k}']) \
                or again[f'short_history_k{k}'] != row[f'short_history_k{k}']:
            changed += 1
checks['time_rule_truncation'] = {'pairs_tested': len(sample), 'seed': RANDOM_SEED, 'values_changed': changed}
assert changed == 0

# (d) Stint boundaries: the first lap of every stint has no earlier lap in its window
first_in_stint = feat.groupby(STINT_KEYS, dropna=False).lap_number.transform('min').eq(feat.lap_number)
checks['stint_boundary'] = {
    f'first_laps_with_history_k{k}': int((first_in_stint & feat[f'short_history_k{k}'].eq(0)).sum())
    for k in WINDOWS}
assert all(v == 0 for v in checks['stint_boundary'].values())

# (e) Coverage of the short-history rule (k = 3 must match recent_count = 1: 456 / 148)
cov = {}
for k in WINDOWS:
    for split in ('train', 'validation'):
        cov[f'short_history_k{k}_{split}'] = int(out.loc[out.split == split, f'short_history_k{k}'].sum())
cov['recent_count_1_train'] = int(train_ids.recent_count.eq(1).sum())
cov['recent_count_1_validation'] = int(val_ids.recent_count.eq(1).sum())
checks['coverage'] = cov
assert cov['short_history_k3_train'] == cov['recent_count_1_train']
assert cov['short_history_k3_validation'] == cov['recent_count_1_validation']

# (f) Same timeline as the official reference: a k=3 rolling MEAN built with this code
#     must reproduce the released recent_mean_s exactly
d = raw.sort_values(KEYS + ['lap_number']).copy()
d['clean'] = (d.lap_number.gt(1) & d.lap_time_s.gt(0) & d.is_accurate
              & ~d.pit_in & ~d.pit_out & ~d.deleted & d.track_status.eq('1'))
d['mean3'] = (d.lap_time_s.where(d.clean).groupby([d[c] for c in STINT_KEYS], dropna=False, sort=False)
              .transform(lambda s: s.rolling(3, min_periods=1).mean()))
d['pair_id'] = (d.season.astype(str) + '-R' + d['round'].astype(str) + '-'
                + d.driver_number.astype(str) + '-L' + d.lap_number.astype(int).astype(str))
official = pd.concat([pd.read_csv(BASE / f, usecols=['pair_id', 'recent_mean_s'])
                      for f in ('train_clean_v1.csv', 'validation_clean_v1.csv')])
cmp = official.merge(d[['pair_id', 'mean3']], on='pair_id', how='left')
checks['timeline_matches_reference'] = {
    'pairs_compared': len(cmp),
    'max_abs_diff_recent_mean_s': float((cmp.recent_mean_s - cmp.mean3).abs().max())}
assert checks['timeline_matches_reference']['max_abs_diff_recent_mean_s'] < 1e-9

(OUT / 'feature_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
print('\nSaved', OUT / 'features_pace.csv', 'and', OUT / 'feature_checks.json')
