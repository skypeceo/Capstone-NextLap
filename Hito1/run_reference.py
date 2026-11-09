"""Hito 1 - reproduce the three NextLap references on the common validation rows.

Adapted from the course helper probar_referencia_v1.py (IIT414W, instructor-supplied),
used with attribution. Reads only the released train/validation CSVs; no test data.

Usage (from this folder):
    python run_reference.py                      # expects ../base_NextLap
    python run_reference.py --base /path/to/base_NextLap
"""
from pathlib import Path
import argparse, sys, json, hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--base', default=str(HERE.parent / 'base_NextLap'))
args = ap.parse_args()
BASE = Path(args.base).resolve()
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import make_model, NUMERIC, CATEGORICAL, TARGET  # noqa: E402

RANDOM_SEED = 414  # no random step in these references; kept fixed for later models
np.random.seed(RANDOM_SEED)

train = pd.read_csv(BASE / 'train_clean_v1.csv')
val = pd.read_csv(BASE / 'validation_clean_v1.csv')

# Checks on data, split and row identity
assert len(train) == 5772 and len(val) == 1808, 'unexpected row counts'
assert set(train['round']) == set(range(1, 8)) and set(val['round']) == {8, 9}, 'unexpected rounds'
assert train.pair_id.is_unique and val.pair_id.is_unique, 'duplicated pair_id'
assert not set(train.pair_id) & set(val.pair_id), 'train/validation overlap'
assert train.clean_pair.all() and val.clean_pair.all(), 'non-clean rows in clean files'
features = NUMERIC + CATEGORICAL
assert TARGET not in features and not any(c.startswith('next_') for c in features), 'future-known input'

# Preprocessing (imputers, scaler, one-hot) is inside the Pipeline: fit on TRAIN only
model = make_model('OLS').fit(train[features], train[TARGET])
preds = {
    'train_median': np.full(len(val), train[TARGET].median()),  # constant learned on train
    'persistence': val.current_lap_s.to_numpy(),                 # next lap = current lap
    'OLS_reference': model.predict(val[features]),
}

rows = []
for name, pred in preds.items():
    assert np.isfinite(pred).all() and len(pred) == len(val)
    for label, mask in [('all_validation', np.ones(len(val), bool)),
                        ('round_8', val['round'].eq(8).to_numpy()),
                        ('round_9', val['round'].eq(9).to_numpy())]:
        rows.append({'model': name, 'population': label, 'n': int(mask.sum()),
                     'MAE_s': float(mean_absolute_error(val.loc[mask, TARGET], pred[mask]))})

out = HERE / 'outputs'
out.mkdir(exist_ok=True)
metrics = pd.DataFrame(rows)
metrics.to_csv(out / 'reference_metrics.csv', index=False)
pd.DataFrame({'pair_id': val.pair_id, 'round': val['round'], 'y_true_s': val[TARGET], **preds}) \
    .to_csv(out / 'validation_reference_predictions.csv', index=False)
# Frozen evaluation row set, used in Hito 2 to assert identical rows
val.pair_id.sort_values().to_csv(out / 'validation_pair_ids.txt', index=False, header=False)

report = {
    'run_by': 'team (own execution)', 'test_used': False, 'seed': RANDOM_SEED,
    'train_rows': len(train), 'validation_rows': len(val),
    'train_rounds': sorted(map(int, set(train['round']))),
    'validation_rounds': sorted(map(int, set(val['round']))),
    'features': features, 'target': TARGET, 'metric': 'MAE in seconds',
    'preprocessing_fit_on': 'train only (sklearn Pipeline)',
    'results': rows,
    'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
    'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [BASE / 'train_clean_v1.csv', BASE / 'validation_clean_v1.csv',
                BASE / 'nextlap_reference_v1.py']},
}
(out / 'reference_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(metrics.to_string(index=False))
print(f"\nSaved to {out}. No test data used.")
