"""Hito 2 - executed comparison, ablation, error analysis and SHAP (NextLap, IIT414W).

Runs now with the reference only. When Benjamin's feature file exists
(Hito2/outputs/features_pace.csv), it also fits and evaluates the candidates.

Feature-file contract (one row per pair_id, all 5,772 train + 1,808 validation pairs):
    pair_id, recent_std_s_k3, short_history_k3, recent_std_s_k5, short_history_k5
Built from the original lap timeline, within driver/race/stint, using clean laps through t,
BEFORE the clean-pair filter. No NaN: short histories use std = 0 and short_history = 1.

Usage (repo root has base_NextLap/, git-ignored):
    python Hito2/run_hito2.py                    # SHAP case = largest-error row (declared rule)
    python Hito2/run_hito2.py --case 2026-R8-1-L2 --explain k3
"""
from pathlib import Path
import argparse, sys, json, hashlib, subprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--base', default=str(HERE.parent / 'base_NextLap'))
ap.add_argument('--features', default=str(HERE / 'outputs' / 'features_pace.csv'))
ap.add_argument('--case', default=None, help='pair_id for the local SHAP / inspected case')
ap.add_argument('--explain', default=None, help="model to explain with SHAP: 'reference', 'k3' or 'k5' (default: k3 if present, else reference)")
args = ap.parse_args()
BASE = Path(args.base).resolve()
sys.path.insert(0, str(BASE))
from nextlap_reference_v1 import make_model, NUMERIC, CATEGORICAL, TARGET  # noqa: E402

RANDOM_SEED = 414  # OLS has no randomness; kept for any later model
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Data and split (C1)
train = pd.read_csv(BASE / 'train_clean_v1.csv')
val = pd.read_csv(BASE / 'validation_clean_v1.csv')
assert len(train) == 5772 and len(val) == 1808
assert sorted(train['round'].unique()) == list(range(1, 8)), 'train must be R1-7'
assert sorted(val['round'].unique()) == [8, 9], 'validation must be R8-9'
assert train.pair_id.is_unique and val.pair_id.is_unique
assert not set(train.pair_id) & set(val.pair_id)
print('Train rounds:', sorted(train['round'].unique()), 'n =', len(train))
print('Validation rounds:', sorted(val['round'].unique()), 'n =', len(val),
      '| R8 =', int(val['round'].eq(8).sum()), '| R9 =', int(val['round'].eq(9).sum()))

# Same validation IDs as the frozen Hito 1 list
h1_ids = HERE.parent / 'Hito1' / 'outputs' / 'validation_pair_ids.txt'
if h1_ids.is_file():
    frozen = set(pd.read_csv(h1_ids, header=None)[0])
    assert frozen == set(val.pair_id), 'validation IDs differ from Hito 1'
    print('Validation pair_ids identical to Hito 1 frozen list.')

# ---------------------------------------------------------------- 2. Optional candidate features
EXTRA = {}
coverage = {}
fpath = Path(args.features)
if fpath.is_file():
    feats = pd.read_csv(fpath)
    assert feats.pair_id.is_unique, 'duplicated pair_id in feature file'
    for name, df in [('train', train), ('validation', val)]:
        missing = set(df.pair_id) - set(feats.pair_id)
        assert not missing, f'{len(missing)} {name} pair_ids missing in feature file'
    n_tr, n_va = len(train), len(val)
    train = train.merge(feats, on='pair_id', how='left', validate='one_to_one')
    val = val.merge(feats, on='pair_id', how='left', validate='one_to_one')
    assert len(train) == n_tr and len(val) == n_va, 'join changed row counts'
    for k in (3, 5):
        cols = [f'recent_std_s_k{k}', f'short_history_k{k}']
        if all(c in feats.columns for c in cols):
            assert not train[cols].isna().any().any() and not val[cols].isna().any().any(), 'NaN in features'
            EXTRA[f'OLS + pace variability k={k}'] = cols
            coverage[f'k{k}'] = {'short_history_train': int(train[cols[1]].sum()),
                                 'short_history_validation': int(val[cols[1]].sum())}
    print('Candidate feature sets:', list(EXTRA) or 'none found in file')
else:
    print(f'No feature file at {fpath} yet: running reference only.')

# ---------------------------------------------------------------- 3. Fit (train only) and predict (C2, C3)
REF = 'OLS reference = matched control'
specs = {REF: []} | EXTRA
models, preds = {}, {}
for name, extra in specs.items():
    num = NUMERIC + extra
    m = make_model('OLS', numeric=num).fit(train[num + CATEGORICAL], train[TARGET])  # fit on TRAIN only
    models[name] = (m, num)
    preds[name] = m.predict(val[num + CATEGORICAL])
    assert np.isfinite(preds[name]).all() and len(preds[name]) == len(val)
preds_df = pd.DataFrame({'pair_id': val.pair_id, 'round': val['round'], 'recent_count': val.recent_count,
                         'y_true_s': val[TARGET], 'persistence': val.current_lap_s, 'train_median': train[TARGET].median(),
                         **preds})
preds_df.to_csv(OUT / 'validation_predictions.csv', index=False)

y = val[TARGET].to_numpy()
ctrl_mae = mean_absolute_error(y, preds[REF])
rows = [{'version': 'persistence', 'inputs': 'next = current_lap_s', 'n': len(val),
         'MAE_s': mean_absolute_error(y, val.current_lap_s), 'control_minus_candidate_s': np.nan}]
for name, extra in specs.items():
    mae = mean_absolute_error(y, preds[name])
    rows.append({'version': name, 'inputs': '6 reference inputs' + (' + ' + ', '.join(extra) if extra else ''),
                 'n': len(val), 'MAE_s': mae,
                 'control_minus_candidate_s': np.nan if name == REF else ctrl_mae - mae})
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / 'comparison.csv', index=False)
print('\n== Comparison (validation, MAE in seconds; positive difference = candidate better)')
print(comparison.to_string(index=False))

# ---------------------------------------------------------------- 4. Errors by race and slice (C4)
by_race = []
for name in specs:
    for r in (8, 9):
        mk = (val['round'] == r).to_numpy()
        c, m = np.abs(y[mk] - preds[REF][mk]).mean(), np.abs(y[mk] - preds[name][mk]).mean()
        by_race.append({'version': name, 'round': r, 'n': int(mk.sum()), 'MAE_control_s': c,
                        'MAE_version_s': m, 'control_minus_version_s': c - m})
by_race = pd.DataFrame(by_race)
for name in specs:  # pooled MAE must equal the n-weighted mean of race MAEs
    t = by_race[by_race.version == name]
    pooled = (t.MAE_version_s * t.n).sum() / t.n.sum()
    assert abs(pooled - mean_absolute_error(y, preds[name])) < 1e-9
by_race.to_csv(OUT / 'mae_by_race.csv', index=False)
print('\n== MAE by race (n-weighted pooling checked)')
print(by_race.to_string(index=False))

slices = []
for name in specs:
    for rc, g in preds_df.groupby('recent_count'):
        slices.append({'version': name, 'recent_count': int(rc), 'n': len(g),
                       'MAE_s': float(np.abs(g.y_true_s - g[name]).mean())})
pd.DataFrame(slices).to_csv(OUT / 'mae_by_recent_count.csv', index=False)

# ---------------------------------------------------------------- 5. SHAP (C5, C6) on the explained model
key = args.explain or ('k3' if any('k=3' in n for n in specs) else 'reference')
explained = REF if key == 'reference' else next(n for n in specs if f"k={key[-1]}" in n)
model, num = models[explained]
prep, reg = model.named_steps['prepare'], model.named_steps['regression']
names = prep.get_feature_names_out()
Z_tr = prep.transform(train[num + CATEGORICAL])
Z_va = prep.transform(val[num + CATEGORICAL])
background = Z_tr.mean(axis=0)                              # background = all training rows
base_value = float(reg.intercept_ + reg.coef_ @ background)
phi = reg.coef_ * (Z_va - background)                      # exact linear SHAP (interventional)

def original(col):  # map encoded column -> original input
    col = col.split('__', 1)[1]
    for c in CATEGORICAL:
        if col.startswith(c + '_'):
            return c
    return col
mapping = pd.DataFrame({'encoded_input': names, 'original_input': [original(n) for n in names]})
mapping.to_csv(OUT / 'shap_encoded_to_original.csv', index=False)
phi_orig = pd.DataFrame(phi, columns=names).T.groupby(mapping.original_input.values).sum().T
recon = base_value + phi_orig.sum(axis=1).to_numpy()
assert np.allclose(recon, preds[explained], atol=1e-6), 'base + contributions != prediction'

# Optional cross-check with the shap library
shap_check = 'shap library not installed'
try:
    import shap
    ex = shap.LinearExplainer(reg, Z_tr, feature_perturbation='interventional')
    sv = ex.shap_values(Z_va)
    shap_check = f'shap {shap.__version__}: max |diff| = {float(np.abs(sv - phi).max()):.2e}'
except Exception as e:  # noqa: BLE001
    shap_check = f'shap library unavailable ({type(e).__name__}); exact linear decomposition used'
print('\nSHAP cross-check:', shap_check)

# Local explanation: declared case
abs_err = np.abs(y - preds[explained])
if args.case:
    idx = int(np.flatnonzero(val.pair_id.to_numpy() == args.case)[0])
    rule = f'pair_id chosen by the team: {args.case}'
else:
    idx = int(abs_err.argmax())
    rule = 'largest absolute error of the explained model on validation'
local = {'explained_model': explained, 'selection_rule': rule, 'pair_id': val.pair_id.iloc[idx],
         'round': int(val['round'].iloc[idx]), 'observed_s': float(y[idx]),
         'prediction_s': float(preds[explained][idx]), 'base_value_s': base_value,
         'contributions_s': {k: float(v) for k, v in phi_orig.iloc[idx].sort_values(key=abs, ascending=False).items()},
         'base_plus_contributions_s': float(recon[idx]), 'tolerance_s': 1e-6,
         'absolute_error_s': float(abs_err[idx]),
         'control_prediction_s': float(preds[REF][idx]),
         'control_absolute_error_s': float(abs(y[idx] - preds[REF][idx]))}
(OUT / 'shap_local_case.json').write_text(json.dumps(local, indent=2), encoding='utf-8')
print('\n== Local SHAP / inspected case'); print(json.dumps(local, indent=2))

# Global explanation: mean |SHAP| over ALL validation rows
glob = (phi_orig.abs().mean().sort_values(ascending=False).rename('mean_abs_shap_s').reset_index()
        .rename(columns={'index': 'original_input'}))
glob.insert(0, 'explained_model', explained)
glob['sample'] = f'all validation rows R8-R9, n = {len(val)}'
glob.to_csv(OUT / 'shap_global_mean_abs.csv', index=False)
print('\n== Global SHAP (mean |contribution| in seconds)'); print(glob.to_string(index=False))

# ---------------------------------------------------------------- 6. Run record (C7)
try:
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
except Exception:  # noqa: BLE001
    commit = 'unknown'
run = {'commit_at_run': commit, 'command': ' '.join(['python'] + sys.argv), 'test_used': False, 'seed': RANDOM_SEED,
       'train_rounds': list(range(1, 8)), 'validation_rounds': [8, 9], 'n_train': len(train), 'n_validation': len(val),
       'fit_on': 'training rows only (sklearn Pipeline)', 'versions_compared': list(specs), 'coverage': coverage,
       'shap': {'explained_model': explained, 'method': 'exact linear SHAP: coef * (x - mean of training rows) on encoded inputs, summed per original input',
                'background': 'all 5,772 training rows', 'cross_check': shap_check},
       'environment': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
       'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [BASE / 'train_clean_v1.csv', BASE / 'validation_clean_v1.csv', BASE / 'nextlap_reference_v1.py']
                  + ([fpath] if fpath.is_file() else [])}}
(OUT / 'hito2_run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
print('\nSaved outputs in', OUT)
