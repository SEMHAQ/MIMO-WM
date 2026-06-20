"""Compute statistics for paper tables."""
import json, numpy as np
from scipy import stats as st

with open('experiments/multiseed_results.json') as f:
    d = json.load(f)

# Table 1: Humanoid
print('=== TABLE 1: Humanoid ===')
for model in ['LSTM-WM', 'Transformer-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM']:
    seeds = d[f'{model}_humanoid']
    mses = [seeds[s]['mse'] for s in sorted(seeds.keys())]
    r2s = [seeds[s]['r2'] for s in sorted(seeds.keys())]
    m, s = np.mean(mses)*100, np.std(mses, ddof=1)*100
    r, rs = np.mean(r2s), np.std(r2s, ddof=1)
    print(f'{model}: MSE={m:.2f}+-{s:.2f}, R2={r:.3f}+-{rs:.3f}')

# Table 5: Ant
print('\n=== TABLE 5: Ant ===')
for model in ['LSTM-WM', 'Transformer-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM']:
    seeds = d[f'{model}_ant']
    mses = [seeds[s]['mse'] for s in sorted(seeds.keys())]
    r2s = [seeds[s]['r2'] for s in sorted(seeds.keys())]
    m, s = np.mean(mses)*100, np.std(mses, ddof=1)*100
    r, rs = np.mean(r2s), np.std(r2s, ddof=1)
    print(f'{model}: MSE={m:.2f}+-{s:.2f}, R2={r:.3f}+-{rs:.3f}')

# Table 6: Walker2d
print('\n=== TABLE 6: Walker2d ===')
for model in ['LSTM-WM', 'Transformer-WM', 'GRU-WM', 'Mamba-WM', 'S4D-WM']:
    seeds = d[f'{model}_walker2d']
    mses = [seeds[s]['mse'] for s in sorted(seeds.keys())]
    r2s = [seeds[s]['r2'] for s in sorted(seeds.keys())]
    m, s = np.mean(mses)*100, np.std(mses, ddof=1)*100
    r, rs = np.mean(r2s), np.std(r2s, ddof=1)
    print(f'{model}: MSE={m:.2f}+-{s:.2f}, R2={r:.3f}+-{rs:.3f}')

# Table 2: t-test (Humanoid)
print('\n=== TABLE 2: t-test (Humanoid) ===')
s4d = [d['S4D-WM_humanoid'][s]['mse'] for s in sorted(d['S4D-WM_humanoid'].keys())]
for model in ['LSTM-WM', 'Transformer-WM', 'GRU-WM', 'Mamba-WM']:
    base = [d[f'{model}_humanoid'][s]['mse'] for s in sorted(d[f'{model}_humanoid'].keys())]
    diff = np.array(base) - np.array(s4d)
    d_cohen = np.mean(diff) / np.std(diff, ddof=1)
    t_stat, p_val = st.ttest_rel(base, s4d)
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_low = np.mean(diff) - st.t.ppf(0.975, 4) * se
    ci_high = np.mean(diff) + st.t.ppf(0.975, 4) * se
    delta = np.mean(diff) * 100
    print(f'{model}: delta={delta:.2f}, p={p_val:.4f}, d={d_cohen:.2f}, CI=[{ci_low*100:.1f}, {ci_high*100:.1f}]')

# Ant t-test
print('\n=== Ant t-test ===')
s4d_ant = [d['S4D-WM_ant'][s]['mse'] for s in sorted(d['S4D-WM_ant'].keys())]
for model in ['LSTM-WM', 'Transformer-WM', 'GRU-WM', 'Mamba-WM']:
    base = [d[f'{model}_ant'][s]['mse'] for s in sorted(d[f'{model}_ant'].keys())]
    diff = np.array(base) - np.array(s4d_ant)
    d_cohen = np.mean(diff) / np.std(diff, ddof=1)
    t_stat, p_val = st.ttest_rel(base, s4d_ant)
    delta = np.mean(diff) * 100
    print(f'{model}: delta={delta:.2f}, p={p_val:.4f}, d={d_cohen:.2f}')
