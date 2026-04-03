"""
DGP 3 Comparison: v1 (φ in outer solver) vs v2 (unified concentration)
======================================================================
Runs R=10 replications of DGP 3 and compares:
  - v1: current estimator (φ penalised in 3×3 outer LM solver)
  - v2: unified concentration (φ absorbed into 3×3 linear system,
         outer solver is 2×2 in γ only)

DGP 3: e_a correct (logit-on-sin), OR and e_b wrong.  True ATT = 4.0.
"""

import numpy as np
from simulation import dgp3
from triple_robust import triply_robust_att, triply_robust_att_v2

R = 10
n = 2000
seed = 42
TRUE_ATT = 4.0

print(f"DGP 3 comparison: v1 vs v2 (unified concentration)")
print(f"R={R} replications, n={n}, True ATT={TRUE_ATT}")
print("=" * 85)

rng = np.random.default_rng(seed + 3)

header = (f"{'Rep':>4}  {'v1 ATT':>8}  {'v1 φ':>10}  {'v1 conv':>7}  "
          f"{'v2 ATT':>8}  {'v2 φ':>10}  {'v2 conv':>7}  {'v2 cond':>10}")
print(header)
print("-" * len(header))

v1_list, v2_list = [], []
v1_phi_list, v2_phi_list = [], []
v1_conv, v2_conv = 0, 0

for rep in range(R):
    Y, D, X = dgp3(rng, n)

    res1 = triply_robust_att(Y, D, X)
    res2 = triply_robust_att_v2(Y, D, X)

    c1 = res1['converged']
    c2 = res2['converged']

    if c1:
        v1_list.append(res1['tau_att'])
        v1_phi_list.append(res1['phi'])
        v1_conv += 1
    if c2:
        v2_list.append(res2['tau_att'])
        v2_phi_list.append(res2['phi'])
        v2_conv += 1

    print(f"{rep+1:>4}  {res1['tau_att']:>+8.3f}  {res1['phi']:>+10.3f}  "
          f"{'  Y' if c1 else '  N':>7}  "
          f"{res2['tau_att']:>+8.3f}  {res2['phi']:>+10.3f}  "
          f"{'  Y' if c2 else '  N':>7}  "
          f"{res2['cond_number']:>10.1f}")

print("-" * len(header))

# Summary
def _summarize(label, lst, phi_lst, n_conv, R):
    if lst:
        arr = np.array(lst)
        bias = np.mean(arr) - TRUE_ATT
        sd = np.std(arr)
        rmse = np.sqrt(bias**2 + sd**2)
        phi_arr = np.array(phi_lst)
        phi_mean = np.mean(phi_arr)
        phi_sd = np.std(phi_arr)
    else:
        bias = sd = rmse = phi_mean = phi_sd = float('nan')
    print(f"\n{label}:")
    print(f"  Conv: {n_conv}/{R}  Bias: {bias:+.4f}  SD: {sd:.4f}  RMSE: {rmse:.4f}")
    print(f"  φ mean: {phi_mean:+.3f}  φ SD: {phi_sd:.3f}")

_summarize("v1 (φ in outer solver, penalty λ=0.01)", v1_list, v1_phi_list, v1_conv, R)
_summarize("v2 (unified concentration, no φ penalty)", v2_list, v2_phi_list, v2_conv, R)

print(f"\nNote: DGP 3 needs large |φ| to correct for misspecified OR+e_b.")
print(f"v2 frees φ from the penalty, letting the linear system determine it from data.")
