"""
Monte Carlo Simulation — Log-Logistic IV Estimator
===================================================
Tests the triply robust ATT estimator (log-logistic e_b + 2SLS variant)
across 8 configurations using two DGPs.

Run:
    python simulation_iv.py          # R=10 quick smoke test
    python simulation_iv.py 2000     # R=2000 full simulation
    python simulation_iv.py 2000 5000  # R=2000, n=5000
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from triple_robust_iv import _invlogit, _clip_ps, triply_robust_iv


# ---------------------------------------------------------------------------
# DGP definitions
# ---------------------------------------------------------------------------

def dgp1(rng, n):
    """
    DGP 1 (baseline): true PS is logit in sin(X).
      e0(X) = invlogit(-1 + 0.1 * sin(X))
      mu0(X) = 2 + 0.5 * log(X) + 0.3 * X
      tau = 3  (constant)
    ea with sin(X) is CORRECT, eb with linear odds in X is WRONG.
    """
    X = rng.uniform(10, 20, n)
    e0 = _clip_ps(_invlogit(-1.0 + 0.1 * np.sin(X)))
    D = rng.binomial(1, e0, n).astype(float)
    mu0 = 2.0 + 0.5 * np.log(X) + 0.3 * X
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = Y0 + 3.0
    Y = D * Y1 + (1.0 - D) * Y0
    return Y, D, X


def dgp2(rng, n):
    """
    DGP 2 (linear odds): true PS has linear odds in X.
      e0(X) = (delta_0 + delta_1*X) / (1 + delta_0 + delta_1*X)
      with delta_0 = -0.5, delta_1 = 0.05
      mu0(X) = 2 + 0.5 * log(X) + 0.3 * X  (same as DGP 1)
      tau = 3  (constant)
    eb with linear odds in X is CORRECT, ea with sin(X) is WRONG.
    """
    X = rng.uniform(10, 20, n)
    delta_0, delta_1 = -0.5, 0.05
    odds = delta_0 + delta_1 * X  # odds for X in [10,20]: [0.0, 0.5]
    odds = np.maximum(odds, 1e-6)  # ensure positive
    e0 = _clip_ps(odds / (1.0 + odds))
    D = rng.binomial(1, e0, n).astype(float)
    mu0 = 2.0 + 0.5 * np.log(X) + 0.3 * X
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = Y0 + 3.0
    Y = D * Y1 + (1.0 - D) * Y0
    return Y, D, X


# ---------------------------------------------------------------------------
# 8 test cases
# ---------------------------------------------------------------------------

CASES = [
    # (case_label, dgp_fn, or_columns, ea_basis, eb_basis, r_ok, ea_ok, eb_ok, expected)
    # --- DGP 1 ---
    ('1',  dgp1, 'logX_X', 'sin', 'X', True,  True,  False, 'Consistent'),
    ('2',  dgp1, 'logX_X', 'X2',  'X', True,  False, False, 'Consistent'),
    ('3',  dgp1, 'logX',   'sin', 'X', False, True,  False, 'Consistent'),
    ('4',  dgp1, 'logX',   'X2',  'X', False, False, False, 'Biased'),
    # --- DGP 2 ---
    ('5',  dgp2, 'logX',   'sin', 'X', False, False, True,  'Consistent'),
    ('6',  dgp2, 'logX_X', 'sin', 'X', True,  False, True,  'Consistent'),
    ('7',  dgp2, 'logX',   'X2',  'X', False, False, True,  'Consistent'),
    ('4b', dgp2, 'logX',   'X2',  'sin', False, False, False, 'Biased'),
]

TRUE_ATT = 3.0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(R=10, n=5000, seed=42):
    """Run all 8 cases and print summary table."""

    print(f"Monte Carlo Simulation: Triply Robust IV Estimator")
    print(f"R={R} replications, n={n}, True ATT={TRUE_ATT}")
    print("=" * 100)

    # Verify true ATT with large sample
    rng_check = np.random.default_rng(999)
    Y_c, D_c, X_c = dgp1(rng_check, 500_000)
    empirical_att = float(np.mean(Y_c[D_c == 1]) - np.mean(
        (2.0 + 0.5 * np.log(X_c) + 0.3 * X_c)[D_c == 1]))
    print(f"Sanity check — DGP 1 empirical ATT (n=500k): {empirical_att:.4f}")

    Y_c2, D_c2, X_c2 = dgp2(rng_check, 500_000)
    empirical_att2 = float(np.mean(Y_c2[D_c2 == 1]) - np.mean(
        (2.0 + 0.5 * np.log(X_c2) + 0.3 * X_c2)[D_c2 == 1]))
    print(f"Sanity check — DGP 2 empirical ATT (n=500k): {empirical_att2:.4f}")
    print()

    header = (f"{'Case':>4}  {'DGP':>4}  {'r':>3}  {'ea':>3}  {'eb':>3}  "
              f"{'Expected':>11}  {'Mean τ̂':>8}  {'Bias':>8}  "
              f"{'SD':>7}  {'RMSE':>7}  {'1st-F':>7}  {'eb conv%':>8}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    all_results = {}

    for case_label, dgp_fn, or_cols, ea_b, eb_b, r_ok, ea_ok, eb_ok, expected in CASES:
        rng = np.random.default_rng(seed + hash(case_label) % 10000)

        tau_list = []
        f_list = []
        eb_conv_count = 0

        dgp_label = '1' if dgp_fn is dgp1 else '2'

        for _ in range(R):
            Y, D, X = dgp_fn(rng, n)
            try:
                result = triply_robust_iv(Y, D, X,
                                          ea_basis=ea_b,
                                          eb_basis=eb_b,
                                          or_columns=or_cols)
                tau_list.append(result['tau_att'])
                f_list.append(result['first_F'])
                if result['eb_converged']:
                    eb_conv_count += 1
            except Exception as e:
                # Skip failed replications
                pass

        if tau_list:
            arr = np.array(tau_list)
            mean_tau = float(np.mean(arr))
            bias = mean_tau - TRUE_ATT
            sd = float(np.std(arr))
            rmse = float(np.sqrt(bias ** 2 + sd ** 2))
            mean_F = float(np.mean(f_list))
            eb_conv_pct = 100.0 * eb_conv_count / len(tau_list)
        else:
            mean_tau = bias = sd = rmse = mean_F = float('nan')
            eb_conv_pct = 0.0

        all_results[case_label] = arr if tau_list else np.array([])

        r_sym = '✓' if r_ok else '✗'
        ea_sym = '✓' if ea_ok else '✗'
        eb_sym = '✓' if eb_ok else '✗'

        print(f"{case_label:>4}  {dgp_label:>4}  {r_sym:>3}  {ea_sym:>3}  {eb_sym:>3}  "
              f"{expected:>11}  {mean_tau:>8.4f}  {bias:>+8.4f}  "
              f"{sd:>7.4f}  {rmse:>7.4f}  {mean_F:>7.1f}  {eb_conv_pct:>7.1f}%")

    print(sep)
    print(f"\nExpected: Bias ≈ 0 for Cases 1-3, 5-7.  Clear bias for Cases 4, 4b.")
    print(f"True ATT = {TRUE_ATT}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if R >= 5:
        _make_plots(all_results, R)

    return all_results


def _make_plots(all_results, R):
    """Generate histogram and box plot of tau_hat across cases."""

    case_labels = [c[0] for c in CASES]
    plot_data = [all_results.get(label, np.array([])) for label in case_labels]

    # Box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    non_empty = [(label, data) for label, data in zip(case_labels, plot_data) if len(data) > 0]
    if non_empty:
        bp_labels, bp_data = zip(*non_empty)
        ax.boxplot(bp_data, tick_labels=bp_labels)
        ax.axhline(y=TRUE_ATT, color='r', linestyle='--', label=f'True ATT = {TRUE_ATT}')
        ax.set_ylabel('τ̂')
        ax.set_xlabel('Case')
        ax.set_title(f'Triply Robust IV Estimator — Box Plot (R={R})')
        ax.legend()
        fig.tight_layout()
        fig.savefig('boxplot_iv.png', dpi=150)
        print(f"\nBox plot saved to boxplot_iv.png")
    plt.close(fig)

    # Small-multiples histogram
    n_cases = len(case_labels)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, (label, data) in enumerate(zip(case_labels, plot_data)):
        ax = axes[i]
        if len(data) > 0:
            ax.hist(data, bins=min(30, max(5, R // 5)), edgecolor='black', alpha=0.7)
            ax.axvline(x=TRUE_ATT, color='r', linestyle='--', linewidth=2)
            ax.axvline(x=np.mean(data), color='blue', linestyle='-', linewidth=1.5)
        ax.set_title(f'Case {label}')
        ax.set_xlabel('τ̂')
    fig.suptitle(f'Triply Robust IV — Histograms (R={R})', fontsize=14)
    fig.tight_layout()
    fig.savefig('histograms_iv.png', dpi=150)
    print(f"Histograms saved to histograms_iv.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run_simulation(R=R, n=n)
