"""
Surgical Monte Carlo — Isolating the e_b Channel
=================================================
Tests whether the simplified 2SLS estimator (S) inherits consistency
through the e_b channel, using a DGP designed to kill all escape routes:

  - OR is truly wrong (linear X vs true quadratic + sin),
  - e_a is truly wrong (logit on X vs true linear-odds in exp(X)),
  - the IV regressor space {1, X, exp(X)} cannot span the true mu_0.

Compares three estimators:
  S — Simplified (log-logistic MLE for e_b, then 2SLS)
  F — Full (same, but 2SLS includes the M6 telescoping instrument)
  N — Negative control (S with deliberately wrong e_b basis)

Run:
    python simulation_surgical.py           # R=10  smoke test
    python simulation_surgical.py 500       # R=500 full
    python simulation_surgical.py 500 5000  # R=500, max_n=5000
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from triple_robust_iv import (_invlogit, _clip_ps, triply_robust_iv,
                               triply_robust_iv_full, _log_logistic_mle)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRUE_ATT = 2.0
DELTA_0, DELTA_1 = 1.5, 0.8   # true odds parameters


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def dgp_surgical(rng, n, kappa=0.0):
    """
    Surgical DGP where only e_b (linear odds in exp(X)) is correct.

    X ~ Uniform[-2, 2]
    e_0(X) / (1 - e_0(X)) = delta_0 + delta_1 * exp(X) + kappa * X^2
    mu_0(X) = 1 + 0.7 * X^2 + 0.5 * sin(2X)
    tau = 2 (constant treatment effect)

    kappa > 0 introduces near-miss misspecification in e_b.
    """
    X = rng.uniform(-2, 2, n)
    odds = DELTA_0 + DELTA_1 * np.exp(X) + kappa * X ** 2
    odds = np.maximum(odds, 1e-6)
    e0 = _clip_ps(odds / (1.0 + odds))
    D = rng.binomial(1, e0, n).astype(float)
    mu0 = 1.0 + 0.7 * X ** 2 + 0.5 * np.sin(2.0 * X)
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = Y0 + TRUE_ATT
    Y = D * Y1 + (1.0 - D) * Y0
    return Y, D, X


# ---------------------------------------------------------------------------
# Diagnostics (run once on large sample)
# ---------------------------------------------------------------------------

def run_diagnostics(n_large=200_000, seed=999):
    """Span check and e_a misspecification check."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n_large)

    # True mu_0
    mu0 = 1.0 + 0.7 * X ** 2 + 0.5 * np.sin(2.0 * X)

    # Span check: regress mu_0 on {1, X, exp(X)}
    R_span = np.column_stack([np.ones(n_large), X, np.exp(X)])
    coef, _, _, _ = np.linalg.lstsq(R_span, mu0, rcond=None)
    mu0_hat = R_span @ coef
    ss_res = np.sum((mu0 - mu0_hat) ** 2)
    ss_tot = np.sum((mu0 - np.mean(mu0)) ** 2)
    R2_span = 1.0 - ss_res / ss_tot

    # e_a misspecification: compare true PS vs logit-on-X approximation
    odds_true = DELTA_0 + DELTA_1 * np.exp(X)
    e_true = odds_true / (1.0 + odds_true)
    # Best logit-on-X fit
    from statsmodels.discrete.discrete_model import Logit
    import statsmodels.api as sm
    D_large = rng.binomial(1, e_true, n_large).astype(float)
    Xa = sm.add_constant(X)
    try:
        logit_fit = Logit(D_large, Xa).fit(disp=False)
        e_logit = _clip_ps(logit_fit.predict(Xa))
    except Exception:
        e_logit = np.full(n_large, np.mean(D_large))
    mae_ea = float(np.mean(np.abs(e_true - e_logit)))
    corr_ea = float(np.corrcoef(e_true, e_logit)[0, 1])

    # Empirical ATT check
    Y, D, X_s = dgp_surgical(rng, n_large)
    mu0_s = 1.0 + 0.7 * X_s ** 2 + 0.5 * np.sin(2.0 * X_s)
    empirical_att = float(np.mean(Y[D == 1]) - np.mean(mu0_s[D == 1]))

    return {
        'R2_span': R2_span,
        'mae_ea': mae_ea,
        'corr_ea': corr_ea,
        'empirical_att': empirical_att,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(R=10, sample_sizes=None, seed=42):
    """Run the surgical Monte Carlo and print results."""
    if sample_sizes is None:
        sample_sizes = [500, 2000, 5000]

    print("=" * 100)
    print("Surgical Monte Carlo: Isolating the e_b Channel")
    print(f"R={R} replications, n ∈ {sample_sizes}, True ATT={TRUE_ATT}")
    print("=" * 100)

    # --- Diagnostics ---
    print("\n--- Diagnostics (large-sample checks) ---")
    diag = run_diagnostics()
    print(f"  Span check R² (true μ₀ on {{1, X, exp(X)}}): {diag['R2_span']:.4f}")
    print(f"  e_a misspecification: MAE = {diag['mae_ea']:.4f}, corr = {diag['corr_ea']:.4f}")
    print(f"  Empirical ATT (n=200k): {diag['empirical_att']:.4f}")
    if diag['R2_span'] > 0.95:
        print("  *** WARNING: R² near 1 → design may be contaminated by accidental spanning ***")
    else:
        print(f"  ✓ R² = {diag['R2_span']:.4f} — no accidental spanning.")
    print()

    # --- Define cases ---
    # (label, kappa, eb_basis, estimator_fn, estimator_name)
    cases = [
        # Case A: only e_b correct
        ('A-S', 0.0,  'expX', triply_robust_iv,      'S (simplified)'),
        ('A-F', 0.0,  'expX', triply_robust_iv_full,  'F (full + M6)'),
        ('A-N', 0.0,  'X',    triply_robust_iv,      'N (wrong e_b)'),
        # Case B: wrong e_b (= negative control)
        ('B',   0.0,  'X',    triply_robust_iv,      'N (wrong e_b)'),
        # Case C: near-miss e_b
        ('C',   0.05, 'expX', triply_robust_iv,      'S (near-miss)'),
    ]

    # Header
    header = (f"{'Case':>5}  {'Est':>16}  {'n':>6}  {'Mean τ̂':>8}  {'Bias':>8}  "
              f"{'SD':>7}  {'RMSE':>7}  {'1st-F':>7}  {'eb%':>5}  "
              f"{'Mean|S-F|':>10}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    # Storage for S-F comparison
    sf_diffs = {}  # n → list of (tau_S - tau_F) per replication

    for n in sample_sizes:
        # Pre-generate seeds for reproducibility — same data for S, F, N
        rng_base = np.random.default_rng(seed + n)
        rep_seeds = rng_base.integers(0, 2**31, size=R)

        # Run all cases that share DGP (kappa=0): A-S, A-F, A-N
        results_A = {'S': [], 'F': [], 'N': []}
        f_stats_A = {'S': [], 'F': [], 'N': []}
        conv_A = {'S': 0, 'F': 0, 'N': 0}
        m0_diffs = []  # max|m0_S - m0_F| per replication

        for rep_idx in range(R):
            rng_rep = np.random.default_rng(rep_seeds[rep_idx])
            Y, D, X = dgp_surgical(rng_rep, n, kappa=0.0)

            # Estimator S
            try:
                res_S = triply_robust_iv(Y, D, X, ea_basis='X',
                                         eb_basis='expX', or_columns='X')
                results_A['S'].append(res_S['tau_att'])
                f_stats_A['S'].append(res_S['first_F'])
                if res_S['eb_converged']:
                    conv_A['S'] += 1
            except Exception:
                res_S = None

            # Estimator F
            try:
                res_F = triply_robust_iv_full(Y, D, X, ea_basis='X',
                                              eb_basis='expX', or_columns='X')
                results_A['F'].append(res_F['tau_att'])
                f_stats_A['F'].append(res_F['first_F'])
                if res_F['eb_converged']:
                    conv_A['F'] += 1
            except Exception:
                res_F = None

            # S-F comparison
            if res_S is not None and res_F is not None:
                sf_key = n
                sf_diffs.setdefault(sf_key, []).append(
                    res_S['tau_att'] - res_F['tau_att'])
                m0_diffs.append(
                    float(np.max(np.abs(res_S['m0_hat'] - res_F['m0_hat']))))

            # Estimator N (wrong e_b)
            try:
                res_N = triply_robust_iv(Y, D, X, ea_basis='X',
                                         eb_basis='X', or_columns='X')
                results_A['N'].append(res_N['tau_att'])
                f_stats_A['N'].append(res_N['first_F'])
                if res_N['eb_converged']:
                    conv_A['N'] += 1
            except Exception:
                pass

        # Print A-S, A-F, A-N rows
        for key, label in [('S', 'A-S'), ('F', 'A-F'), ('N', 'A-N')]:
            est_name = {'S': 'S (simplified)', 'F': 'F (full+M6)',
                        'N': 'N (wrong e_b)'}[key]
            arr = np.array(results_A[key]) if results_A[key] else np.array([])
            if len(arr) > 0:
                mean_tau = float(np.mean(arr))
                bias = mean_tau - TRUE_ATT
                sd = float(np.std(arr))
                rmse = float(np.sqrt(bias**2 + sd**2))
                mean_F_stat = float(np.mean(f_stats_A[key]))
                conv_pct = 100.0 * conv_A[key] / len(arr)
            else:
                mean_tau = bias = sd = rmse = mean_F_stat = float('nan')
                conv_pct = 0.0

            sf_str = ''
            if key == 'S' and n in sf_diffs and sf_diffs[n]:
                mean_sf = float(np.mean(np.abs(sf_diffs[n])))
                sf_str = f'{mean_sf:10.5f}'
            else:
                sf_str = f'{"—":>10}'

            print(f"{label:>5}  {est_name:>16}  {n:>6}  {mean_tau:>8.4f}  "
                  f"{bias:>+8.4f}  {sd:>7.4f}  {rmse:>7.4f}  "
                  f"{mean_F_stat:>7.1f}  {conv_pct:>4.0f}%  {sf_str}")

        # Case C: near-miss e_b
        results_C = []
        f_stats_C = []
        conv_C = 0
        for rep_idx in range(R):
            rng_rep = np.random.default_rng(rep_seeds[rep_idx] + 10**6)
            Y, D, X = dgp_surgical(rng_rep, n, kappa=0.05)
            try:
                res = triply_robust_iv(Y, D, X, ea_basis='X',
                                       eb_basis='expX', or_columns='X')
                results_C.append(res['tau_att'])
                f_stats_C.append(res['first_F'])
                if res['eb_converged']:
                    conv_C += 1
            except Exception:
                pass

        arr_C = np.array(results_C) if results_C else np.array([])
        if len(arr_C) > 0:
            mean_tau_C = float(np.mean(arr_C))
            bias_C = mean_tau_C - TRUE_ATT
            sd_C = float(np.std(arr_C))
            rmse_C = float(np.sqrt(bias_C**2 + sd_C**2))
            mean_F_C = float(np.mean(f_stats_C))
            conv_pct_C = 100.0 * conv_C / len(arr_C)
        else:
            mean_tau_C = bias_C = sd_C = rmse_C = mean_F_C = float('nan')
            conv_pct_C = 0.0

        print(f"{'C':>5}  {'S (near-miss)':>16}  {n:>6}  {mean_tau_C:>8.4f}  "
              f"{bias_C:>+8.4f}  {sd_C:>7.4f}  {rmse_C:>7.4f}  "
              f"{mean_F_C:>7.1f}  {conv_pct_C:>4.0f}%  {'—':>10}")

        print(sep)

    # --- Summary of S-F convergence ---
    print("\n--- S vs F Convergence (Case A) ---")
    print(f"{'n':>6}  {'Mean|τ̂_S - τ̂_F|':>18}  {'SD(τ̂_S - τ̂_F)':>16}  "
          f"{'Max|τ̂_S - τ̂_F|':>18}")
    for n_val in sample_sizes:
        if n_val in sf_diffs and sf_diffs[n_val]:
            diffs = np.array(sf_diffs[n_val])
            print(f"{n_val:>6}  {np.mean(np.abs(diffs)):>18.6f}  "
                  f"{np.std(diffs):>16.6f}  {np.max(np.abs(diffs)):>18.6f}")

    print(f"\nExpected: S bias → 0 as n grows (Cases A-S).")
    print(f"          |S - F| → 0 as n grows.")
    print(f"          N biased at all n (Cases A-N / B).")
    print(f"          C shows mild degradation from κ={0.05}.")

    # --- Plots ---
    if R >= 5:
        _make_surgical_plots(sf_diffs, sample_sizes, R)

    return sf_diffs, diag


def _make_surgical_plots(sf_diffs, sample_sizes, R):
    """Generate bias-vs-n and S-F difference plots."""

    # Bias vs 1/sqrt(n) plot placeholder — we need to re-run to collect
    # this data. For now, just plot S-F differences.

    if not sf_diffs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: |S - F| differences by n
    ax = axes[0]
    ns = sorted(sf_diffs.keys())
    means = [np.mean(np.abs(sf_diffs[n])) for n in ns]
    ax.plot(ns, means, 'bo-', markersize=8, linewidth=2)
    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Mean |τ̂_S − τ̂_F|')
    ax.set_title('S vs F: Estimator Difference (Case A)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Bias vs 1/sqrt(n) — we don't have full data here,
    # so show box plots of S-F differences
    ax = axes[1]
    bp_data = [sf_diffs[n] for n in ns if n in sf_diffs]
    bp_labels = [str(n) for n in ns if n in sf_diffs]
    if bp_data:
        ax.boxplot(bp_data, tick_labels=bp_labels)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Sample size n')
        ax.set_ylabel('τ̂_S − τ̂_F')
        ax.set_title('Distribution of S−F Difference (Case A)')

    fig.tight_layout()
    fig.savefig('surgical_sf_diff.png', dpi=150)
    print(f"\nS-F difference plot saved to surgical_sf_diff.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    sizes = [500, 2000, max_n] if max_n > 2000 else [500, max_n]
    run_simulation(R=R, sample_sizes=sizes, seed=42)
