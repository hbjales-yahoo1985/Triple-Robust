"""
Monte Carlo: Estimator G (exactly-identified, M6 replaces M3) vs S, F, N
=========================================================================
Surgical DGP where only e_b is correct:
  X ~ Uniform[-2, 2]
  odds = 1.5 + 0.8 exp(X)  (log-logistic, linear in exp(X))
  mu_0(X) = 1 + 0.7 X^2 + 0.5 sin(2X)
  tau = 2 (constant treatment effect)

Estimation:
  ea: logit on X  (wrong)
  eb: log-logistic on exp(X)  (correct)
  OR: beta_0 + beta_1 X  (wrong)

Estimators:
  G — Exactly-identified IV: instruments {ob-Z, q, Z} for {1, q, W}
      (replaces constant instrument M3 with telescope M6)
  S — Simplified: instruments {1, q, Z} for {1, q, W}
  F — Full (overidentified): instruments {1, q, Z, ob-Z} for {1, q, W}
  N — Negative control: S with wrong e_b basis (X instead of exp(X))

Run:
    python simulation_centered.py           # R=10 smoke test
    python simulation_centered.py 500       # R=500 full run
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from triple_robust_iv import (triply_robust_iv, triply_robust_iv_full,
                               triply_robust_iv_centered, _clip_ps)

# ---------------------------------------------------------------------------
TRUE_ATT = 2.0
DELTA_0, DELTA_1 = 1.5, 0.8


def dgp_surgical(rng, n, kappa=0.0):
    """DGP where only e_b (linear odds in exp(X)) is correct."""
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


def run_diagnostics(n_large=200_000, seed=999):
    """Span check and e_a misspecification check."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, n_large)
    mu0 = 1.0 + 0.7 * X ** 2 + 0.5 * np.sin(2.0 * X)

    # Span check: regress mu_0 on {1, X, exp(X)}
    R_span = np.column_stack([np.ones(n_large), X, np.exp(X)])
    coef, _, _, _ = np.linalg.lstsq(R_span, mu0, rcond=None)
    mu0_hat = R_span @ coef
    ss_res = np.sum((mu0 - mu0_hat) ** 2)
    ss_tot = np.sum((mu0 - np.mean(mu0)) ** 2)
    R2_span = 1.0 - ss_res / ss_tot

    # e_a misspecification: true PS vs logit-on-X
    odds_true = DELTA_0 + DELTA_1 * np.exp(X)
    e_true = odds_true / (1.0 + odds_true)
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

    return {'R2_span': R2_span, 'mae_ea': mae_ea, 'corr_ea': corr_ea}


def run_simulation(R=10, sample_sizes=None, seed=42):
    """Run the surgical Monte Carlo comparing G, S, F, N."""
    if sample_sizes is None:
        sample_sizes = [500, 2000, 10000]

    print("=" * 100)
    print("Monte Carlo: Estimator G (exactly-identified, M6 replaces M3) vs S, F, N")
    print(f"R={R} replications, n in {sample_sizes}, True ATT={TRUE_ATT}")
    print("=" * 100)

    # Diagnostics
    print("\n--- Diagnostics ---")
    diag = run_diagnostics()
    print(f"  Span R^2 (mu_0 on {{1, X, exp(X)}}): {diag['R2_span']:.4f}")
    print(f"  e_a misspec: MAE={diag['mae_ea']:.4f}, corr={diag['corr_ea']:.4f}")
    print()

    # Header
    header = (f"{'Est':>18}  {'n':>6}  {'Mean':>8}  {'Bias':>8}  "
              f"{'SD':>7}  {'RMSE':>7}  {'1stF':>7}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    # Collect results for plotting
    all_results = {}  # (estimator, n) -> list of tau_hat
    f_stats_all = {}  # (estimator, n) -> list

    for n in sample_sizes:
        rng_base = np.random.default_rng(seed + n)
        rep_seeds = rng_base.integers(0, 2**31, size=R)

        for est in ['G', 'S', 'F', 'N']:
            all_results[(est, n)] = []
            f_stats_all[(est, n)] = []

        for rep in range(R):
            rng_rep = np.random.default_rng(rep_seeds[rep])
            Y, D, Xr = dgp_surgical(rng_rep, n, kappa=0.0)

            # Estimator G (exactly-identified, M6 replaces M3)
            try:
                res_G = triply_robust_iv_centered(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                all_results[('G', n)].append(res_G['tau_att'])
                f_stats_all[('G', n)].append(res_G['first_F'])
            except Exception:
                pass

            # Estimator S (simplified, M3+M4+M5)
            try:
                res_S = triply_robust_iv(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                all_results[('S', n)].append(res_S['tau_att'])
                f_stats_all[('S', n)].append(res_S['first_F'])
            except Exception:
                pass

            # Estimator F (full, overidentified M3+M4+M5+M6)
            try:
                res_F = triply_robust_iv_full(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                all_results[('F', n)].append(res_F['tau_att'])
                f_stats_all[('F', n)].append(res_F['first_F'])
            except Exception:
                pass

            # Estimator N (negative control: wrong e_b)
            try:
                res_N = triply_robust_iv(
                    Y, D, Xr, ea_basis='X', eb_basis='X', or_columns='X')
                all_results[('N', n)].append(res_N['tau_att'])
                f_stats_all[('N', n)].append(res_N['first_F'])
            except Exception:
                pass

        # Print rows
        for est in ['G', 'S', 'F', 'N']:
            arr = np.array(all_results[(est, n)])
            if len(arr) == 0:
                continue
            mean_t = float(np.mean(arr))
            bias = mean_t - TRUE_ATT
            sd = float(np.std(arr))
            rmse = float(np.sqrt(bias**2 + sd**2))
            f_arr = f_stats_all[(est, n)]
            mean_F_stat = float(np.mean(f_arr)) if f_arr else 0

            name = {'G': 'G (M6 replaces M3)',
                    'S': 'S (simplified)',
                    'F': 'F (full+M6)',
                    'N': 'N (wrong e_b)'}[est]

            print(f"{name:>18}  {n:>6}  {mean_t:>8.4f}  {bias:>+8.4f}  "
                  f"{sd:>7.4f}  {rmse:>7.4f}  {mean_F_stat:>7.1f}")

        print(sep)

    # --- Pairwise comparison table ---
    print("\n--- Pairwise Differences (Case A: only e_b correct) ---")
    print(f"{'n':>6}  {'Mean(G-F)':>10}  {'SD(G-F)':>10}  "
          f"{'Mean(G-S)':>10}  {'Mean(S-F)':>10}")
    for n in sample_sizes:
        G_arr = np.array(all_results[('G', n)])
        S_arr = np.array(all_results[('S', n)])
        F_arr = np.array(all_results[('F', n)])
        k = min(len(G_arr), len(S_arr), len(F_arr))
        if k > 0:
            gf = G_arr[:k] - F_arr[:k]
            gs = G_arr[:k] - S_arr[:k]
            sf = S_arr[:k] - F_arr[:k]
            print(f"{n:>6}  {np.mean(gf):>+10.6f}  {np.std(gf):>10.6f}  "
                  f"{np.mean(gs):>+10.6f}  {np.mean(sf):>+10.6f}")

    # --- Plots ---
    if R >= 5:
        _make_plots(all_results, sample_sizes)

    return all_results


def _make_plots(all_results, sample_sizes):
    """Generate publication-quality figures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Figure 1: Bias vs n
    ax = axes[0]
    colors = {'G': 'tab:blue', 'S': 'tab:orange', 'F': 'tab:green', 'N': 'tab:red'}
    labels = {'G': 'G (M6 replaces M3)', 'S': 'S (simplified)',
              'F': 'F (full+M6)', 'N': 'N (wrong e_b)'}
    for est in ['G', 'S', 'F', 'N']:
        biases = []
        ns = []
        for n in sample_sizes:
            arr = np.array(all_results[(est, n)])
            if len(arr) > 0:
                biases.append(float(np.mean(arr)) - TRUE_ATT)
                ns.append(n)
        if ns:
            ax.plot(ns, biases, 'o-', color=colors[est], label=labels[est],
                    markersize=8, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Bias')
    ax.set_title('Bias vs Sample Size (Case A: only e_b correct)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Figure 2: Distribution of G - F
    ax = axes[1]
    bp_data = []
    bp_labels = []
    for n in sample_sizes:
        G_arr = np.array(all_results[('G', n)])
        F_arr = np.array(all_results[('F', n)])
        k = min(len(G_arr), len(F_arr))
        if k > 0:
            bp_data.append(G_arr[:k] - F_arr[:k])
            bp_labels.append(str(n))
    if bp_data:
        ax.boxplot(bp_data, tick_labels=bp_labels)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Sample size n')
        ax.set_ylabel(r'$\hat\tau_G - \hat\tau_F$')
        ax.set_title('Distribution of G - F')

    fig.tight_layout()
    fig.savefig('centered_monte_carlo.png', dpi=150)
    print(f"\nPlot saved to centered_monte_carlo.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_simulation(R=R, sample_sizes=[500, 2000, 10000], seed=42)