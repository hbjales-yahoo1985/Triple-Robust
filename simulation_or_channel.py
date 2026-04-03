"""
Monte Carlo: OR Channel Test for Estimator G
=============================================
Tests whether Estimator G (exactly-identified, M6 replaces M3) recovers
a consistent ATT when ONLY the outcome regression is correct.

DGP:
  X ~ Uniform[-2, 2]
  e_true = 0.3 + 0.4 * exp(-X**2)  (bell-shaped, non-monotone)
  mu_0(X) = beta_0_true + 1.5 * X   (linear => OR q(X)=X is correct)
  tau = 2.0

Model specs:
  ea: logit on X          -> WRONG (true PS is non-monotone)
  eb: log-logistic on exp(X) -> WRONG (same reason)
  OR: q(X) = X            -> CORRECT

Estimators:
  G   — Exactly-identified IV: instruments {ob-Z, q, Z} for {1, q, W}
  S   — Simplified: instruments {1, q, Z} for {1, q, W}
  F   — Full (overidentified): instruments {1, q, Z, ob-Z} for {1, q, W}
  OLS — Plain OLS Y~{1,X} among controls (oracle exploiting correct OR)

Run:
    python simulation_or_channel.py           # R=10 smoke test
    python simulation_or_channel.py 500       # R=500 full run
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from triple_robust_iv import (triply_robust_iv, triply_robust_iv_full,
                               triply_robust_iv_centered, _clip_ps)

TRUE_ATT = 2.0


# ---------------------------------------------------------------------------
# DGP: only OR is correct
# ---------------------------------------------------------------------------
def dgp_or_only(rng, n, beta0=3.0):
    """
    DGP where ONLY the outcome regression q(X) = X is correct.

    True PS is bell-shaped (non-monotone), so logit-on-X and
    log-logistic-on-exp(X) are both genuinely wrong.
    True mu_0 = beta0 + 1.5 X is linear.
    """
    X = rng.uniform(-2, 2, n)

    # True propensity score: bell-shaped, non-monotone
    e_true = 0.3 + 0.4 * np.exp(-X ** 2)
    e_true = _clip_ps(e_true)
    D = rng.binomial(1, e_true, n).astype(float)

    # True outcome model: LINEAR
    mu0 = beta0 + 1.5 * X
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = Y0 + TRUE_ATT
    Y = D * Y1 + (1.0 - D) * Y0

    return Y, D, X


# ---------------------------------------------------------------------------
# OLS oracle estimator
# ---------------------------------------------------------------------------
def ols_oracle(Y, D, X):
    """OLS of Y on {1, X} among controls, then impute for treated."""
    ctrl = (D == 0)
    treated = (D == 1)
    X_ctrl = np.column_stack([np.ones(ctrl.sum()), X[ctrl]])
    coef, _, _, _ = np.linalg.lstsq(X_ctrl, Y[ctrl], rcond=None)
    X_all = np.column_stack([np.ones(len(X)), X])
    m0_hat = X_all @ coef
    tau = float(np.mean((Y - m0_hat)[treated]))
    return {'tau_att': tau, 'beta': coef, 'phi': 0.0}


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation(R=10, sample_sizes=None, seed=42, beta0=3.0):
    if sample_sizes is None:
        sample_sizes = [500, 2000, 10000]

    print("=" * 105)
    print(f"OR Channel Test: Estimator G vs S, F, OLS  (beta_0 = {beta0})")
    print(f"R={R}, n in {sample_sizes}, True ATT={TRUE_ATT}")
    print(f"DGP: e_true = 0.3+0.4*exp(-X^2), mu_0 = {beta0}+1.5X")
    print("=" * 105)

    header = (f"{'Est':>10}  {'n':>6}  {'Mean':>8}  {'Bias':>8}  "
              f"{'SD':>7}  {'RMSE':>7}  {'beta0':>8}  {'beta1':>8}  {'phi':>8}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    all_results = {}
    param_results = {}  # (est, n) -> list of (beta0, beta1, phi)

    for n in sample_sizes:
        rng_base = np.random.default_rng(seed + n)
        rep_seeds = rng_base.integers(0, 2**31, size=R)

        for est in ['G', 'S', 'F', 'OLS']:
            all_results[(est, n)] = []
            param_results[(est, n)] = []

        for rep in range(R):
            rng_rep = np.random.default_rng(rep_seeds[rep])
            Y, D, Xr = dgp_or_only(rng_rep, n, beta0=beta0)

            # Estimator G
            try:
                res = triply_robust_iv_centered(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                all_results[('G', n)].append(res['tau_att'])
                param_results[('G', n)].append(
                    (res['beta'][0], res['beta'][1], res['phi']))
            except Exception:
                pass

            # Estimator S
            try:
                res = triply_robust_iv(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                all_results[('S', n)].append(res['tau_att'])
                param_results[('S', n)].append(
                    (res['beta'][0], res['beta'][1], res['phi']))
            except Exception:
                pass

            # Estimator F
            try:
                res = triply_robust_iv_full(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                all_results[('F', n)].append(res['tau_att'])
                param_results[('F', n)].append(
                    (res['beta'][0], res['beta'][1], res['phi']))
            except Exception:
                pass

            # OLS oracle
            try:
                res = ols_oracle(Y, D, Xr)
                all_results[('OLS', n)].append(res['tau_att'])
                param_results[('OLS', n)].append(
                    (res['beta'][0], res['beta'][1], res['phi']))
            except Exception:
                pass

        for est in ['G', 'S', 'F', 'OLS']:
            arr = np.array(all_results[(est, n)])
            prm = param_results[(est, n)]
            if len(arr) == 0:
                continue
            mean_t = float(np.mean(arr))
            bias = mean_t - TRUE_ATT
            sd = float(np.std(arr))
            rmse = float(np.sqrt(bias**2 + sd**2))

            b0s = [p[0] for p in prm]
            b1s = [p[1] for p in prm]
            phis = [p[2] for p in prm]
            mb0 = float(np.mean(b0s))
            mb1 = float(np.mean(b1s))
            mphi = float(np.mean(phis))

            print(f"{est:>10}  {n:>6}  {mean_t:>8.4f}  {bias:>+8.4f}  "
                  f"{sd:>7.4f}  {rmse:>7.4f}  {mb0:>8.4f}  {mb1:>8.4f}  "
                  f"{mphi:>+8.4f}")

        print(sep)

    # --- Condition number of the IV system for G ---
    print("\n--- Condition Number of IV System for G ---")
    col2 = "Mean cond(IV'Reg)"
    print(f"{'n':>6}  {col2:>20}  {'Median':>12}")
    for n in [max(sample_sizes)]:
        rng_base = np.random.default_rng(seed + n)
        rep_seeds = rng_base.integers(0, 2**31, size=min(R, 50))
        conds = []
        for rep in range(min(R, 50)):
            rng_rep = np.random.default_rng(rep_seeds[rep])
            Y, D, Xr = dgp_or_only(rng_rep, n, beta0=beta0)
            try:
                res = triply_robust_iv_centered(
                    Y, D, Xr, ea_basis='X', eb_basis='expX', or_columns='X')
                # Reconstruct to compute condition number
                from statsmodels.discrete.discrete_model import Logit
                import statsmodels.api as sm
                from triple_robust_iv import _invlogit, _log_logistic_mle

                ea_feature = Xr
                Xa = sm.add_constant(ea_feature)
                logit_mod = Logit(D, Xa).fit(disp=False, maxiter=200)
                ea_hat = _clip_ps(_invlogit(logit_mod.params[0]
                                            + logit_mod.params[1] * ea_feature))
                Za = ea_hat / (1.0 - ea_hat)

                eb_feature = np.exp(Xr)
                gamma, _ = _log_logistic_mle(D, eb_feature)
                odds_b = np.maximum(gamma[0] + gamma[1] * eb_feature, 1e-10)
                W = 1.0 + odds_b
                telescope = odds_b - Za

                ctrl = (D == 0)
                iv_ctrl = np.column_stack([telescope[ctrl], Xr[ctrl], Za[ctrl]])
                reg_ctrl = np.column_stack([np.ones(ctrl.sum()), Xr[ctrl], W[ctrl]])
                IvR = iv_ctrl.T @ reg_ctrl
                conds.append(float(np.linalg.cond(IvR)))
            except Exception:
                pass
        if conds:
            print(f"{n:>6}  {np.mean(conds):>20.2f}  {np.median(conds):>12.2f}")

    # --- Plots ---
    if R >= 5:
        _make_plots(all_results, sample_sizes, beta0)

    return all_results


def _make_plots(all_results, sample_sizes, beta0):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = {'G': 'tab:blue', 'S': 'tab:orange', 'F': 'tab:green',
              'OLS': 'tab:purple'}
    labels = {'G': 'G (M6 replaces M3)', 'S': 'S (simplified)',
              'F': 'F (full+M6)', 'OLS': 'OLS oracle'}
    for est in ['G', 'S', 'F', 'OLS']:
        biases, ns = [], []
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
    ax.set_title(f'Bias vs n (OR channel, beta_0={beta0})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bp_data, bp_labels = [], []
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
        ax.set_title('G - F difference (OR channel)')

    fig.tight_layout()
    fig.savefig('or_channel_monte_carlo.png', dpi=150)
    print(f"\nPlot saved to or_channel_monte_carlo.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Robustness sweep: vary beta_0
# ---------------------------------------------------------------------------
def run_intercept_sweep(R=10, n=10000, seed=42):
    """Vary beta_0 to test intercept recovery robustness."""
    print("\n" + "=" * 90)
    print(f"Intercept Sweep: beta_0 in {{0, 1, 5, 10}}, n={n}, R={R}")
    print("=" * 90)

    header = (f"{'beta0':>6}  {'Est':>5}  {'Bias':>8}  "
              f"{'SD':>7}  {'mean_b0':>8}  {'mean_b1':>8}  {'mean_phi':>8}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for b0 in [0.0, 1.0, 5.0, 10.0]:
        rng_base = np.random.default_rng(seed + n + int(b0 * 100))
        rep_seeds = rng_base.integers(0, 2**31, size=R)

        ests_data = {e: {'taus': [], 'b0s': [], 'b1s': [], 'phis': []}
                     for e in ['G', 'S', 'F', 'OLS']}

        for rep in range(R):
            rng_rep = np.random.default_rng(rep_seeds[rep])
            Y, D, Xr = dgp_or_only(rng_rep, n, beta0=b0)

            for est_name, func, kwargs in [
                ('G', triply_robust_iv_centered,
                 dict(ea_basis='X', eb_basis='expX', or_columns='X')),
                ('S', triply_robust_iv,
                 dict(ea_basis='X', eb_basis='expX', or_columns='X')),
                ('F', triply_robust_iv_full,
                 dict(ea_basis='X', eb_basis='expX', or_columns='X')),
                ('OLS', ols_oracle, {}),
            ]:
                try:
                    if est_name == 'OLS':
                        res = func(Y, D, Xr)
                    else:
                        res = func(Y, D, Xr, **kwargs)
                    ests_data[est_name]['taus'].append(res['tau_att'])
                    ests_data[est_name]['b0s'].append(res['beta'][0])
                    ests_data[est_name]['b1s'].append(res['beta'][1])
                    ests_data[est_name]['phis'].append(res['phi'])
                except Exception:
                    pass

        for est in ['G', 'S', 'F', 'OLS']:
            d = ests_data[est]
            if not d['taus']:
                continue
            arr = np.array(d['taus'])
            bias = float(np.mean(arr)) - TRUE_ATT
            sd = float(np.std(arr))
            mb0 = float(np.mean(d['b0s']))
            mb1 = float(np.mean(d['b1s']))
            mphi = float(np.mean(d['phis']))
            print(f"{b0:>6.1f}  {est:>5}  {bias:>+8.4f}  "
                  f"{sd:>7.4f}  {mb0:>8.4f}  {mb1:>8.4f}  {mphi:>+8.4f}")

        print(sep)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_simulation(R=R, sample_sizes=[500, 2000, 10000], seed=42, beta0=3.0)
    run_intercept_sweep(R=R, n=10000, seed=42)
