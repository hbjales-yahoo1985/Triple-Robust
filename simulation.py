"""
Simulation Harness — Four DGPs
================================
Verifies triple robustness of the triply_robust_att estimator.

Run:
    python simulation.py

The script runs R = 1000 Monte Carlo replications for each of four DGPs
(n = 2000 per replication) and prints a summary table comparing:
  - Triply Robust (TR) estimator  [sequential two-step: logit MLE for alpha,
                                    then solve M3-M7 for the rest]
  - Naive OLS imputation
  - Standard AIPW (logit-on-sin PS only)
  - IPW (logit-on-sin PS only)
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from scipy.stats import norm
from triple_robust import triply_robust_att, _invlogit, _clip_ps

# ---------------------------------------------------------------------------
# DGP definitions
# ---------------------------------------------------------------------------

def dgp1(rng, n):
    """OR correct, both PS wrong. True ATT = 3."""
    X = rng.uniform(10, 20, n)
    e0 = _clip_ps(_invlogit(-1 + 0.3 * X - 0.01 * X ** 2))
    D = rng.binomial(1, e0, n)
    mu0 = 2 + 3 * np.log(X)
    mu1 = 5 + 3 * np.log(X)
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = mu1 + rng.normal(0, 1, n)
    Y = D * Y1 + (1 - D) * Y0
    return Y, D, X


def dgp2(rng, n):
    """e_b (probit) correct, OR and e_a wrong. True ATT = 5."""
    X = rng.uniform(10, 20, n)
    e0 = _clip_ps(norm.cdf(-2.5 + 0.2 * X))
    D = rng.binomial(1, e0, n)
    mu0 = 10 + 0.5 * X + 0.02 * X ** 2
    mu1 = 15 + 0.5 * X + 0.02 * X ** 2
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = mu1 + rng.normal(0, 1, n)
    Y = D * Y1 + (1 - D) * Y0
    return Y, D, X


def dgp3(rng, n):
    """e_a (logit-on-sin) correct, OR and e_b wrong. True ATT = 4.
    True alpha: (0.0, -1.5) -> propensities in ~[0.18, 0.82] for good common support.
    """
    X = rng.uniform(10, 20, n)
    e0 = _clip_ps(_invlogit(0.0 - 1.5 * np.sin(X)))
    D = rng.binomial(1, e0, n)
    mu0 = 5 * np.sin(X) + 0.3 * X
    mu1 = 5 * np.sin(X) + 0.3 * X + 4
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = mu1 + rng.normal(0, 1, n)
    Y = D * Y1 + (1 - D) * Y0
    return Y, D, X


def dgp4(rng, n):
    """All three models misspecified (negative control). True ATT = 5."""
    X = rng.uniform(10, 20, n)
    e0 = _clip_ps(_invlogit(-1 + 0.3 * X - 0.01 * X ** 2))
    D = rng.binomial(1, e0, n)
    mu0 = 10 + 0.5 * X + 0.02 * X ** 2
    mu1 = 15 + 0.5 * X + 0.02 * X ** 2
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = mu1 + rng.normal(0, 1, n)
    Y = D * Y1 + (1 - D) * Y0
    return Y, D, X


# ---------------------------------------------------------------------------
# Benchmark estimators
# ---------------------------------------------------------------------------

def ols_imputation(Y, D, X):
    """Naive OLS imputation: regress Y on log(X) among controls, impute."""
    logX = np.log(X)
    ctrl = (D == 0)
    Xc = sm.add_constant(logX[ctrl])
    ols = sm.OLS(Y[ctrl], Xc).fit()
    Xall = sm.add_constant(logX)
    m0_hat = ols.predict(Xall)
    treated = (D == 1)
    return float(np.mean((Y - m0_hat)[treated]))


def aipw_logit_sin(Y, D, X):
    """Standard AIPW using logit-on-sin(X) as the propensity score."""
    sinX = np.sin(X)
    logX = np.log(X)
    ctrl = (D == 0)
    treated = (D == 1)
    n = len(Y)
    p_treat = np.mean(D)

    # Fit PS
    try:
        Xa = sm.add_constant(sinX)
        e_hat = _clip_ps(Logit(D, Xa).fit(disp=False).predict(Xa))
    except Exception:
        e_hat = np.full(n, p_treat)

    # Fit OR on controls
    try:
        Xc = sm.add_constant(logX[ctrl])
        ols = sm.OLS(Y[ctrl], Xc).fit()
        m0_hat = ols.predict(sm.add_constant(logX))
    except Exception:
        m0_hat = np.full(n, float(np.mean(Y[ctrl])))

    # AIPW-ATT estimator
    ipw_weight = D / p_treat - (1 - D) * e_hat / ((1 - e_hat) * p_treat)
    tau = np.mean(ipw_weight * (Y - m0_hat)) + np.mean(m0_hat[treated]) - np.mean(m0_hat[treated])
    # Standard AIPW-ATT formula:
    tau = (np.mean(D * (Y - m0_hat)) / p_treat
           - np.mean((1 - D) * e_hat / (1 - e_hat) * (Y - m0_hat)) / p_treat)
    return float(tau)


def ipw_logit_sin(Y, D, X):
    """IPW estimator using logit-on-sin(X)."""
    sinX = np.sin(X)
    n = len(Y)
    p_treat = np.mean(D)
    try:
        Xa = sm.add_constant(sinX)
        e_hat = _clip_ps(Logit(D, Xa).fit(disp=False).predict(Xa))
    except Exception:
        e_hat = np.full(n, p_treat)

    tau = (np.mean(D * Y) / p_treat
           - np.mean((1 - D) * e_hat / (1 - e_hat) * Y) / p_treat)
    return float(tau)


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def run_replication(dgp_fn, rng, n):
    Y, D, X = dgp_fn(rng, n)

    # Triply robust
    result = triply_robust_att(Y, D, X)
    tau_tr = result['tau_att']
    conv = result['converged']

    # Benchmarks
    tau_ols = ols_imputation(Y, D, X)
    tau_aipw = aipw_logit_sin(Y, D, X)
    tau_ipw = ipw_logit_sin(Y, D, X)

    return tau_tr, conv, tau_ols, tau_aipw, tau_ipw


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(R=1000, n=2000, seed=42):
    dgps = [
        (dgp1, 3.0, 'OR',   1),
        (dgp2, 5.0, 'e_b',  2),
        (dgp3, 4.0, 'e_a',  3),
        (dgp4, 5.0, 'None', 4),
    ]

    print(f"Running simulation: R={R} replications, n={n} per replication\n")

    header = (f"{'DGP':>4}  {'True ATT':>8}  {'Correct':>7}  "
              f"{'TR Bias':>8}  {'TR RMSE':>8}  {'TR SD':>7}  {'TR Conv%':>8}  "
              f"{'OLS Bias':>9}  {'AIPW Bias':>10}  {'IPW Bias':>9}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for dgp_fn, true_att, correct_model, dgp_id in dgps:
        rng = np.random.default_rng(seed + dgp_id)

        tr_list, ols_list, aipw_list, ipw_list = [], [], [], []
        n_converged = 0

        for _ in range(R):
            tau_tr, conv, tau_ols, tau_aipw, tau_ipw = run_replication(
                dgp_fn, rng, n)
            ols_list.append(tau_ols)
            aipw_list.append(tau_aipw)
            ipw_list.append(tau_ipw)
            if conv:
                tr_list.append(tau_tr)
                n_converged += 1

        conv_pct = 100.0 * n_converged / R

        if tr_list:
            tr_arr = np.array(tr_list)
            tr_bias = float(np.mean(tr_arr) - true_att)
            tr_sd = float(np.std(tr_arr))
            tr_rmse = float(np.sqrt(tr_bias ** 2 + tr_sd ** 2))
        else:
            tr_bias = tr_sd = tr_rmse = float('nan')

        ols_bias = float(np.mean(ols_list) - true_att)
        aipw_bias = float(np.mean(aipw_list) - true_att)
        ipw_bias = float(np.mean(ipw_list) - true_att)

        print(f"{dgp_id:>4}  {true_att:>8.1f}  {correct_model:>7}  "
              f"{tr_bias:>+8.4f}  {tr_rmse:>8.4f}  {tr_sd:>7.4f}  {conv_pct:>7.1f}%  "
              f"{ols_bias:>+9.4f}  {aipw_bias:>+10.4f}  {ipw_bias:>+9.4f}")

    print(sep)
    print("\nExpected: TR Bias ~0 for DGPs 1-3, non-zero for DGP 4.")
    print("DGP 2 is the sharpest test: only TR should be unbiased.\n")


def run_dgp3_oracle_diagnostic(R=1000, n=2000, seed=42):
    """
    Oracle-alpha diagnostic for DGP 3 (§4.2 Bang-Robins channel).

    Both variants use the sequential two-step estimator (logit MLE for alpha,
    then solve M3-M7).  They differ only in where alpha comes from:

      - TR (estimated alpha) : alpha estimated by logit MLE  [standard mode]
      - TR (oracle alpha)    : alpha fixed to the DGP-3 true values (1.5, -0.8)

    Comparing the two isolates whether any remaining bias/non-convergence in
    DGP 3 is caused by error in the logit MLE of alpha, or by a structural
    problem in the M3-M7 sub-system itself.

    True alpha: alpha0=0.0, alpha1=-1.5  (DGP 3 true propensity e_a parameters)
    True ATT  : 4.0
    """
    TRUE_ATT = 4.0
    ALPHA_TRUE = (0.0, -1.5)

    print("\n" + "=" * 70)
    print("DGP 3 Oracle-Alpha Diagnostic (§4.2 Bang-Robins channel)")
    print(f"  Oracle alpha = {ALPHA_TRUE},  True ATT = {TRUE_ATT}")
    print(f"  R={R} replications, n={n}")
    print("=" * 70)

    rng = np.random.default_rng(seed + 3)

    oracle_list, est_list = [], []
    n_oracle_conv = 0
    n_est_conv = 0

    for _ in range(R):
        Y, D, X = dgp3(rng, n)

        # Oracle: skip logit MLE, pin alpha to DGP-3 truth
        res_oracle = triply_robust_att(Y, D, X, alpha_oracle=ALPHA_TRUE)
        if res_oracle['converged']:
            oracle_list.append(res_oracle['tau_att'])
            n_oracle_conv += 1

        # Standard: logit MLE for alpha (two-step sequential)
        res_est = triply_robust_att(Y, D, X)
        if res_est['converged']:
            est_list.append(res_est['tau_att'])
            n_est_conv += 1

    oracle_conv_pct = 100.0 * n_oracle_conv / R
    est_conv_pct = 100.0 * n_est_conv / R

    def _summary(lst, label, conv_pct):
        if lst:
            arr = np.array(lst)
            bias = float(np.mean(arr) - TRUE_ATT)
            sd = float(np.std(arr))
            rmse = float(np.sqrt(bias ** 2 + sd ** 2))
        else:
            bias = sd = rmse = float('nan')
        print(f"  {label:<26}  Bias={bias:+.4f}  RMSE={rmse:.4f}  SD={sd:.4f}  "
              f"Conv={conv_pct:.1f}%")

    _summary(oracle_list, "TR (oracle alpha)",    oracle_conv_pct)
    _summary(est_list,    "TR (estimated alpha)", est_conv_pct)

    print()
    if oracle_conv_pct > 50 and oracle_list:
        oracle_bias = abs(np.mean(oracle_list) - TRUE_ATT)
        if oracle_bias < 0.1:
            print("  => Oracle α: small bias.  M3-M7 sub-system is sound.")
            if est_conv_pct < oracle_conv_pct - 10 or (
                    est_list and abs(np.mean(est_list) - TRUE_ATT) > 0.2):
                print("     Estimated α: worse performance — logit MLE of alpha is the")
                print("     limiting factor for DGP 3, not the M3-M7 moment structure.")
            else:
                print("     Estimated α performs similarly — logit MLE is reliable here.")
        else:
            print("  => Oracle α converges but bias is non-trivial: investigate")
            print("     the M3-M7 sub-system itself (possible proof/design issue).")
    else:
        print("  => Oracle α also fails: the issue is in M3-M7, not the logit step.")
        print("     This may indicate a structural problem beyond alpha estimation.")
    print()


if __name__ == '__main__':
    import sys
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    run_simulation(R=R, n=n)
    run_dgp3_oracle_diagnostic(R=R, n=n)
