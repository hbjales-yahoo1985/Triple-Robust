"""
Simulation Harness — Four DGPs
================================
Verifies triple robustness of the triply_robust_att estimator.

Run:
    python simulation.py

The script runs R = 1000 Monte Carlo replications for each of four DGPs
(n = 2000 per replication) and prints a summary table comparing:
  - Triply Robust (TR) estimator
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
    """e_a (logit-on-sin) correct, OR and e_b wrong. True ATT = 4."""
    X = rng.uniform(10, 20, n)
    e0 = _clip_ps(_invlogit(1.5 - 0.8 * np.sin(X)))
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


if __name__ == '__main__':
    import sys
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    run_simulation(R=R, n=n)
