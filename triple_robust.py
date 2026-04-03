"""
Triply Robust ATT Estimator
===========================
Implements the triply robust estimator for the Average Treatment Effect on
the Treated (ATT) described in the specification.

The estimator is a pure imputation estimator whose imputation model m0_hat
is estimated via a GMM system that embeds two propensity score models into
the moment conditions.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import root, least_squares
from statsmodels.discrete.discrete_model import Logit, Probit
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _invlogit(z):
    """Logistic (sigmoid) function, numerically stable."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _clip_ps(p, eps=1e-6):
    """Clip propensity scores away from 0 and 1."""
    return np.clip(p, eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def triply_robust_att(Y, D, X):
    """
    Triply robust estimator for the Average Treatment Effect on the Treated.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Observed outcomes.
    D : array-like, shape (n,)
        Binary treatment indicators (1 = treated, 0 = control).
    X : array-like, shape (n,)
        Single pre-treatment covariate.

    Returns
    -------
    dict with keys:
        tau_att   : float  — ATT point estimate
        beta      : (beta0, beta1)  — outcome regression parameters
        phi       : float — augmentation coefficient
        alpha     : (alpha0, alpha1) — augmentation PS (logit) parameters
        gamma     : (gamma0, gamma1) — balance PS (probit) parameters
        m0_hat    : array, shape (n,) — fitted imputation values
        converged : bool
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)

    logX = np.log(np.maximum(X, 1e-300))   # guard log(0)
    sinX = np.sin(X)
    ctrl = (D == 0)   # boolean mask for control units

    # ------------------------------------------------------------------
    # Step 1: Starting values
    # ------------------------------------------------------------------

    # a) Logit of D on sin(X) -> alpha starting values
    try:
        Xa = sm.add_constant(sinX)
        logit_mod = Logit(D, Xa).fit(disp=False, maxiter=200)
        alpha0_init, alpha1_init = logit_mod.params
    except Exception:
        alpha0_init, alpha1_init = 0.0, 0.0

    # b) Probit of D on X -> gamma starting values
    try:
        Xb = sm.add_constant(X)
        probit_mod = Probit(D, Xb).fit(disp=False, maxiter=200)
        gamma0_init, gamma1_init = probit_mod.params
    except Exception:
        gamma0_init, gamma1_init = 0.0, 0.0

    # c) OLS of Y on log(X) among controls -> beta starting values
    try:
        Xc = sm.add_constant(logX[ctrl])
        ols_mod = sm.OLS(Y[ctrl], Xc).fit()
        beta0_init, beta1_init = ols_mod.params
    except Exception:
        beta0_init, beta1_init = float(np.mean(Y[ctrl])), 0.0

    # d) phi starting value
    phi_init = 0.0

    theta0 = np.array([alpha0_init, alpha1_init,
                       gamma0_init, gamma1_init,
                       beta0_init, beta1_init,
                       phi_init])

    # ------------------------------------------------------------------
    # Step 2: Solve the 7 moment conditions
    # ------------------------------------------------------------------

    def moment_conditions(theta):
        a0, a1, g0, g1, b0, b1, phi = theta

        idx_a = a0 + a1 * sinX
        ea = _clip_ps(_invlogit(idx_a))

        idx_g = g0 + g1 * X
        eb = _clip_ps(norm.cdf(idx_g))

        m0 = b0 + b1 * logX + phi / (1.0 - eb)
        R = Y - m0

        odds_a = ea / (1.0 - ea)
        odds_b = eb / (1.0 - eb)

        w = 1.0 - D   # indicator for control units

        M1 = np.mean((D - ea) * 1.0)
        M2 = np.mean((D - ea) * sinX)
        M3 = np.mean(w * R)
        M4 = np.mean(w * logX * R)
        M5 = np.mean(w * odds_a * R)
        M6 = np.mean(w * (odds_b - odds_a) * R)
        M7 = np.mean(w * X * (odds_b - odds_a) * R)

        return np.array([M1, M2, M3, M4, M5, M6, M7])

    # Convergence tolerance: accept if norm of moments is below this threshold
    _CONV_TOL = 1e-4

    def _is_converged(theta):
        return bool(np.linalg.norm(moment_conditions(theta)) < _CONV_TOL)

    best_theta = None
    best_norm = np.inf

    # --- Pass 1: Powell hybrid (fast, accurate) ---
    try:
        sol = root(moment_conditions, theta0, method='hybr',
                   options={'maxfev': 20_000, 'xtol': 1e-10})
        cand_norm = np.linalg.norm(moment_conditions(sol.x))
        if cand_norm < best_norm:
            best_norm = cand_norm
            best_theta = sol.x.copy()
    except Exception:
        pass

    # --- Pass 2: Levenberg-Marquardt root-finder ---
    if best_theta is None or best_norm >= _CONV_TOL:
        try:
            sol2 = root(moment_conditions, theta0, method='lm',
                        options={'maxiter': 20_000, 'col_deriv': 0})
            cand_norm = np.linalg.norm(moment_conditions(sol2.x))
            if cand_norm < best_norm:
                best_norm = cand_norm
                best_theta = sol2.x.copy()
        except Exception:
            pass

    # --- Pass 3: Levenberg-Marquardt least-squares minimiser ---
    if best_theta is None or best_norm >= _CONV_TOL:
        try:
            ls_sol = least_squares(moment_conditions, theta0, method='lm',
                                   max_nfev=100_000, ftol=1e-12, xtol=1e-12)
            cand_norm = np.linalg.norm(ls_sol.fun)
            if cand_norm < best_norm:
                best_norm = cand_norm
                best_theta = ls_sol.x.copy()
        except Exception:
            pass

    # --- Pass 4: Perturbed starting values (tiebreaker for stuck solvers) ---
    if best_theta is None or best_norm >= _CONV_TOL:
        rng_perturb = np.random.default_rng(0)
        for _ in range(5):
            theta_pert = theta0 + rng_perturb.normal(0, 0.5, size=len(theta0))
            try:
                sol_p = root(moment_conditions, theta_pert, method='hybr',
                             options={'maxfev': 10_000, 'xtol': 1e-10})
                cand_norm = np.linalg.norm(moment_conditions(sol_p.x))
                if cand_norm < best_norm:
                    best_norm = cand_norm
                    best_theta = sol_p.x.copy()
                    if best_norm < _CONV_TOL:
                        break
            except Exception:
                continue

    theta_hat = best_theta if best_theta is not None else theta0
    converged = bool(best_norm < _CONV_TOL)

    a0_h, a1_h, g0_h, g1_h, b0_h, b1_h, phi_h = theta_hat

    # ------------------------------------------------------------------
    # Step 3: Compute the ATT
    # ------------------------------------------------------------------

    eb_hat = _clip_ps(norm.cdf(g0_h + g1_h * X))
    m0_hat = b0_h + b1_h * logX + phi_h / (1.0 - eb_hat)

    tau_att = float(np.mean((Y - m0_hat)[D == 1]))

    return {
        'tau_att': tau_att,
        'beta': (b0_h, b1_h),
        'phi': phi_h,
        'alpha': (a0_h, a1_h),
        'gamma': (g0_h, g1_h),
        'm0_hat': m0_hat,
        'converged': converged,
    }
