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

def triply_robust_att(Y, D, X, alpha_oracle=None):
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
    alpha_oracle : array-like of length 2, optional
        If provided, (alpha0, alpha1) are fixed to these oracle values and M1/M2
        are dropped.  Only M3–M7 are solved for (gamma0, gamma1, beta0, beta1, phi).
        Useful for isolating whether DGP-3 failures stem from logit estimation
        of e_a or from a deeper issue in the remaining moment conditions (§4.2).

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
        oracle_alpha : bool — True when alpha was pinned to supplied oracle values
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)

    logX = np.log(np.maximum(X, 1e-300))   # guard log(0)
    sinX = np.sin(X)
    ctrl = (D == 0)   # boolean mask for control units

    use_oracle_alpha = alpha_oracle is not None
    if use_oracle_alpha:
        a0_fixed, a1_fixed = float(alpha_oracle[0]), float(alpha_oracle[1])

    # ------------------------------------------------------------------
    # Step 1: Starting values
    # ------------------------------------------------------------------

    # a) Logit of D on sin(X) -> alpha starting values (skipped when oracle)
    if not use_oracle_alpha:
        try:
            Xa = sm.add_constant(sinX)
            logit_mod = Logit(D, Xa).fit(disp=False, maxiter=200)
            alpha0_init, alpha1_init = logit_mod.params
        except Exception:
            alpha0_init, alpha1_init = 0.0, 0.0
    else:
        alpha0_init, alpha1_init = a0_fixed, a1_fixed

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

    # ------------------------------------------------------------------
    # Step 2: Solve moment conditions
    # ------------------------------------------------------------------
    # Full mode  : 7 equations, 7 unknowns (a0,a1,g0,g1,b0,b1,phi)
    # Oracle mode: 5 equations, 5 unknowns (g0,g1,b0,b1,phi); alpha fixed
    # ------------------------------------------------------------------

    if not use_oracle_alpha:
        theta0 = np.array([alpha0_init, alpha1_init,
                           gamma0_init, gamma1_init,
                           beta0_init, beta1_init,
                           phi_init])

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

    else:
        # Oracle mode: alpha pinned, solve 5-equation system for (g0,g1,b0,b1,phi)
        theta0 = np.array([gamma0_init, gamma1_init,
                           beta0_init, beta1_init,
                           phi_init])

        ea_fixed = _clip_ps(_invlogit(a0_fixed + a1_fixed * sinX))
        odds_a_fixed = ea_fixed / (1.0 - ea_fixed)

        def moment_conditions(theta):
            g0, g1, b0, b1, phi = theta

            idx_g = g0 + g1 * X
            eb = _clip_ps(norm.cdf(idx_g))

            m0 = b0 + b1 * logX + phi / (1.0 - eb)
            R = Y - m0

            odds_b = eb / (1.0 - eb)

            w = 1.0 - D   # indicator for control units

            M3 = np.mean(w * R)
            M4 = np.mean(w * logX * R)
            M5 = np.mean(w * odds_a_fixed * R)
            M6 = np.mean(w * (odds_b - odds_a_fixed) * R)
            M7 = np.mean(w * X * (odds_b - odds_a_fixed) * R)

            return np.array([M3, M4, M5, M6, M7])

    # Convergence tolerance: accept if norm of moments is below this threshold
    _CONV_TOL = 1e-4

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

    # ------------------------------------------------------------------
    # Step 3: Unpack solution and compute the ATT
    # ------------------------------------------------------------------

    if not use_oracle_alpha:
        a0_h, a1_h, g0_h, g1_h, b0_h, b1_h, phi_h = theta_hat
    else:
        a0_h, a1_h = a0_fixed, a1_fixed
        g0_h, g1_h, b0_h, b1_h, phi_h = theta_hat

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
        'oracle_alpha': use_oracle_alpha,
    }
