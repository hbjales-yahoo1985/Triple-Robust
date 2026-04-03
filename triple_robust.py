"""
Triply Robust ATT Estimator
===========================
Implements the triply robust estimator for the Average Treatment Effect on
the Treated (ATT) described in the specification.

The estimator uses a sequential two-step procedure that respects the block-
triangular structure of the moment conditions:

  Step 1 - Estimate alpha via standard logit MLE.
            M1 = E[(D - e_a) * 1]  = 0
            M2 = E[(D - e_a) * sin(X)] = 0
            These two equations involve only (alpha0, alpha1) and are completely
            decoupled from all other parameters.  Standard MLE solves them
            optimally and there is no benefit to folding them into a larger system.

  Step 2 - With alpha fixed at alpha-hat, solve for (phi, gamma0, gamma1) via
            M5-M7, concentrating out (beta0, beta1) analytically via OLS (M3-M4).

            The key structural observation is that M3 and M4 are the OLS normal
            equations for regressing (Y - phi/(1 - e_b)) on (1, log(X)) among
            controls.  Given any trial (phi, gamma0, gamma1), they therefore
            determine (beta0, beta1) in closed form via OLS -- no solver needed.

            Substituting this concentrated solution back reduces the problem to
            finding (phi, gamma0, gamma1) that satisfy M5-M7:
              M5 = E[(1-D) * odds_a * R] = 0
              M6 = E[(1-D) * (odds_b - odds_a) * R] = 0
              M7 = E[(1-D) * X * (odds_b - odds_a) * R] = 0
            where R = Y - m0,  m0 = beta0(phi,g) + beta1(phi,g)*log(X) + phi/(1-e_b).

            This concentrated 3x3 system is smaller, better-conditioned, and
            faster to solve than the original 5x5 or 7x7 system.
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

    Uses a sequential two-step procedure:
      Step 1: estimate (alpha0, alpha1) via logit MLE on D ~ sin(X).
      Step 2: with alpha fixed, solve M5-M7 for (phi, gamma0, gamma1),
              concentrating out (beta0, beta1) analytically via OLS (M3-M4).

    Parameters
    ----------
    Y : array-like, shape (n,)
        Observed outcomes.
    D : array-like, shape (n,)
        Binary treatment indicators (1 = treated, 0 = control).
    X : array-like, shape (n,)
        Single pre-treatment covariate.
    alpha_oracle : array-like of length 2, optional
        If provided, Step 1 is skipped and (alpha0, alpha1) are fixed to
        these values.  Useful for diagnostic comparisons (e.g. section 4.2
        oracle experiment).

    Returns
    -------
    dict with keys:
        tau_att      : float  -- ATT point estimate
        beta         : (beta0, beta1)  -- outcome regression parameters
        phi          : float -- augmentation coefficient
        alpha        : (alpha0, alpha1) -- augmentation PS (logit) parameters
        gamma        : (gamma0, gamma1) -- balance PS (probit) parameters
        m0_hat       : array, shape (n,) -- fitted imputation values
        converged    : bool -- True if M5-M7 solver norm < 1e-4
        oracle_alpha : bool -- True when alpha was supplied via alpha_oracle
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)

    logX = np.log(np.maximum(X, 1e-300))   # guard log(0)
    sinX = np.sin(X)
    ctrl = (D == 0)   # boolean mask for control units
    w    = 1.0 - D    # same, as float weights

    # ------------------------------------------------------------------
    # Step 1: Estimate alpha via logit MLE  (or accept oracle values)
    # ------------------------------------------------------------------
    # M1 and M2 involve only (alpha0, alpha1).  They are the score equations
    # of a logit model for D on sin(X), completely decoupled from everything
    # else.  Standard MLE is the efficient solver for this sub-problem.
    # ------------------------------------------------------------------

    use_oracle_alpha = alpha_oracle is not None

    if use_oracle_alpha:
        a0_hat, a1_hat = float(alpha_oracle[0]), float(alpha_oracle[1])
    else:
        try:
            Xa = sm.add_constant(sinX)
            logit_mod = Logit(D, Xa).fit(disp=False, maxiter=200)
            a0_hat, a1_hat = logit_mod.params
        except Exception:
            a0_hat, a1_hat = 0.0, 0.0

    # Pre-compute the fixed propensity score and its odds ratio
    ea_hat = _clip_ps(_invlogit(a0_hat + a1_hat * sinX))
    odds_a = ea_hat / (1.0 - ea_hat)

    # ------------------------------------------------------------------
    # Step 2: Solve M5-M7 for (phi, gamma0, gamma1), concentrating out
    #         (beta0, beta1) analytically via OLS (M3-M4).
    # ------------------------------------------------------------------
    # For any trial (phi, g0, g1), M3=M4=0 determines (beta0, beta1)
    # as the OLS fit of [Y - phi/(1-e_b)] on [1, log(X)] among controls.
    # ------------------------------------------------------------------

    def _beta_from_ols(phi, eb_ctrl):
        """Solve M3=M4=0 analytically: OLS of adjusted Y on (1, logX) for controls."""
        adj_Y = Y[ctrl] - phi / (1.0 - eb_ctrl)
        Xc = np.column_stack([np.ones(ctrl.sum()), logX[ctrl]])
        coef, _, _, _ = np.linalg.lstsq(Xc, adj_Y, rcond=None)
        return coef[0], coef[1]

    def moment_conditions_concentrated(pgg):
        """M5, M6, M7 as a function of (phi, gamma0, gamma1) only."""
        phi, g0, g1 = pgg

        # Guard: if the probit index is extreme the augmentation term blows up.
        # Return a large residual to steer the solver away from those regions.
        idx_g = g0 + g1 * X
        if np.any(np.abs(idx_g) > 20) or np.abs(phi) > 1e4:
            return np.array([1e6, 1e6, 1e6])

        eb     = _clip_ps(norm.cdf(idx_g))
        b0, b1 = _beta_from_ols(phi, eb[ctrl])

        m0  = b0 + b1 * logX + phi / (1.0 - eb)
        Res = Y - m0

        odds_b = eb / (1.0 - eb)

        M5 = np.mean(w * odds_a * Res)
        M6 = np.mean(w * (odds_b - odds_a) * Res)
        M7 = np.mean(w * X * (odds_b - odds_a) * Res)

        return np.array([M5, M6, M7])

    # Starting values for (phi, gamma0, gamma1)
    try:
        Xb = sm.add_constant(X)
        probit_mod = Probit(D, Xb).fit(disp=False, maxiter=200)
        gamma0_init, gamma1_init = probit_mod.params
    except Exception:
        gamma0_init, gamma1_init = 0.0, 0.0

    phi_init = 0.0
    pgg0 = np.array([phi_init, gamma0_init, gamma1_init])

    _CONV_TOL = 1e-4

    best_pgg  = None
    best_norm = np.inf

    # --- Pass 1: Powell hybrid ---
    try:
        sol = root(moment_conditions_concentrated, pgg0, method='hybr',
                   options={'maxfev': 5_000, 'xtol': 1e-8})
        cand_norm = np.linalg.norm(moment_conditions_concentrated(sol.x))
        if cand_norm < best_norm:
            best_norm = cand_norm
            best_pgg  = sol.x.copy()
    except Exception:
        pass

    # --- Pass 2: Levenberg-Marquardt root-finder ---
    if best_pgg is None or best_norm >= _CONV_TOL:
        try:
            sol2 = root(moment_conditions_concentrated, pgg0, method='lm',
                        options={'maxiter': 5_000, 'col_deriv': 0})
            cand_norm = np.linalg.norm(moment_conditions_concentrated(sol2.x))
            if cand_norm < best_norm:
                best_norm = cand_norm
                best_pgg  = sol2.x.copy()
        except Exception:
            pass

    # --- Pass 3: Levenberg-Marquardt least-squares minimiser ---
    if best_pgg is None or best_norm >= _CONV_TOL:
        try:
            ls_sol = least_squares(moment_conditions_concentrated, pgg0,
                                   method='lm', max_nfev=10_000,
                                   ftol=1e-8, xtol=1e-8)
            cand_norm = np.linalg.norm(ls_sol.fun)
            if cand_norm < best_norm:
                best_norm = cand_norm
                best_pgg  = ls_sol.x.copy()
        except Exception:
            pass

    # --- Pass 4: Perturbed starting values (tiebreaker for stuck solvers) ---
    if best_pgg is None or best_norm >= _CONV_TOL:
        rng_perturb = np.random.default_rng(0)
        for _ in range(5):
            pgg_pert = pgg0 + rng_perturb.normal(0, 0.5, size=3)
            try:
                sol_p = root(moment_conditions_concentrated, pgg_pert,
                             method='hybr', options={'maxfev': 3_000, 'xtol': 1e-8})
                cand_norm = np.linalg.norm(moment_conditions_concentrated(sol_p.x))
                if cand_norm < best_norm:
                    best_norm = cand_norm
                    best_pgg  = sol_p.x.copy()
                    if best_norm < _CONV_TOL:
                        break
            except Exception:
                continue

    pgg_hat   = best_pgg if best_pgg is not None else pgg0
    converged = bool(best_norm < _CONV_TOL)

    # ------------------------------------------------------------------
    # Step 3: Recover beta from concentrated-out OLS and compute the ATT
    # ------------------------------------------------------------------

    phi_h, g0_h, g1_h = pgg_hat
    eb_hat = _clip_ps(norm.cdf(g0_h + g1_h * X))
    b0_h, b1_h = _beta_from_ols(phi_h, eb_hat[ctrl])

    m0_hat  = b0_h + b1_h * logX + phi_h / (1.0 - eb_hat)
    tau_att = float(np.mean((Y - m0_hat)[D == 1]))

    return {
        'tau_att': tau_att,
        'beta': (b0_h, b1_h),
        'phi': phi_h,
        'alpha': (a0_hat, a1_hat),
        'gamma': (g0_h, g1_h),
        'm0_hat': m0_hat,
        'converged': converged,
        'oracle_alpha': use_oracle_alpha,
    }
