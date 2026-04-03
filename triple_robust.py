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

  Step 2 - With alpha fixed at alpha-hat, minimise the L2-penalised GMM
            objective for (phi, gamma0, gamma1):

              Q_pen(φ,γ) = M5² + M6² + M7²
                         + λ·[(φ−φ₀)² + (γ₀−γ₀⁰)² + (γ₁−γ₁⁰)²]

            where (φ₀, γ₀⁰, γ₁⁰) are the MLE starting values (probit MLE
            for γ, 0 for φ) and β is concentrated out analytically (OLS).

            Viewing the unpenalised M5=M6=M7=0 system as an exactly-identified
            GMM with identity weight matrix, the unpenalised objective is
            M5²+M6²+M7².  Adding the ridge term keeps parameters close to
            their MLE anchors, preventing the solver from drifting into
            degenerate regions (e.g. φ→−∞ causing the OLS augmentation to
            absorb all variance).

            At fixed λ the estimator is slightly biased (O(λ)); for asymptotic
            consistency λ should shrink to 0 as n→∞.  The default λ=0.01 is
            a proof-of-concept value suited for n=2000.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
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

def triply_robust_att(Y, D, X, alpha_oracle=None, l2_penalty=0.01):
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
    l2_penalty : float, optional (default 0.01)
        Ridge penalty coefficient λ.  The solver minimises:

            Q_pen(φ,γ) = M5² + M6² + M7²
                       + λ·[(φ-φ₀)² + (γ₀-γ₀⁰)² + (γ₁-γ₁⁰)²]

        where (φ₀, γ₀⁰, γ₁⁰) are the MLE starting values (probit MLE for
        γ, 0 for φ).  At fixed λ the estimator has an O(λ) bias; for
        asymptotic consistency λ should shrink to 0 as n grows.

    Returns
    -------
    dict with keys:
        tau_att      : float  -- ATT point estimate
        beta         : (beta0, beta1)  -- outcome regression parameters
        phi          : float -- augmentation coefficient
        alpha        : (alpha0, alpha1) -- augmentation PS (logit) parameters
        gamma        : (gamma0, gamma1) -- balance PS (probit) parameters
        m0_hat       : array, shape (n,) -- fitted imputation values
        converged    : bool -- True when the LM solver reached its stopping
                              criterion (gradient/step norm small)
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

    # ------------------------------------------------------------------
    # Penalised residual vector (6-element, overdetermined 6×3 system)
    # ------------------------------------------------------------------
    # Minimising ½·‖r(pgg)‖² is equivalent to:
    #   Q_pen = M5² + M6² + M7² + λ·[(φ−φ₀)² + (γ₀−γ₀⁰)² + (γ₁−γ₁⁰)²]
    # The three penalty rows act as a soft ridge prior anchored at the MLE
    # starting values, preventing the solver from reaching degenerate
    # regions where large |φ| lets the OLS soak up all variance.
    # ------------------------------------------------------------------
    sq_lam = np.sqrt(l2_penalty)

    def penalized_residuals(pgg):
        moments = moment_conditions_concentrated(pgg)
        return np.concatenate([moments, sq_lam * (pgg - pgg0)])

    sol = least_squares(penalized_residuals, pgg0,
                        method='lm', max_nfev=10_000,
                        ftol=1e-10, xtol=1e-10)

    pgg_hat   = sol.x
    # "converged" = the LM solver reached its own stopping criterion
    # (gradient norm small, or step size small).  LM status > 0 signals this.
    converged = bool(sol.status > 0)

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
