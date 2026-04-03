"""
Triply Robust ATT Estimator — Log-Logistic / IV Variant
========================================================
Replaces the probit balance-PS with a *log-logistic* model whose odds are
linear in X:

    e_b(X) / (1 - e_b(X))  =  gamma_0 + gamma_1 * f(X)

so that  1 / (1 - e_b)  =  1 + gamma_0 + gamma_1 * f(X),  which is linear
in f(X).  This makes the augmentation regressor W = 1/(1 - e_b) a known
linear function of X once the log-logistic MLE is done, collapsing the
entire Step 2 into a standard 2SLS regression — no nonlinear solver, no
ridge penalty, no convergence flags.

Estimator steps:
  1.  Fit a logit PS  e_a  by MLE  →  instrument  Z = e_a/(1 - e_a)
  2.  Fit a log-logistic PS  e_b  by MLE  →  regressor  W = 1/(1 - e_b)
  3.  2SLS among controls (D = 0):
          Y  ~  {exog columns, W}   instrumented by  {exog columns, Z}
  4.  Impute  m_0(X)  for treated units, compute ATT.
"""

import numpy as np
from scipy.optimize import minimize
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Helpers (reused from triple_robust.py)
# ---------------------------------------------------------------------------

def _invlogit(z):
    """Numerically stable logistic function."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _clip_ps(p, eps=1e-3):
    """Clip propensity scores away from 0 and 1."""
    return np.clip(p, eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# Log-logistic MLE
# ---------------------------------------------------------------------------

def _log_logistic_mle(D, basis, gamma_init=None):
    """
    MLE for the log-logistic model:  odds(X) = gamma_0 + gamma_1 * basis.

    Parameters
    ----------
    D : array (n,)  — binary treatment indicator
    basis : array (n,) — the single covariate entering the odds function
    gamma_init : (gamma0, gamma1) or None

    Returns
    -------
    gamma : array (2,) — (gamma0, gamma1) MLE estimates
    converged : bool
    """
    n = len(D)
    D = np.asarray(D, dtype=float)
    basis = np.asarray(basis, dtype=float)

    # Starting values: OLS of D on {1, basis}
    if gamma_init is None:
        X_ols = np.column_stack([np.ones(n), basis])
        beta_ols, _, _, _ = np.linalg.lstsq(X_ols, D, rcond=None)
        gamma_init = beta_ols
    gamma_init = np.asarray(gamma_init, dtype=float)

    def neg_log_lik(gamma):
        g0, g1 = gamma
        odds = g0 + g1 * basis
        # Barrier: odds must be > 0 for valid probabilities
        if np.any(odds <= 1e-10):
            return 1e12
        eb = odds / (1.0 + odds)
        eb = np.clip(eb, 1e-10, 1.0 - 1e-10)
        ll = np.sum(D * np.log(eb) + (1.0 - D) * np.log(1.0 - eb))
        return -ll

    # Use L-BFGS-B; no explicit bounds needed since we penalize via barrier
    result = minimize(neg_log_lik, gamma_init, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12})

    return result.x, result.success


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def triply_robust_iv(Y, D, X,
                     ea_basis='sin',
                     eb_basis='X',
                     or_columns='logX'):
    """
    Triply robust ATT estimator using log-logistic e_b and 2SLS.

    Parameters
    ----------
    Y : array (n,) — observed outcomes
    D : array (n,) — binary treatment (1 = treated)
    X : array (n,) — single covariate
    ea_basis : str — basis for logit PS:
        'sin' → sin(X),  'X2' → X²,  'X' → X
    eb_basis : str — basis for log-logistic PS:
        'X' → X,  'sin' → sin(X)
    or_columns : str — outcome regression columns:
        'logX'  → {1, log(X)}  (misspecified)
        'logX_X' → {1, log(X), X}  (correct for the DGP)

    Returns
    -------
    dict with keys:
        tau_att    : float — ATT point estimate
        beta       : array — outcome regression coefficients
        phi        : float — coefficient on W in the structural model
        alpha      : array — logit PS parameters
        gamma      : array — log-logistic PS parameters
        first_F    : float — first-stage F-statistic for Z→W
        eb_converged : bool — whether log-logistic MLE converged
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)

    logX = np.log(np.maximum(X, 1e-300))
    ctrl = (D == 0)
    treated = (D == 1)
    n1 = treated.sum()

    # ------------------------------------------------------------------
    # Step 1: Fit logit PS (e_a) by MLE
    # ------------------------------------------------------------------
    if ea_basis == 'sin':
        ea_feature = np.sin(X)
    elif ea_basis == 'X2':
        ea_feature = X ** 2
    elif ea_basis == 'X':
        ea_feature = X
    else:
        raise ValueError(f"Unknown ea_basis: {ea_basis}")

    try:
        Xa = sm.add_constant(ea_feature)
        logit_mod = Logit(D, Xa).fit(disp=False, maxiter=200)
        alpha = logit_mod.params.copy()
    except Exception:
        alpha = np.array([0.0, 0.0])

    ea_hat = _clip_ps(_invlogit(alpha[0] + alpha[1] * ea_feature))
    Z = ea_hat / (1.0 - ea_hat)  # instrument: fitted odds

    # ------------------------------------------------------------------
    # Step 2: Fit log-logistic PS (e_b) by MLE
    # ------------------------------------------------------------------
    if eb_basis == 'X':
        eb_feature = X
    elif eb_basis == 'sin':
        eb_feature = np.sin(X)
    else:
        raise ValueError(f"Unknown eb_basis: {eb_basis}")

    gamma, eb_converged = _log_logistic_mle(D, eb_feature)

    odds_b = gamma[0] + gamma[1] * eb_feature
    odds_b = np.maximum(odds_b, 1e-10)  # safety floor
    eb_hat = _clip_ps(odds_b / (1.0 + odds_b))
    W = 1.0 / (1.0 - eb_hat)  # = 1 + odds_b

    # ------------------------------------------------------------------
    # Step 3: 2SLS among controls
    # ------------------------------------------------------------------
    # Build exogenous regressors (always includes intercept + log(X))
    if or_columns == 'logX':
        exog_ctrl = np.column_stack([np.ones(ctrl.sum()), logX[ctrl]])
        exog_all = np.column_stack([np.ones(n), logX])
    elif or_columns == 'logX_X':
        exog_ctrl = np.column_stack([np.ones(ctrl.sum()), logX[ctrl], X[ctrl]])
        exog_all = np.column_stack([np.ones(n), logX, X])
    else:
        raise ValueError(f"Unknown or_columns: {or_columns}")

    W_ctrl = W[ctrl]
    Z_ctrl = Z[ctrl]
    Y_ctrl = Y[ctrl]

    # First stage: regress W on {exog, Z} among controls
    fs_X = np.column_stack([exog_ctrl, Z_ctrl])
    fs_coef, _, _, _ = np.linalg.lstsq(fs_X, W_ctrl, rcond=None)
    W_hat_ctrl = fs_X @ fs_coef

    # First-stage F-statistic (on the excluded instrument Z)
    n_ctrl = ctrl.sum()
    k_fs = fs_X.shape[1]
    # F = (SSR_restricted - SSR_unrestricted) / SSR_unrestricted * (n-k)/1
    # Restricted model: W ~ exog only (no Z)
    fs_coef_r, _, _, _ = np.linalg.lstsq(exog_ctrl, W_ctrl, rcond=None)
    W_hat_r = exog_ctrl @ fs_coef_r
    ssr_r = np.sum((W_ctrl - W_hat_r) ** 2)
    ssr_u = np.sum((W_ctrl - W_hat_ctrl) ** 2)
    first_F = max(0.0, ((ssr_r - ssr_u) / 1.0) / (ssr_u / max(n_ctrl - k_fs, 1)))

    # Second stage: regress Y on {exog, W_hat} among controls
    ss_X = np.column_stack([exog_ctrl, W_hat_ctrl])
    ss_coef, _, _, _ = np.linalg.lstsq(ss_X, Y_ctrl, rcond=None)

    # ss_coef = [beta_0, beta_1, (beta_2 if logX_X), phi]
    n_exog = exog_ctrl.shape[1]
    beta = ss_coef[:n_exog]
    phi = ss_coef[n_exog]

    # ------------------------------------------------------------------
    # Step 4: Impute m0 for ALL units, compute ATT
    # ------------------------------------------------------------------
    m0_hat = exog_all @ beta + phi * W
    tau_att = float(np.mean((Y - m0_hat)[treated]))

    return {
        'tau_att': tau_att,
        'beta': beta,
        'phi': phi,
        'alpha': alpha,
        'gamma': gamma,
        'first_F': first_F,
        'eb_converged': eb_converged,
    }
