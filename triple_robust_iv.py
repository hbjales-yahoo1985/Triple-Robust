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

# Minimum threshold for odds to ensure valid probabilities
MIN_ODDS = 1e-10


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
        if np.any(odds <= MIN_ODDS):
            return 1e12
        eb = odds / (1.0 + odds)
        eb = np.clip(eb, MIN_ODDS, 1.0 - MIN_ODDS)
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
    elif eb_basis == 'expX':
        eb_feature = np.exp(X)
    else:
        raise ValueError(f"Unknown eb_basis: {eb_basis}")

    gamma, eb_converged = _log_logistic_mle(D, eb_feature)

    odds_b = gamma[0] + gamma[1] * eb_feature
    odds_b = np.maximum(odds_b, MIN_ODDS)  # safety floor
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
    elif or_columns == 'X':
        exog_ctrl = np.column_stack([np.ones(ctrl.sum()), X[ctrl]])
        exog_all = np.column_stack([np.ones(n), X])
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
        'm0_hat': m0_hat,
    }


# ---------------------------------------------------------------------------
# Full estimator (Estimator F): includes the M6 moment explicitly
# ---------------------------------------------------------------------------

def triply_robust_iv_full(Y, D, X,
                          ea_basis='sin',
                          eb_basis='X',
                          or_columns='logX'):
    """
    Full triply robust ATT estimator with explicit M6 moment.

    Same as triply_robust_iv (Estimator S), but the imputation regression
    uses FOUR instruments [exog, Z, (odds_b - odds_a)] instead of three
    [exog, Z], making it an over-identified GMM/IV system that explicitly
    activates the e_b-channel moment M6.

    The key difference from S is that the instrument set includes the
    "telescoping" instrument (odds_b - odds_a) from the balance PS.

    Parameters / Returns: same as triply_robust_iv.
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)

    logX = np.log(np.maximum(X, 1e-300))
    ctrl = (D == 0)
    treated = (D == 1)
    n1 = treated.sum()

    # Step 1: Fit logit PS (e_a) by MLE
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
    Z = ea_hat / (1.0 - ea_hat)  # instrument: fitted odds of e_a

    # Step 2: Fit log-logistic PS (e_b) by MLE
    if eb_basis == 'X':
        eb_feature = X
    elif eb_basis == 'sin':
        eb_feature = np.sin(X)
    elif eb_basis == 'expX':
        eb_feature = np.exp(X)
    else:
        raise ValueError(f"Unknown eb_basis: {eb_basis}")

    gamma, eb_converged = _log_logistic_mle(D, eb_feature)

    odds_b = gamma[0] + gamma[1] * eb_feature
    odds_b = np.maximum(odds_b, MIN_ODDS)
    eb_hat = _clip_ps(odds_b / (1.0 + odds_b))
    W = 1.0 / (1.0 - eb_hat)

    # Step 3: Over-identified IV/GMM among controls
    # Instruments: {exog columns, Z, (odds_b - odds_a)}  → k_iv columns
    # Regressors:  {exog columns, W}                     → k_reg columns
    if or_columns == 'logX':
        exog_ctrl = np.column_stack([np.ones(ctrl.sum()), logX[ctrl]])
        exog_all = np.column_stack([np.ones(n), logX])
    elif or_columns == 'logX_X':
        exog_ctrl = np.column_stack([np.ones(ctrl.sum()), logX[ctrl], X[ctrl]])
        exog_all = np.column_stack([np.ones(n), logX, X])
    elif or_columns == 'X':
        exog_ctrl = np.column_stack([np.ones(ctrl.sum()), X[ctrl]])
        exog_all = np.column_stack([np.ones(n), X])
    else:
        raise ValueError(f"Unknown or_columns: {or_columns}")

    W_ctrl = W[ctrl]
    Z_ctrl = Z[ctrl]
    Y_ctrl = Y[ctrl]

    odds_a_ctrl = Z[ctrl]  # odds of e_a among controls
    odds_b_ctrl = odds_b[ctrl]
    telescope_ctrl = odds_b_ctrl - odds_a_ctrl  # the M6 instrument

    # Instrument matrix: [exog, Z, telescope]
    iv_ctrl = np.column_stack([exog_ctrl, Z_ctrl, telescope_ctrl])
    # Regressor matrix: [exog, W]
    reg_ctrl = np.column_stack([exog_ctrl, W_ctrl])

    # 2SLS-style GMM: θ = (R'P_Z R)^{-1} R'P_Z Y where P_Z = Z(Z'Z)^{-1}Z'
    ZtZ = iv_ctrl.T @ iv_ctrl
    try:
        ZtZ_inv = np.linalg.inv(ZtZ)
    except np.linalg.LinAlgError:
        ZtZ_inv = np.linalg.pinv(ZtZ)

    P_Z = iv_ctrl @ ZtZ_inv @ iv_ctrl.T  # projection matrix
    RtPR = reg_ctrl.T @ P_Z @ reg_ctrl
    RtPY = reg_ctrl.T @ P_Z @ Y_ctrl

    try:
        theta = np.linalg.solve(RtPR, RtPY)
    except np.linalg.LinAlgError:
        theta, _, _, _ = np.linalg.lstsq(RtPR, RtPY, rcond=None)

    n_exog = exog_ctrl.shape[1]
    beta = theta[:n_exog]
    phi = theta[n_exog]

    # First-stage F-statistic (same as S: Z as excluded instrument)
    n_ctrl = ctrl.sum()
    fs_X = np.column_stack([exog_ctrl, Z_ctrl])
    k_fs = fs_X.shape[1]
    fs_coef, _, _, _ = np.linalg.lstsq(fs_X, W_ctrl, rcond=None)
    W_hat_fs = fs_X @ fs_coef
    fs_coef_r, _, _, _ = np.linalg.lstsq(exog_ctrl, W_ctrl, rcond=None)
    W_hat_r = exog_ctrl @ fs_coef_r
    ssr_r = np.sum((W_ctrl - W_hat_r) ** 2)
    ssr_u = np.sum((W_ctrl - W_hat_fs) ** 2)
    first_F = max(0.0, ((ssr_r - ssr_u) / 1.0) / (ssr_u / max(n_ctrl - k_fs, 1)))

    # Step 4: Impute m0 for ALL units, compute ATT
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
        'm0_hat': m0_hat,
    }


# ---------------------------------------------------------------------------
# Estimator G: Centered exactly-identified estimator with M6
# ---------------------------------------------------------------------------

def triply_robust_iv_centered(Y, D, X,
                               ea_basis='X',
                               eb_basis='expX',
                               or_columns='X',
                               max_iter=100,
                               tol=1e-8):
    """
    Centered exactly-identified triply robust ATT estimator (Estimator G).

    Key differences from S and F:
    1. Uses log-logistic MLE for e_b, but only fixes gamma_1 from MLE;
       gamma_0 is left free and pinned by M6.
    2. Centers Y and q(X) around control-group means, drops the intercept.
    3. Uses exactly-identified IV: instruments {q_tilde, Z} for
       regressors {q_tilde, W}, enforcing M4 and M5.
    4. Solves for gamma_0 via a closed-form update from M6, iterating
       with the IV residuals.

    Parameters
    ----------
    Y, D, X : arrays (n,)
    ea_basis : str -- basis for logit PS ('X', 'sin', 'X2')
    eb_basis : str -- basis for log-logistic PS ('expX', 'X', 'sin')
    or_columns : str -- OR columns ('X', 'logX', 'logX_X')
    max_iter : int -- max iterations for gamma_0 update
    tol : float -- convergence tolerance for gamma_0

    Returns
    -------
    dict with keys: tau_att, beta1, phi, alpha, gamma, gamma0_mle,
                    gamma0_final, first_F, eb_converged, m0_hat,
                    converged, iterations, gamma0_path, denom_path
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)

    ctrl = (D == 0)
    treated = (D == 1)
    n0 = ctrl.sum()
    n1 = treated.sum()

    # ------------------------------------------------------------------
    # Step 1: Fit logit PS (e_a) by MLE -> odds Z
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
    Z = ea_hat / (1.0 - ea_hat)   # logit odds instrument

    # ------------------------------------------------------------------
    # Step 2: Fit log-logistic PS (e_b) by MLE -> fix gamma_1
    # ------------------------------------------------------------------
    if eb_basis == 'X':
        eb_feature = X
    elif eb_basis == 'sin':
        eb_feature = np.sin(X)
    elif eb_basis == 'expX':
        eb_feature = np.exp(X)
    else:
        raise ValueError(f"Unknown eb_basis: {eb_basis}")

    gamma_mle, eb_converged = _log_logistic_mle(D, eb_feature)
    gamma0_mle = gamma_mle[0]
    gamma1 = gamma_mle[1]     # fixed from MLE

    # ------------------------------------------------------------------
    # Step 3: Center Y and q(X) among controls
    # ------------------------------------------------------------------
    if or_columns == 'X':
        q = X.copy()
    elif or_columns == 'logX':
        q = np.log(np.maximum(X, 1e-300))
    elif or_columns == 'logX_X':
        q = X.copy()
    else:
        raise ValueError(f"Unknown or_columns: {or_columns}")

    Y_bar0 = np.mean(Y[ctrl])
    q_bar0 = np.mean(q[ctrl])

    Y_tilde = Y - Y_bar0
    q_tilde = q - q_bar0

    # ------------------------------------------------------------------
    # Step 4: Iterative exactly-identified IV with M6 gamma_0 update
    # ------------------------------------------------------------------
    gamma0 = gamma0_mle   # starting value
    gamma0_path = [gamma0]
    denom_path = []
    converged = False
    beta1 = 0.0
    phi = 0.0
    iteration = 0

    for iteration in range(max_iter):
        # Construct W for all observations
        odds_b = gamma0 + gamma1 * eb_feature
        odds_b_safe = np.maximum(odds_b, MIN_ODDS)
        W = 1.0 + odds_b_safe

        # IV regression among controls, NO intercept:
        #   Y_tilde = beta1 * q_tilde + phi * W
        #   Instruments: {q_tilde, Z}
        #   Regressors:  {q_tilde, W}
        q_c = q_tilde[ctrl]
        Z_c = Z[ctrl]
        W_c = W[ctrl]
        Y_c = Y_tilde[ctrl]

        iv_mat = np.column_stack([q_c, Z_c])    # n0 x 2
        reg_mat = np.column_stack([q_c, W_c])   # n0 x 2

        IvR = iv_mat.T @ reg_mat     # 2x2
        IvY = iv_mat.T @ Y_c        # 2x1

        try:
            theta = np.linalg.solve(IvR, IvY)
        except np.linalg.LinAlgError:
            theta, _, _, _ = np.linalg.lstsq(IvR, IvY, rcond=None)

        beta1 = theta[0]
        phi = theta[1]

        # Residuals among controls
        R = Y_c - beta1 * q_c - phi * W_c

        # Closed-form gamma_0 update from M6:
        #   gamma_0 = [sum Z*R - gamma1 * sum f(X)*R] / [sum R]
        sum_ZR = np.sum(Z_c * R)
        sum_fR = np.sum(eb_feature[ctrl] * R)
        sum_R = np.sum(R)

        denom_path.append(float(sum_R))

        if np.abs(sum_R) < 1e-12:
            gamma0_new = gamma0
        else:
            gamma0_new = (sum_ZR - gamma1 * sum_fR) / sum_R

        gamma0_path.append(float(gamma0_new))

        if np.abs(gamma0_new - gamma0) < tol:
            converged = True
            gamma0 = gamma0_new
            break

        gamma0 = gamma0_new

    # Final W with converged gamma0
    odds_b_final = gamma0 + gamma1 * eb_feature
    odds_b_final_safe = np.maximum(odds_b_final, MIN_ODDS)
    W_final = 1.0 + odds_b_final_safe

    # Re-run final IV with converged gamma0
    W_c_final = W_final[ctrl]
    reg_mat_final = np.column_stack([q_tilde[ctrl], W_c_final])
    iv_mat_final = np.column_stack([q_tilde[ctrl], Z[ctrl]])
    IvR_f = iv_mat_final.T @ reg_mat_final
    IvY_f = iv_mat_final.T @ Y_tilde[ctrl]
    try:
        theta_final = np.linalg.solve(IvR_f, IvY_f)
    except np.linalg.LinAlgError:
        theta_final, _, _, _ = np.linalg.lstsq(IvR_f, IvY_f, rcond=None)
    beta1 = theta_final[0]
    phi = theta_final[1]

    # ------------------------------------------------------------------
    # Step 5: Imputation and ATT
    # ------------------------------------------------------------------
    m0_hat = Y_bar0 + beta1 * q_tilde + phi * W_final
    tau_att = float(np.mean((Y - m0_hat)[treated]))

    # ------------------------------------------------------------------
    # First-stage F-statistic: Z -> W among controls (no intercept)
    # ------------------------------------------------------------------
    fs_r_coef = np.linalg.lstsq(q_tilde[ctrl].reshape(-1, 1),
                                 W_c_final, rcond=None)[0]
    W_hat_r = q_tilde[ctrl] * fs_r_coef[0]
    ssr_r = np.sum((W_c_final - W_hat_r) ** 2)

    fs_u_X = np.column_stack([q_tilde[ctrl], Z[ctrl]])
    fs_u_coef = np.linalg.lstsq(fs_u_X, W_c_final, rcond=None)[0]
    W_hat_u = fs_u_X @ fs_u_coef
    ssr_u = np.sum((W_c_final - W_hat_u) ** 2)
    first_F = max(0.0, ((ssr_r - ssr_u) / 1.0) / (ssr_u / max(n0 - 2, 1)))

    n_violations = int(np.sum(odds_b_final <= 0))

    return {
        'tau_att': tau_att,
        'beta1': beta1,
        'phi': phi,
        'alpha': alpha,
        'gamma': np.array([gamma0, gamma1]),
        'gamma0_mle': gamma0_mle,
        'gamma0_final': gamma0,
        'first_F': first_F,
        'eb_converged': eb_converged,
        'm0_hat': m0_hat,
        'converged': converged,
        'iterations': iteration + 1,
        'gamma0_path': gamma0_path,
        'denom_path': denom_path,
        'n_violations': n_violations,
    }
