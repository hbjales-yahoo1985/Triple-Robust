"""
Stacked IV Multiply Robust Estimator — Simulation Study
========================================================
Compares two multiply robust estimators for E[Y] / ATT:

1. Chan (2013) OLS stacking:
     Regress Y on {1, 1/π̂_1,...,1/π̂_K, â_1,...,â_K} among complete cases.
     Average predictions over all n.  Fits 2K+1 parameters.

2. Proposed IV stacking:
     Among complete cases, IV of Y on {m̂_1,...,m̂_K} with instruments
     {o_1,...,o_K} where o_k = ê_k/(1−ê_k).  Fits K parameters.

Both achieve 2K-fold robustness: consistency if any one of K PS models
or K OR models is correctly specified.

Hypothesis: IV has lower RMSE than Chan when K is moderate and models
are correlated, because it fits fewer parameters.

Run:
    python stacked_iv_mr.py              # R=10 smoke test
    python stacked_iv_mr.py 1000         # R=1000 full run
"""

import sys
import warnings
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

PS_CLIP = (0.01, 0.99)


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _clip_ps(p, lo=PS_CLIP[0], hi=PS_CLIP[1]):
    return np.clip(p, lo, hi)


def _invlogit(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2: PS and OR model fitting
# ═══════════════════════════════════════════════════════════════════════════

def fit_ps_model(D, X, model_type, covariate_func):
    """
    Fit a propensity score model by MLE.

    Parameters
    ----------
    D : (n,) binary indicator (1=treated/observed)
    X : (n,) covariate
    model_type : 'logit', 'probit', or 'cloglog'
    covariate_func : callable X -> design matrix (n, p) without constant

    Returns
    -------
    ps_fitted : (n,) fitted probabilities (clipped)
    """
    n = len(D)
    features = covariate_func(X)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xd = sm.add_constant(features)

    if model_type == 'logit':
        try:
            mod = Logit(D, Xd).fit(disp=False, maxiter=200)
            ps = _clip_ps(mod.predict(Xd))
        except Exception:
            ps = np.full(n, np.clip(np.mean(D), *PS_CLIP))
    elif model_type == 'probit':
        try:
            mod = Probit(D, Xd).fit(disp=False, maxiter=200)
            ps = _clip_ps(mod.predict(Xd))
        except Exception:
            ps = np.full(n, np.clip(np.mean(D), *PS_CLIP))
    elif model_type == 'cloglog':
        ps = _fit_cloglog(D, Xd)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return ps


def _fit_cloglog(D, Xd):
    """Complementary log-log by MLE via scipy."""
    n, p = Xd.shape
    D = np.asarray(D, float)

    def neg_ll(beta):
        eta = Xd @ beta
        # cloglog link: P = 1 - exp(-exp(eta))
        exp_eta = np.clip(eta, -30, 30)
        p_hat = 1.0 - np.exp(-np.exp(exp_eta))
        p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
        ll = np.sum(D * np.log(p_hat) + (1 - D) * np.log(1 - p_hat))
        return -ll

    beta0 = np.zeros(p)
    try:
        res = minimize(neg_ll, beta0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10})
        eta = Xd @ res.x
        ps = 1.0 - np.exp(-np.exp(np.clip(eta, -30, 30)))
    except Exception:
        ps = np.full(n, np.clip(np.mean(D), *PS_CLIP))
    return _clip_ps(ps)


def fit_or_model(Y, X, is_complete, covariate_func):
    """
    Fit outcome regression by OLS among is_complete==1.

    Parameters
    ----------
    Y : (n,) outcome vector
    X : (n,) covariate
    is_complete : (n,) binary indicator (1 = use in fitting).
        For E[Y]: R (missingness indicator).
        For ATT: 1-D (control indicator).
    covariate_func : callable

    Returns
    -------
    or_fitted : (n,) fitted values for ALL observations
    """
    n = len(Y)
    features = covariate_func(X)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xd_all = sm.add_constant(features)

    complete = (is_complete == 1)
    Xd_c = Xd_all[complete]
    Y_c = Y[complete]

    try:
        coef, _, _, _ = np.linalg.lstsq(Xd_c, Y_c, rcond=None)
        fitted = Xd_all @ coef
    except Exception:
        fitted = np.full(n, np.nanmean(Y[complete]))
    return fitted


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3: Core estimators
# ═══════════════════════════════════════════════════════════════════════════

def chan_ols_estimator(Y, R, ps_fitted_list, or_fitted_list):
    """
    Chan (2013) OLS stacking estimator for E[Y].

    Regress Y on {1, 1/π̂_1,...,1/π̂_J, â_1,...,â_K} among R==1.
    Average predictions over all n.

    Returns: (mu_hat, cond_num)
    """
    n = len(Y)
    complete = (R == 1)

    J = len(ps_fitted_list)
    K = len(or_fitted_list)

    # Build regressors: [1, 1/π̂_1,...,1/π̂_J, â_1,...,â_K]
    cols_all = [np.ones(n)]
    for ps in ps_fitted_list:
        cols_all.append(1.0 / _clip_ps(ps))
    for orr in or_fitted_list:
        cols_all.append(orr)
    U_all = np.column_stack(cols_all)

    U_c = U_all[complete]
    Y_c = Y[complete]

    cond_num = float(np.linalg.cond(U_c.T @ U_c))

    try:
        coef, _, _, _ = np.linalg.lstsq(U_c, Y_c, rcond=None)
    except Exception:
        return np.nan, cond_num

    mu_hat = float(np.mean(U_all @ coef))
    return mu_hat, cond_num


def iv_stacking_estimator(Y, R, ps_fitted_list, or_fitted_list):
    """
    Proposed IV stacking estimator for E[Y].

    Among R==1: IV of Y on {m̂_1,...,m̂_K} with instruments {o_1,...,o_K}
    where o_k = ê_k/(1−ê_k).
    No intercept.

    Returns: (mu_hat, cond_num, first_stage_F)
    """
    n = len(Y)
    K = len(ps_fitted_list)
    complete = (R == 1)
    m_c = int(complete.sum())

    # Build matrices among complete cases
    M_all = np.column_stack(or_fitted_list)          # (n, K)
    O_all = np.column_stack([_clip_ps(ps)/(1.0 - _clip_ps(ps))
                             for ps in ps_fitted_list])  # (n, K) odds

    M_c = M_all[complete]
    O_c = O_all[complete]
    Y_c = Y[complete]

    OtM = O_c.T @ M_c
    OtY = O_c.T @ Y_c

    cond_num = float(np.linalg.cond(OtM))

    try:
        phi = np.linalg.solve(OtM, OtY)
    except np.linalg.LinAlgError:
        phi, _, _, _ = np.linalg.lstsq(OtM, OtY, rcond=None)

    mu_hat = float(np.mean(M_all @ phi))

    # First-stage F: regress each m_k on {o_1,...,o_K}
    # Use average F across the K equations
    f_stats = []
    for k in range(K):
        m_k = M_c[:, k]
        coef_k, _, _, _ = np.linalg.lstsq(O_c, m_k, rcond=None)
        resid = m_k - O_c @ coef_k
        ssr = np.sum(resid**2)
        sst = np.sum((m_k - np.mean(m_k))**2)
        if ssr > 0 and sst > 0:
            r2 = 1 - ssr / sst
            f_val = max(0, (r2 / max(K, 1)) / ((1 - r2) / max(m_c - K, 1)))
            f_stats.append(f_val)
    first_stage_F = float(np.mean(f_stats)) if f_stats else 0.0

    return mu_hat, cond_num, first_stage_F


def single_dr_estimator(Y, R, ps_fitted, or_fitted):
    """
    Standard Bang-Robins doubly robust estimator for E[Y].

    μ̂_DR = (1/n) Σ [ â(X_i) + R_i(Y_i - â(X_i))/π̂(X_i) ]
    """
    n = len(Y)
    ps_c = _clip_ps(ps_fitted)
    dr = or_fitted + R * (Y - or_fitted) / ps_c
    return float(np.mean(dr))


def ipw_estimator(Y, R, ps_fitted):
    """Horvitz-Thompson IPW estimator: μ̂ = (1/n) Σ R_i Y_i / π̂(X_i)."""
    n = len(Y)
    ps_c = _clip_ps(ps_fitted)
    return float(np.mean(R * Y / ps_c))


def or_average_estimator(Y, R, or_fitted_list):
    """OLS of Y on {â_1,...,â_K} among R==1, average predictions."""
    n = len(Y)
    complete = (R == 1)
    K = len(or_fitted_list)
    M_all = np.column_stack(or_fitted_list)
    M_c = M_all[complete]
    Y_c = Y[complete]
    try:
        coef, _, _, _ = np.linalg.lstsq(M_c, Y_c, rcond=None)
    except Exception:
        return np.nan
    return float(np.mean(M_all @ coef))


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3b: ATT versions
# ═══════════════════════════════════════════════════════════════════════════

def chan_ols_att(Y, D, ps_fitted_list, or_fitted_list):
    """
    Chan OLS stacking for ATT.

    Among controls (D==0): regress Y on {1, 1/(1-ê_1),..., m̂_1,...,m̂_K}.
    Impute m̂₀(X_i) for treated. ATT = n₁⁻¹ Σ D_i(Y_i - m̂₀(X_i)).
    """
    n = len(Y)
    ctrl = (D == 0)
    treated = (D == 1)
    n1 = int(treated.sum())
    if n1 == 0:
        return np.nan, np.nan

    K = len(ps_fitted_list)

    cols_all = [np.ones(n)]
    for ps in ps_fitted_list:
        cols_all.append(1.0 / (1.0 - _clip_ps(ps)))
    for orr in or_fitted_list:
        cols_all.append(orr)
    U_all = np.column_stack(cols_all)

    U_ctrl = U_all[ctrl]
    Y_ctrl = Y[ctrl]

    cond_num = float(np.linalg.cond(U_ctrl.T @ U_ctrl))

    try:
        coef, _, _, _ = np.linalg.lstsq(U_ctrl, Y_ctrl, rcond=None)
    except Exception:
        return np.nan, cond_num

    m0_hat = U_all @ coef
    tau_att = float(np.mean((Y - m0_hat)[treated]))
    return tau_att, cond_num


def iv_stacking_att(Y, D, ps_fitted_list, or_fitted_list):
    """
    IV stacking for ATT.

    Among controls: IV of Y on {m̂_1,...,m̂_K} with instruments
    {o_1,...,o_K} where o_k = ê_k/(1−ê_k). No intercept.
    Impute. ATT.
    """
    n = len(Y)
    ctrl = (D == 0)
    treated = (D == 1)
    n1 = int(treated.sum())
    if n1 == 0:
        return np.nan, np.nan, np.nan

    K = len(ps_fitted_list)

    M_all = np.column_stack(or_fitted_list)
    O_all = np.column_stack([_clip_ps(ps)/(1.0 - _clip_ps(ps))
                             for ps in ps_fitted_list])

    M_ctrl = M_all[ctrl]
    O_ctrl = O_all[ctrl]
    Y_ctrl = Y[ctrl]

    OtM = O_ctrl.T @ M_ctrl
    OtY = O_ctrl.T @ Y_ctrl
    cond_num = float(np.linalg.cond(OtM))

    try:
        phi = np.linalg.solve(OtM, OtY)
    except np.linalg.LinAlgError:
        phi, _, _, _ = np.linalg.lstsq(OtM, OtY, rcond=None)

    m0_hat = M_all @ phi
    tau_att = float(np.mean((Y - m0_hat)[treated]))
    return tau_att, cond_num, 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4: DGP
# ═══════════════════════════════════════════════════════════════════════════

def dgp_base(rng, n, true_ps_type, true_or_type, tau=2.0, att_mode=False):
    """
    Generate data with specified PS and OR types.

    Returns: Y_obs, R (or D), X, mu_true, e_true, true_mean
    """
    X = rng.uniform(-2, 2, n)

    # True propensity score
    if true_ps_type == 'logistic_quad':
        logit_e = -0.5 + 0.3*X - 0.2*X**2
        e_true = _clip_ps(_invlogit(logit_e))
    elif true_ps_type == 'cloglog_exp':
        eta = 0.2 + 0.4*X - 0.3*np.exp(X/2)
        e_true = _clip_ps(1.0 - np.exp(-np.exp(np.clip(eta, -30, 30))))
    else:
        raise ValueError(f"Unknown true_ps_type: {true_ps_type}")

    R = rng.binomial(1, e_true, n).astype(float)

    # True outcome regression
    if true_or_type == 'quadratic':
        mu = 1.0 + 2.0*X + 1.5*X**2
    elif true_or_type == 'trig':
        mu = 1.0 + 2.0*X + 1.5*np.sin(2*X)
    elif true_or_type == 'exp':
        mu = 1.0 + 0.5*np.exp(X)
    else:
        raise ValueError(f"Unknown true_or_type: {true_or_type}")

    if att_mode:
        # ATT mode: R → D, Y observed for all, treatment effect = tau
        D = R.copy()
        Y0 = mu + rng.normal(0, 1, n)
        Y1 = Y0 + tau
        Y = D * Y1 + (1.0 - D) * Y0
        true_att = tau
        return Y, D, X, mu, e_true, true_att
    else:
        # Missing data mode: Y observed only if R==1
        Y_full = mu + rng.normal(0, 1, n)
        true_mean = float(np.mean(mu))  # E[Y] = E[μ(X)]
        Y_obs = np.where(R == 1, Y_full, np.nan)
        return Y_obs, R, X, mu, e_true, true_mean


# ═══════════════════════════════════════════════════════════════════════════
#  Section 5: Candidate working models
# ═══════════════════════════════════════════════════════════════════════════

# PS covariate functions
def _ps_cov_quad(X):
    """PS1: {X, X²} for logit — correct under logistic_quad."""
    return np.column_stack([X, X**2])

def _ps_cov_cloglog_exp(X):
    """PS2: {X, exp(X/2)} for cloglog — correct under cloglog_exp."""
    return np.column_stack([X, np.exp(X/2)])

def _ps_cov_sin(X):
    """PS3: {sin(X)} for logit — always wrong."""
    return np.sin(X).reshape(-1, 1)

def _ps_cov_linear(X):
    """PS4: {X} for probit — always wrong (wrong link or covs)."""
    return X.reshape(-1, 1)

# OR covariate functions
def _or_cov_quad(X):
    """OR1: {X, X²} — correct under quadratic."""
    return np.column_stack([X, X**2])

def _or_cov_trig(X):
    """OR2: {X, sin(2X)} — correct under trig."""
    return np.column_stack([X, np.sin(2*X)])

def _or_cov_exp(X):
    """OR3: {exp(X)} — correct under exp."""
    return np.exp(X).reshape(-1, 1)

def _or_cov_linear(X):
    """OR4: {X} — always wrong (too simple)."""
    return X.reshape(-1, 1)

def _or_cov_cubic(X):
    """OR5 (extra): {X, X³} — always wrong."""
    return np.column_stack([X, X**3])

def _or_cov_log(X):
    """OR6 (extra): {log(|X|+1)} — always wrong."""
    return np.log(np.abs(X) + 1).reshape(-1, 1)

# Extra PS models for K>4
def _ps_cov_x_sin(X):
    """PS5: {X, sin(X)} for logit — wrong."""
    return np.column_stack([X, np.sin(X)])

def _ps_cov_abs(X):
    """PS6: {|X|} for logit — wrong."""
    return np.abs(X).reshape(-1, 1)


# Model catalogs
PS_MODELS = {
    'PS1': ('logit',  _ps_cov_quad,       'logit on {X, X²}'),
    'PS2': ('cloglog', _ps_cov_cloglog_exp, 'cloglog on {X, exp(X/2)}'),
    'PS3': ('logit',  _ps_cov_sin,        'logit on {sin(X)}'),
    'PS4': ('probit', _ps_cov_linear,     'probit on {X}'),
    'PS5': ('logit',  _ps_cov_x_sin,      'logit on {X, sin(X)}'),
    'PS6': ('logit',  _ps_cov_abs,        'logit on {|X|}'),
}

OR_MODELS = {
    'OR1': (_or_cov_quad,    'OLS on {1, X, X²}'),
    'OR2': (_or_cov_trig,    'OLS on {1, X, sin(2X)}'),
    'OR3': (_or_cov_exp,     'OLS on {1, exp(X)}'),
    'OR4': (_or_cov_linear,  'OLS on {1, X}'),
    'OR5': (_or_cov_cubic,   'OLS on {1, X, X³}'),
    'OR6': (_or_cov_log,     'OLS on {1, log(|X|+1)}'),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Section 6: Scenario definitions
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'A': {
        'desc': 'K=2, one PS correct, all OR wrong',
        'dgp': ('logistic_quad', 'trig'),
        'ps_keys': ['PS1', 'PS3'],   # PS1 correct
        'or_keys': ['OR3', 'OR4'],   # both wrong
    },
    'B': {
        'desc': 'K=2, one OR correct, all PS wrong',
        'dgp': ('cloglog_exp', 'quadratic'),
        'ps_keys': ['PS3', 'PS4'],   # both wrong (PS1 wrong for cloglog)
        'or_keys': ['OR1', 'OR4'],   # OR1 correct
    },
    'C': {
        'desc': 'K=2, all wrong (negative control)',
        'dgp': ('cloglog_exp', 'trig'),
        'ps_keys': ['PS3', 'PS4'],   # both wrong
        'or_keys': ['OR3', 'OR4'],   # both wrong
    },
    'D': {
        'desc': 'K=4, one PS correct',
        'dgp': ('logistic_quad', 'trig'),
        'ps_keys': ['PS1', 'PS3', 'PS4', 'PS5'],  # PS1 correct
        'or_keys': ['OR3', 'OR4', 'OR5', 'OR6'],  # all wrong
    },
    'E': {
        'desc': 'K=4, one OR correct',
        'dgp': ('cloglog_exp', 'quadratic'),
        'ps_keys': ['PS3', 'PS4', 'PS5', 'PS6'],  # all wrong
        'or_keys': ['OR1', 'OR3', 'OR4', 'OR5'],  # OR1 correct
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  Section 7: Simulation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(scenario_key, n, R_reps, seed=42):
    """
    Run a complete scenario.

    Returns dict of {estimator_name: list of estimates}
    and dict of {estimator_name: list of condition numbers}
    """
    sc = SCENARIOS[scenario_key]
    ps_type, or_type = sc['dgp']
    ps_keys = sc['ps_keys']
    or_keys = sc['or_keys']
    K = len(ps_keys)

    results = {
        'IV': [], 'Chan': [],
        'OR_avg': [], 'Oracle': [],
        'IV_cond': [], 'Chan_cond': [],
        'IV_F': [],
    }
    # DR and IPW results
    for j, pk in enumerate(ps_keys):
        results[f'IPW_{pk}'] = []
        for k, ok in enumerate(or_keys):
            results[f'DR_{pk}_{ok}'] = []

    rng_base = np.random.default_rng(seed)
    rep_seeds = rng_base.integers(0, 2**31, size=R_reps)

    for rep in range(R_reps):
        rng_rep = np.random.default_rng(rep_seeds[rep])
        Y_obs, R, X, mu_true, e_true, true_mean = dgp_base(
            rng_rep, n, ps_type, or_type)

        # Fit all PS models
        ps_fitted = {}
        for pk in ps_keys:
            mtype, cfunc, _ = PS_MODELS[pk]
            ps_fitted[pk] = fit_ps_model(R, X, mtype, cfunc)

        # Fit all OR models
        or_fitted = {}
        for ok in or_keys:
            cfunc, _ = OR_MODELS[ok]
            # Need Y without NaN for OLS — use Y_obs
            Y_for_or = np.where(np.isnan(Y_obs), 0.0, Y_obs)
            or_fitted[ok] = fit_or_model(Y_for_or, X, R, cfunc)

        ps_list = [ps_fitted[pk] for pk in ps_keys]
        or_list = [or_fitted[ok] for ok in or_keys]

        # Replace NaN Y with 0 for computations (R==0 entries don't matter)
        Y_safe = np.where(np.isnan(Y_obs), 0.0, Y_obs)

        # IV stacking
        try:
            mu_iv, cond_iv, f_iv = iv_stacking_estimator(
                Y_safe, R, ps_list, or_list)
            results['IV'].append(mu_iv)
            results['IV_cond'].append(cond_iv)
            results['IV_F'].append(f_iv)
        except Exception:
            results['IV'].append(np.nan)
            results['IV_cond'].append(np.nan)
            results['IV_F'].append(np.nan)

        # Chan OLS
        try:
            mu_chan, cond_chan = chan_ols_estimator(
                Y_safe, R, ps_list, or_list)
            results['Chan'].append(mu_chan)
            results['Chan_cond'].append(cond_chan)
        except Exception:
            results['Chan'].append(np.nan)
            results['Chan_cond'].append(np.nan)

        # OR average
        try:
            mu_or = or_average_estimator(Y_safe, R, or_list)
            results['OR_avg'].append(mu_or)
        except Exception:
            results['OR_avg'].append(np.nan)

        # Oracle
        results['Oracle'].append(true_mean)

        # IPW with each PS
        for pk in ps_keys:
            try:
                mu_ipw = ipw_estimator(Y_safe, R, ps_fitted[pk])
                results[f'IPW_{pk}'].append(mu_ipw)
            except Exception:
                results[f'IPW_{pk}'].append(np.nan)

        # DR for each (PS, OR) pair
        for pk in ps_keys:
            for ok in or_keys:
                try:
                    mu_dr = single_dr_estimator(
                        Y_safe, R, ps_fitted[pk], or_fitted[ok])
                    results[f'DR_{pk}_{ok}'].append(mu_dr)
                except Exception:
                    results[f'DR_{pk}_{ok}'].append(np.nan)

    return results


def _stats(arr, true_val):
    """Compute bias, SD, RMSE from array of estimates."""
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mn = float(np.mean(a))
    bias = mn - true_val
    sd = float(np.std(a))
    rmse = float(np.sqrt(bias**2 + sd**2))
    return mn, bias, sd, rmse


# ═══════════════════════════════════════════════════════════════════════════
#  Section 8: Main tables and figures
# ═══════════════════════════════════════════════════════════════════════════

def print_table1(scenario_keys, sample_sizes, R_reps, seed=42):
    """Table 1/2: Main results."""
    for sc_key in scenario_keys:
        sc = SCENARIOS[sc_key]
        ps_type, or_type = sc['dgp']
        K = len(sc['ps_keys'])

        # Compute true E[Y] from a large sample
        rng_large = np.random.default_rng(seed + 999)
        X_large = rng_large.uniform(-2, 2, 500_000)
        if or_type == 'quadratic':
            mu_large = 1.0 + 2.0*X_large + 1.5*X_large**2
        elif or_type == 'trig':
            mu_large = 1.0 + 2.0*X_large + 1.5*np.sin(2*X_large)
        elif or_type == 'exp':
            mu_large = 1.0 + 0.5*np.exp(X_large)
        true_EY = float(np.mean(mu_large))

        print(f"\n{'='*90}")
        print(f"Scenario {sc_key}: {sc['desc']}  (K={K})")
        print(f"DGP: PS={ps_type}, OR={or_type}")
        print(f"PS models: {sc['ps_keys']}, OR models: {sc['or_keys']}")
        print(f"True E[Y] = {true_EY:.4f}")
        print(f"{'='*90}")

        header = f"{'Estimator':>20}  {'n':>6}  {'Mean':>8}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}"
        sep = "-" * len(header)

        print(header)
        print(sep)

        for n_val in sample_sizes:
            res = run_scenario(sc_key, n_val, R_reps, seed)

            # Core estimators
            rows = []
            for est_name in ['IV', 'Chan', 'OR_avg']:
                mn, bias, sd, rmse = _stats(res[est_name], true_EY)
                rows.append((est_name, mn, bias, sd, rmse))

            # Best and worst DR
            dr_keys = [k for k in res if k.startswith('DR_')]
            if dr_keys:
                dr_biases = {}
                for dk in dr_keys:
                    mn, bias, sd, rmse = _stats(res[dk], true_EY)
                    dr_biases[dk] = (abs(bias) if not np.isnan(bias) else 1e10,
                                     mn, bias, sd, rmse, dk)
                sorted_dr = sorted(dr_biases.values())
                best = sorted_dr[0]
                worst = sorted_dr[-1]
                rows.append((f'DR_best ({best[5][3:]})', best[1], best[2], best[3], best[4]))
                rows.append((f'DR_worst ({worst[5][3:]})', worst[1], worst[2], worst[3], worst[4]))

            # Best IPW
            ipw_keys = [k for k in res if k.startswith('IPW_')]
            if ipw_keys:
                ipw_biases = {}
                for ik in ipw_keys:
                    mn, bias, sd, rmse = _stats(res[ik], true_EY)
                    ipw_biases[ik] = (abs(bias) if not np.isnan(bias) else 1e10,
                                      mn, bias, sd, rmse, ik)
                best_ipw = sorted(ipw_biases.values())[0]
                rows.append((f'IPW_best ({best_ipw[5][4:]})',
                             best_ipw[1], best_ipw[2], best_ipw[3], best_ipw[4]))

            for name, mn, bias, sd, rmse in rows:
                if np.isnan(mn):
                    print(f"{name:>20}  {n_val:>6}  {'NaN':>8}  {'NaN':>8}  {'NaN':>7}  {'NaN':>7}")
                else:
                    print(f"{name:>20}  {n_val:>6}  {mn:>8.4f}  {bias:>+8.4f}  {sd:>7.4f}  {rmse:>7.4f}")

            # Condition numbers
            iv_conds = np.array(res['IV_cond'], dtype=float)
            iv_conds = iv_conds[~np.isnan(iv_conds)]
            chan_conds = np.array(res['Chan_cond'], dtype=float)
            chan_conds = chan_conds[~np.isnan(chan_conds)]
            if len(iv_conds) > 0 and len(chan_conds) > 0:
                print(f"{'':>20}  {'':>6}  Cond(IV)={np.median(iv_conds):.1f}  "
                      f"Cond(Chan)={np.median(chan_conds):.1f}")

            print(sep)


def print_table3_scaling(R_reps, n=1000, seed=42):
    """Table 3: Scaling test — vary K from 2 to 6."""
    ps_type, or_type = 'logistic_quad', 'trig'

    # PS1 is correct; add more wrong models as K grows
    ps_key_sets = {
        2: ['PS1', 'PS3'],
        3: ['PS1', 'PS3', 'PS4'],
        4: ['PS1', 'PS3', 'PS4', 'PS5'],
        5: ['PS1', 'PS3', 'PS4', 'PS5', 'PS6'],
        6: ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6'],
    }
    or_key_sets = {
        2: ['OR3', 'OR4'],
        3: ['OR3', 'OR4', 'OR5'],
        4: ['OR3', 'OR4', 'OR5', 'OR6'],
        5: ['OR3', 'OR4', 'OR5', 'OR6', 'OR2'],
        6: ['OR1', 'OR2', 'OR3', 'OR4', 'OR5', 'OR6'],
    }

    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 1.0 + 2.0*X_large + 1.5*np.sin(2*X_large)
    true_EY = float(np.mean(mu_large))

    print(f"\n{'='*90}")
    print(f"Table 3: Scaling (K=2..6), n={n}, R={R_reps}")
    print(f"Scenario A extended: PS1 correct, OR all wrong")
    print(f"True E[Y] = {true_EY:.4f}")
    print(f"{'='*90}")

    header = (f"{'K':>3}  {'Est':>6}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}  "
              f"{'Cond_med':>10}  {'Params':>6}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    iv_rmses = []
    chan_rmses = []
    ks = []

    for K in [2, 3, 4, 5, 6]:
        # Build a temporary scenario
        SCENARIOS['_temp'] = {
            'desc': f'Scaling K={K}',
            'dgp': (ps_type, or_type),
            'ps_keys': ps_key_sets[K],
            'or_keys': or_key_sets[K],
        }
        res = run_scenario('_temp', n, R_reps, seed)
        del SCENARIOS['_temp']

        for est in ['IV', 'Chan']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            conds = np.array(res.get(f'{est}_cond', []), dtype=float)
            conds = conds[~np.isnan(conds)]
            cond_med = float(np.median(conds)) if len(conds) > 0 else np.nan
            n_params = K if est == 'IV' else 2*K+1
            if not np.isnan(mn):
                print(f"{K:>3}  {est:>6}  {bias:>+8.4f}  {sd:>7.4f}  "
                      f"{rmse:>7.4f}  {cond_med:>10.1f}  {n_params:>6}")
            else:
                print(f"{K:>3}  {est:>6}  {'NaN':>8}  {'NaN':>7}  "
                      f"{'NaN':>7}  {'NaN':>10}  {n_params:>6}")

            if est == 'IV' and not np.isnan(rmse):
                iv_rmses.append(rmse)
            elif est == 'Chan' and not np.isnan(rmse):
                chan_rmses.append(rmse)
        ks.append(K)

        print(sep)

    return ks, iv_rmses, chan_rmses


def run_robustness_all_correct(R_reps, sample_sizes, seed=42):
    """Robustness 5.1: All models correct."""
    SCENARIOS['_allcorr'] = {
        'desc': 'K=2, both PS1 and OR1 correct',
        'dgp': ('logistic_quad', 'quadratic'),
        'ps_keys': ['PS1', 'PS3'],  # PS1 correct
        'or_keys': ['OR1', 'OR4'],  # OR1 correct
    }

    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 1.0 + 2.0*X_large + 1.5*X_large**2
    true_EY = float(np.mean(mu_large))

    print(f"\n{'='*90}")
    print(f"Robustness 5.1: All correct (PS1+OR1 both correct)")
    print(f"True E[Y] = {true_EY:.4f}")
    print(f"{'='*90}")

    header = f"{'Estimator':>20}  {'n':>6}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_val in sample_sizes:
        res = run_scenario('_allcorr', n_val, R_reps, seed)
        for est in ['IV', 'Chan', 'OR_avg']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            if not np.isnan(mn):
                print(f"{est:>20}  {n_val:>6}  {bias:>+8.4f}  {sd:>7.4f}  {rmse:>7.4f}")
        # Best DR
        dr_keys = [k for k in res if k.startswith('DR_')]
        best_rmse = 1e10
        best_name = ''
        for dk in dr_keys:
            mn, bias, sd, rmse = _stats(res[dk], true_EY)
            if not np.isnan(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_name = dk
                best_bias, best_sd = bias, sd
        if best_name:
            print(f"{'DR_best':>20}  {n_val:>6}  {best_bias:>+8.4f}  "
                  f"{best_sd:>7.4f}  {best_rmse:>7.4f}")
        print(sep)

    del SCENARIOS['_allcorr']


def run_att_robustness(R_reps, sample_sizes, seed=42):
    """Robustness 5.4: ATT version of Scenario A."""
    tau = 2.0
    ps_type, or_type = 'logistic_quad', 'trig'
    ps_keys = ['PS1', 'PS3']
    or_keys = ['OR3', 'OR4']

    print(f"\n{'='*90}")
    print(f"Robustness 5.4: ATT version (Scenario A)")
    print(f"True ATT = {tau}")
    print(f"{'='*90}")

    header = f"{'Estimator':>12}  {'n':>6}  {'Mean':>8}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_val in sample_sizes:
        rng_base = np.random.default_rng(seed)
        rep_seeds = rng_base.integers(0, 2**31, size=R_reps)

        iv_taus, chan_taus = [], []

        for rep in range(R_reps):
            rng_rep = np.random.default_rng(rep_seeds[rep])
            Y, D, X, mu_true, e_true, true_att = dgp_base(
                rng_rep, n_val, ps_type, or_type, tau=tau, att_mode=True)

            # Fit PS and OR models
            ps_fitted = {}
            for pk in ps_keys:
                mtype, cfunc, _ = PS_MODELS[pk]
                ps_fitted[pk] = fit_ps_model(D, X, mtype, cfunc)
            or_fitted = {}
            ctrl_indicator = (1 - D).astype(float)  # controls: D==0 → indicator==1
            for ok in or_keys:
                cfunc, _ = OR_MODELS[ok]
                or_fitted[ok] = fit_or_model(Y, X, ctrl_indicator, cfunc)

            ps_list = [ps_fitted[pk] for pk in ps_keys]
            or_list = [or_fitted[ok] for ok in or_keys]

            try:
                tau_iv, _, _ = iv_stacking_att(Y, D, ps_list, or_list)
                iv_taus.append(tau_iv)
            except Exception:
                iv_taus.append(np.nan)

            try:
                tau_chan, _ = chan_ols_att(Y, D, ps_list, or_list)
                chan_taus.append(tau_chan)
            except Exception:
                chan_taus.append(np.nan)

        for est, arr in [('IV_ATT', iv_taus), ('Chan_ATT', chan_taus)]:
            mn, bias, sd, rmse = _stats(arr, tau)
            if not np.isnan(mn):
                print(f"{est:>12}  {n_val:>6}  {mn:>8.4f}  {bias:>+8.4f}  "
                      f"{sd:>7.4f}  {rmse:>7.4f}")
        print(sep)


# ═══════════════════════════════════════════════════════════════════════════
#  Section 9: Figures
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(sample_sizes, R_reps, seed=42):
    """Generate all figures."""

    # ── Figure 1: Bias vs n (Scenario A, K=2) ──
    ps_type, or_type = 'logistic_quad', 'trig'
    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 1.0 + 2.0*X_large + 1.5*np.sin(2*X_large)
    true_EY = float(np.mean(mu_large))

    fig, ax = plt.subplots(figsize=(8, 5))
    for est, color, label in [
        ('IV', 'tab:blue', 'IV stacking'),
        ('Chan', 'tab:orange', 'Chan OLS'),
    ]:
        biases, ns = [], []
        for n_val in sample_sizes:
            res = run_scenario('A', n_val, R_reps, seed)
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            if not np.isnan(bias):
                biases.append(bias)
                ns.append(n_val)
        if ns:
            ax.plot(ns, biases, 'o-', color=color, label=label,
                    markersize=8, linewidth=2)

    # Negative control
    biases_c, ns_c = [], []
    for n_val in sample_sizes:
        res = run_scenario('C', n_val, R_reps, seed)
        mn, bias, sd, rmse = _stats(res['IV'], true_EY)
        if not np.isnan(bias):
            biases_c.append(bias)
            ns_c.append(n_val)
    if ns_c:
        ax.plot(ns_c, biases_c, 's--', color='tab:red',
                label='IV (all wrong, Sc.C)', markersize=6)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Bias')
    ax.set_title('Figure 1: Bias vs n (Scenario A, K=2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig1_bias_vs_n.png', dpi=150)
    print("Saved fig1_bias_vs_n.png")
    plt.close(fig)

    # ── Figure 2: RMSE ratio vs K ──
    ks, iv_rmses, chan_rmses = print_table3_scaling(R_reps, n=1000, seed=seed)
    if iv_rmses and chan_rmses and len(iv_rmses) == len(chan_rmses):
        fig, ax = plt.subplots(figsize=(8, 5))
        ratios = [iv/ch if ch > 0 else np.nan
                  for iv, ch in zip(iv_rmses, chan_rmses)]
        ax.plot(ks, ratios, 'o-', color='tab:blue', markersize=8, linewidth=2)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('K (number of PS/OR models)')
        ax.set_ylabel('RMSE(IV) / RMSE(Chan)')
        ax.set_title('Figure 2: RMSE ratio vs K (n=1000, Scenario A)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig('fig2_rmse_ratio.png', dpi=150)
        print("Saved fig2_rmse_ratio.png")
        plt.close(fig)

    # ── Figure 3: Distribution of estimates (K=4, Scenario D) ──
    res_D = run_scenario('D', 1000, R_reps, seed)
    fig, ax = plt.subplots(figsize=(8, 5))
    iv_arr = np.array(res_D['IV'], dtype=float)
    iv_arr = iv_arr[~np.isnan(iv_arr)]
    chan_arr = np.array(res_D['Chan'], dtype=float)
    chan_arr = chan_arr[~np.isnan(chan_arr)]
    if len(iv_arr) > 5 and len(chan_arr) > 5:
        ax.hist(iv_arr, bins=30, alpha=0.5, color='tab:blue',
                label='IV stacking', density=True)
        ax.hist(chan_arr, bins=30, alpha=0.5, color='tab:orange',
                label='Chan OLS', density=True)

        rng_large2 = np.random.default_rng(seed + 999)
        X_l2 = rng_large2.uniform(-2, 2, 500_000)
        mu_l2 = 1.0 + 2.0*X_l2 + 1.5*np.sin(2*X_l2)
        true_EY2 = float(np.mean(mu_l2))
        ax.axvline(x=true_EY2, color='k', linestyle='--', linewidth=2,
                   label=f'True E[Y]={true_EY2:.2f}')
        ax.set_xlabel(r'$\hat\mu$')
        ax.set_ylabel('Density')
        ax.set_title('Figure 3: Distribution (K=4, Scenario D, n=1000)')
        ax.legend()
    fig.tight_layout()
    fig.savefig('fig3_distribution_K4.png', dpi=150)
    print("Saved fig3_distribution_K4.png")
    plt.close(fig)

    # ── Figure 4: Condition number vs K ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ps_key_sets = {
        2: ['PS1', 'PS3'],
        3: ['PS1', 'PS3', 'PS4'],
        4: ['PS1', 'PS3', 'PS4', 'PS5'],
        5: ['PS1', 'PS3', 'PS4', 'PS5', 'PS6'],
        6: ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6'],
    }
    or_key_sets = {
        2: ['OR3', 'OR4'],
        3: ['OR3', 'OR4', 'OR5'],
        4: ['OR3', 'OR4', 'OR5', 'OR6'],
        5: ['OR3', 'OR4', 'OR5', 'OR6', 'OR2'],
        6: ['OR1', 'OR2', 'OR3', 'OR4', 'OR5', 'OR6'],
    }
    iv_cond_data, chan_cond_data = [], []
    for K in [2, 3, 4, 5, 6]:
        SCENARIOS['_temp'] = {
            'desc': f'K={K}',
            'dgp': ('logistic_quad', 'trig'),
            'ps_keys': ps_key_sets[K],
            'or_keys': or_key_sets[K],
        }
        res = run_scenario('_temp', 1000, min(R_reps, 100), seed)
        del SCENARIOS['_temp']
        iv_c = np.array(res['IV_cond'], dtype=float)
        iv_c = iv_c[~np.isnan(iv_c)]
        chan_c = np.array(res['Chan_cond'], dtype=float)
        chan_c = chan_c[~np.isnan(chan_c)]
        iv_cond_data.append(iv_c if len(iv_c) > 0 else np.array([np.nan]))
        chan_cond_data.append(chan_c if len(chan_c) > 0 else np.array([np.nan]))

    positions_iv = np.arange(5) * 3
    positions_chan = positions_iv + 1
    bp1 = ax.boxplot(iv_cond_data, positions=positions_iv, widths=0.8,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(chan_cond_data, positions=positions_chan, widths=0.8,
                     patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('tab:blue')
        patch.set_alpha(0.5)
    for patch in bp2['boxes']:
        patch.set_facecolor('tab:orange')
        patch.set_alpha(0.5)
    ax.set_xticks(positions_iv + 0.5)
    ax.set_xticklabels([str(k) for k in [2, 3, 4, 5, 6]])
    ax.set_xlabel('K')
    ax.set_ylabel('Condition number')
    ax.set_yscale('log')
    ax.set_title('Figure 4: Condition number vs K (n=1000)')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]],
              ["IV (O'M)", "Chan (U'U)"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig4_condition_numbers.png', dpi=150)
    print("Saved fig4_condition_numbers.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Section 10: Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    R_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample_sizes = [300, 1000, 5000]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Stacked IV Multiply Robust Estimator — Simulation Study   ║")
    print(f"║  R = {R_reps}, n ∈ {sample_sizes}                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Tables 1 & 2: K=2 and K=4 scenarios
    print_table1(['A', 'B', 'C'], sample_sizes, R_reps)
    print_table1(['D', 'E'], sample_sizes, R_reps)

    # Table 3: Scaling
    print_table3_scaling(R_reps, n=1000)

    # Robustness 5.1: All correct
    run_robustness_all_correct(R_reps, sample_sizes)

    # Robustness 5.4: ATT
    run_att_robustness(R_reps, sample_sizes)

    # Figures (only if R >= 5)
    if R_reps >= 5:
        make_figures(sample_sizes, R_reps)


if __name__ == '__main__':
    main()
