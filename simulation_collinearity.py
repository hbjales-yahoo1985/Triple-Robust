"""
Simulation: Chan vs IV Stacking Under Model Similarity & Accidental Spanning
=============================================================================

**KEY FINDING**: IV stacking has OR-channel robustness only (not PS-channel).
When the correct PS is present but no OR model spans μ(X), IV has persistent
bias. Chan converges because its {1, 1/π̂} terms provide PS-channel correction.
IV's consistency requires μ(X) ∈ span(m̂₁,...,m̂_K) under the weighted inner
product w(X) = e²(X)/(1-e(X)), which is NOT guaranteed by correct PS alone.

**INTERCEPT TEST**: Adding a constant column to both instruments and regressors
(IV_int) does NOT restore the PS-channel. At R=100, n=5000: IV bias=+0.048,
IV_int bias=-0.080, Chan bias=+0.004. The constant-on-constant moment only
pins E[Y|R=1], not E[Y]. PS-channel correction requires 1/π̂ as a REGRESSOR
(as Chan does), because only regressors enter the prediction μ̂ = (1/n)ΣM·φ.
Instruments affect coefficient estimation but not the prediction formula.

Two questions:

1. **No accidental spanning**: Craft a DGP where the OR/PS working model
   basis functions do NOT accidentally span the true outcome regression.
   This gives a fairer comparison — accidental spanning is a property of
   the specific DGP, not a general property of the method.

2. **High model similarity / DGP complexity**: Chan's OLS stacking uses
   {1, 1/π̂₁,..., â₁,...} as regressors. When the true DGP is simple (e.g.,
   linear in X), all working models' fitted values collapse onto nearly
   the same function, making Chan's regressor matrix near-singular.
   Despite enormous condition numbers, Chan *may* remain stable if models
   are "different enough" in the function classes they span. But the
   simpler the DGP, the more Chan pays for its extra parameters.

   Key insight: OLS fitted values from {1, X, g(X)} will all converge to
   {1, X} when the truth is linear, regardless of g. This makes the OR
   columns in Chan's regressor matrix near-identical.

   Hypothesis: IV stacking is more robust to this because it fits fewer
   parameters and uses a fundamentally different identification strategy.

Scenarios:
  S1 — No-spanning: true OR = sin(3X), working models use polynomials/exp.
       Verify span R² << 1 for working model basis vs true OR.
  S2 — Diverse models (current setup): different functional forms.
  S3 — Similar models (high collinearity): all OR models include {1,X},
       differ only in a small extra term. True DGP is linear (simplest case).
  S4 — DGP complexity gradient: SAME working models, but vary the true DGP
       from simple (linear) to complex (trig). Simpler DGP → more similar
       fitted values → higher condition number → (hypothesis) worse for Chan.

Run:
    python simulation_collinearity.py          # R=10 smoke test
    python simulation_collinearity.py 500      # R=500 production run
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
warnings.filterwarnings('ignore', category=FutureWarning)

PS_CLIP = (0.01, 0.99)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers (shared with stacked_iv_mr.py)
# ═══════════════════════════════════════════════════════════════════════════

def _clip_ps(p):
    return np.clip(p, PS_CLIP[0], PS_CLIP[1])


def _invlogit(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))


def fit_ps_logit(D, X, covariate_func):
    """Fit logit PS model."""
    n = len(D)
    features = covariate_func(X)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xd = sm.add_constant(features)
    try:
        mod = Logit(D, Xd).fit(disp=False, maxiter=200)
        return _clip_ps(mod.predict(Xd))
    except Exception:
        return np.full(n, np.clip(np.mean(D), *PS_CLIP))


def fit_or_model(Y, X, R, covariate_func):
    """Fit OR model by OLS among R==1, predict for all."""
    n = len(Y)
    features = covariate_func(X)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xd_all = sm.add_constant(features)
    complete = (R == 1)
    try:
        coef, _, _, _ = np.linalg.lstsq(Xd_all[complete], Y[complete],
                                         rcond=None)
        return Xd_all @ coef
    except Exception:
        return np.full(n, np.nanmean(Y[complete]))


# ═══════════════════════════════════════════════════════════════════════════
#  Core estimators
# ═══════════════════════════════════════════════════════════════════════════

def chan_ols_estimator(Y, R, ps_fitted_list, or_fitted_list):
    """Chan (2013) OLS stacking. Returns (mu_hat, cond_num)."""
    n = len(Y)
    complete = (R == 1)
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
    return float(np.mean(U_all @ coef)), cond_num


def iv_stacking_estimator(Y, R, ps_fitted_list, or_fitted_list):
    """IV stacking. Returns (mu_hat, cond_num)."""
    n = len(Y)
    complete = (R == 1)
    M_all = np.column_stack(or_fitted_list)
    O_all = np.column_stack([_clip_ps(ps)/(1.0 - _clip_ps(ps))
                             for ps in ps_fitted_list])
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
    return float(np.mean(M_all @ phi)), cond_num


def iv_stacking_with_intercept(Y, R, ps_fitted_list, or_fitted_list):
    """
    IV stacking WITH intercept — adds constant to both regressors and instruments.

    Regressors: {1, m̂_1,...,m̂_K}  (K+1 columns)
    Instruments: {1, o_1,...,o_K}   (K+1 columns)

    The constant-on-constant moment pins E[Y|R=1] = φ₀ + Σ φ_k E[m̂_k|R=1],
    and the odds moments handle reweighting → should restore PS-channel.

    Returns (mu_hat, cond_num).
    """
    n = len(Y)
    complete = (R == 1)
    M_all = np.column_stack([np.ones(n)] + list(or_fitted_list))
    O_all = np.column_stack([np.ones(n)] +
                            [_clip_ps(ps)/(1.0 - _clip_ps(ps))
                             for ps in ps_fitted_list])
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
    return float(np.mean(M_all @ phi)), cond_num


def single_dr_estimator(Y, R, ps_fitted, or_fitted):
    """Bang-Robins DR: μ̂ = (1/n) Σ [â(X) + R(Y-â(X))/π̂(X)]."""
    ps_c = _clip_ps(ps_fitted)
    return float(np.mean(or_fitted + R * (Y - or_fitted) / ps_c))


def oracle_estimator(mu_true):
    """Oracle: (1/n) Σ μ(X_i)."""
    return float(np.mean(mu_true))


# ═══════════════════════════════════════════════════════════════════════════
#  DGPs
# ═══════════════════════════════════════════════════════════════════════════

def dgp_no_spanning(rng, n):
    """
    DGP S1: No accidental spanning.

    True OR: μ(X) = 1 + 2·sin(3X)  — oscillatory, not in span of
             polynomials or simple exponentials.
    True PS: logit(-0.5 + 0.3X - 0.2X²) — standard logistic quadratic.

    Working OR models will use polynomials and exp, which don't span sin(3X).
    Working PS: PS1 = logit on {X, X²} (correct).
    """
    X = rng.uniform(-2, 2, n)
    logit_e = -0.5 + 0.3*X - 0.2*X**2
    e_true = _clip_ps(_invlogit(logit_e))
    R = rng.binomial(1, e_true, n).astype(float)
    mu = 1.0 + 2.0*np.sin(3*X)
    Y_full = mu + rng.normal(0, 1, n)
    Y_obs = np.where(R == 1, Y_full, np.nan)
    return Y_obs, R, X, mu, e_true


def dgp_simple_linear(rng, n):
    """
    DGP S3/S4: Simple true model (linear).

    True OR: μ(X) = 2 + 1.5X  — the simplest nontrivial model.
    True PS: logit(-0.5 + 0.3X - 0.2X²).

    When all working OR models include {1, X}, their fitted values will
    be nearly identical, making Chan's regressor matrix near-singular.
    This is the "easy DGP, hard for Chan" case.
    """
    X = rng.uniform(-2, 2, n)
    logit_e = -0.5 + 0.3*X - 0.2*X**2
    e_true = _clip_ps(_invlogit(logit_e))
    R = rng.binomial(1, e_true, n).astype(float)
    mu = 2.0 + 1.5*X
    Y_full = mu + rng.normal(0, 1, n)
    Y_obs = np.where(R == 1, Y_full, np.nan)
    return Y_obs, R, X, mu, e_true


def dgp_diverse(rng, n):
    """
    DGP S2: Standard diverse setup (trig true OR).

    True OR: μ(X) = 1 + 2X + 1.5sin(2X).
    True PS: logit(-0.5 + 0.3X - 0.2X²).
    """
    X = rng.uniform(-2, 2, n)
    logit_e = -0.5 + 0.3*X - 0.2*X**2
    e_true = _clip_ps(_invlogit(logit_e))
    R = rng.binomial(1, e_true, n).astype(float)
    mu = 1.0 + 2.0*X + 1.5*np.sin(2*X)
    Y_full = mu + rng.normal(0, 1, n)
    Y_obs = np.where(R == 1, Y_full, np.nan)
    return Y_obs, R, X, mu, e_true


# ═══════════════════════════════════════════════════════════════════════════
#  Working model catalogs
# ═══════════════════════════════════════════════════════════════════════════

# ── Diverse working models (models are structurally different) ──
def _or_quad(X):     return np.column_stack([X, X**2])
def _or_exp(X):      return np.exp(X).reshape(-1, 1)
def _or_linear(X):   return X.reshape(-1, 1)
def _or_cubic(X):    return np.column_stack([X, X**3])

def _ps_quad(X):     return np.column_stack([X, X**2])
def _ps_sin(X):      return np.sin(X).reshape(-1, 1)
def _ps_linear(X):   return X.reshape(-1, 1)
def _ps_abs(X):      return np.abs(X).reshape(-1, 1)

# ── Similar working models: all include {X} + a small extra term ──
# These create near-collinear fitted values when the true DGP is simple.
def _or_sim_x2(X):   return np.column_stack([X, X**2])
def _or_sim_x3(X):   return np.column_stack([X, X**3])
def _or_sim_sinx(X): return np.column_stack([X, np.sin(X)])
def _or_sim_logx(X): return np.column_stack([X, np.log(np.abs(X) + 1)])
def _or_sim_expx(X): return np.column_stack([X, np.exp(X/2)])
def _or_sim_absx(X): return np.column_stack([X, np.abs(X)])

def _ps_sim_x2(X):   return np.column_stack([X, X**2])
def _ps_sim_x3(X):   return np.column_stack([X, X**3])
def _ps_sim_sinx(X): return np.column_stack([X, np.sin(X)])
def _ps_sim_logx(X): return np.column_stack([X, np.log(np.abs(X) + 1)])
def _ps_sim_expx(X): return np.column_stack([X, np.exp(X/2)])
def _ps_sim_absx(X): return np.column_stack([X, np.abs(X)])

# (Scaled similarity functions removed — OLS fitted values are invariant
#  to column scaling, so the α-scaling approach doesn't work.
#  Instead, S4 varies DGP complexity to control fitted value similarity.)


# ═══════════════════════════════════════════════════════════════════════════
#  Spanning diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def check_spanning(X, mu_true, or_cov_funcs, ps_cov_funcs):
    """
    Check whether the working model basis spans the true OR.

    Compute R² of regressing mu_true on the union of OR and PS basis functions.
    If R² ≈ 1, the models accidentally span the truth.
    """
    cols = [np.ones_like(X)]
    for f in or_cov_funcs:
        fv = f(X)
        if fv.ndim == 1:
            fv = fv.reshape(-1, 1)
        for j in range(fv.shape[1]):
            cols.append(fv[:, j])
    # Also include 1/π̂ terms basis
    for f in ps_cov_funcs:
        fv = f(X)
        if fv.ndim == 1:
            fv = fv.reshape(-1, 1)
        for j in range(fv.shape[1]):
            cols.append(fv[:, j])

    Z = np.column_stack(cols)
    # Regress mu_true on Z
    coef, res, _, _ = np.linalg.lstsq(Z, mu_true, rcond=None)
    mu_hat = Z @ coef
    ss_res = np.sum((mu_true - mu_hat)**2)
    ss_tot = np.sum((mu_true - np.mean(mu_true))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2


def _regressor_correlation(R, ps_fitted_list, or_fitted_list):
    """Compute pairwise correlation of Chan's regressor columns among R==1."""
    complete = (R == 1)
    n = int(complete.sum())
    cols = [np.ones(n)]
    for ps in ps_fitted_list:
        cols.append(1.0 / _clip_ps(ps[complete]))
    for orr in or_fitted_list:
        cols.append(orr[complete])
    U = np.column_stack(cols)
    # Correlation matrix
    means = U.mean(axis=0)
    stds = U.std(axis=0)
    stds[stds < 1e-12] = 1.0
    U_norm = (U - means) / stds
    corr = (U_norm.T @ U_norm) / n
    return corr


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation runner
# ═══════════════════════════════════════════════════════════════════════════

def _stats(arr, true_val):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mn = float(np.mean(a))
    bias = mn - true_val
    sd = float(np.std(a))
    rmse = float(np.sqrt(bias**2 + sd**2))
    return mn, bias, sd, rmse


def run_scenario_generic(dgp_func, ps_cov_funcs, or_cov_funcs,
                         n, R_reps, seed=42):
    """
    Run a generic scenario.

    Parameters
    ----------
    dgp_func : callable(rng, n) -> (Y_obs, R, X, mu, e_true)
    ps_cov_funcs : list of callables X -> features
    or_cov_funcs : list of callables X -> features

    Returns
    -------
    dict with keys: IV, Chan, DR_best, DR_worst, Oracle,
                    IV_cond, Chan_cond, span_r2, regressor_corr_max
    """
    K = len(ps_cov_funcs)
    assert K == len(or_cov_funcs), "K must match for IV stacking"

    results = {
        'IV': [], 'IV_int': [], 'Chan': [], 'Oracle': [],
        'IV_cond': [], 'IV_int_cond': [], 'Chan_cond': [],
        'DR_best': [], 'DR_worst': [],
        'span_r2': [], 'regressor_corr_max': [],
    }

    rng_base = np.random.default_rng(seed)
    rep_seeds = rng_base.integers(0, 2**31, size=R_reps)

    for rep in range(R_reps):
        rng_rep = np.random.default_rng(rep_seeds[rep])
        Y_obs, R, X, mu, e_true = dgp_func(rng_rep, n)

        # Spanning check (first rep only, it's deterministic given X)
        if rep == 0:
            r2 = check_spanning(X, mu, or_cov_funcs, ps_cov_funcs)
            results['span_r2'].append(r2)

        # Fit models
        Y_safe = np.where(np.isnan(Y_obs), 0.0, Y_obs)

        ps_fitted = []
        for pf in ps_cov_funcs:
            ps_fitted.append(fit_ps_logit(R, X, pf))

        or_fitted = []
        for of_ in or_cov_funcs:
            or_fitted.append(fit_or_model(Y_safe, X, R, of_))

        # Regressor correlation (first rep)
        if rep == 0:
            corr = _regressor_correlation(R, ps_fitted, or_fitted)
            # Max off-diagonal absolute correlation
            np.fill_diagonal(corr, 0)
            results['regressor_corr_max'].append(float(np.max(np.abs(corr))))

        # Oracle
        results['Oracle'].append(float(np.mean(mu)))

        # IV stacking
        try:
            mu_iv, cond_iv = iv_stacking_estimator(
                Y_safe, R, ps_fitted, or_fitted)
            results['IV'].append(mu_iv)
            results['IV_cond'].append(cond_iv)
        except Exception:
            results['IV'].append(np.nan)
            results['IV_cond'].append(np.nan)

        # IV stacking with intercept
        try:
            mu_iv_int, cond_iv_int = iv_stacking_with_intercept(
                Y_safe, R, ps_fitted, or_fitted)
            results['IV_int'].append(mu_iv_int)
            results['IV_int_cond'].append(cond_iv_int)
        except Exception:
            results['IV_int'].append(np.nan)
            results['IV_int_cond'].append(np.nan)

        # Chan OLS
        try:
            mu_chan, cond_chan = chan_ols_estimator(
                Y_safe, R, ps_fitted, or_fitted)
            results['Chan'].append(mu_chan)
            results['Chan_cond'].append(cond_chan)
        except Exception:
            results['Chan'].append(np.nan)
            results['Chan_cond'].append(np.nan)

        # All DR pairs — track best and worst
        dr_vals = []
        for j in range(K):
            for k in range(K):
                try:
                    dr_val = single_dr_estimator(
                        Y_safe, R, ps_fitted[j], or_fitted[k])
                    dr_vals.append(dr_val)
                except Exception:
                    pass

        if dr_vals:
            true_ey = float(np.mean(mu))
            dr_biases = [abs(v - true_ey) for v in dr_vals]
            results['DR_best'].append(dr_vals[np.argmin(dr_biases)])
            results['DR_worst'].append(dr_vals[np.argmax(dr_biases)])
        else:
            results['DR_best'].append(np.nan)
            results['DR_worst'].append(np.nan)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario S1: No accidental spanning
# ═══════════════════════════════════════════════════════════════════════════

def run_S1(R_reps, sample_sizes, seed=42):
    """
    S1: No-spanning DGP.

    True OR = 1 + 2·sin(3X). Working OR: polynomials and exp.
    PS1 = logit on {X, X²} (correct).
    PS2 = logit on {sin(X)} (wrong).

    sin(3X) is not in the span of {1, X, X², exp(X), X³}.
    """
    ps_funcs = [_ps_quad, _ps_sin]  # PS1 correct
    or_funcs = [_or_exp, _or_linear]  # both wrong

    # Compute true E[Y]
    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 1.0 + 2.0*np.sin(3*X_large)
    true_EY = float(np.mean(mu_large))

    # Spanning check on a large sample
    r2 = check_spanning(X_large, mu_large, or_funcs, ps_funcs)

    print(f"\n{'='*90}")
    print(f"S1: No-Spanning DGP — μ(X) = 1 + 2·sin(3X)")
    print(f"PS: logit on {{X,X²}} (correct), logit on {{sin(X)}} (wrong)")
    print(f"OR: OLS on {{1,exp(X)}} (wrong), OLS on {{1,X}} (wrong)")
    print(f"True E[Y] = {true_EY:.4f}")
    print(f"Spanning R² = {r2:.4f}  (should be << 1)")
    print(f"{'='*90}")

    header = (f"{'Est':>10}  {'n':>6}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}  "
              f"{'Cond_med':>12}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_val in sample_sizes:
        res = run_scenario_generic(dgp_no_spanning, ps_funcs, or_funcs,
                                   n_val, R_reps, seed)
        for est in ['IV', 'IV_int', 'Chan', 'DR_best', 'DR_worst']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            cond_key = f'{est}_cond' if est in ['IV', 'IV_int', 'Chan'] else None
            if cond_key and cond_key in res:
                conds = np.array(res[cond_key], dtype=float)
                conds = conds[~np.isnan(conds)]
                cond_med = float(np.median(conds)) if len(conds) > 0 else np.nan
            else:
                cond_med = np.nan

            if not np.isnan(mn):
                cond_str = f"{cond_med:.1f}" if not np.isnan(cond_med) else ""
                print(f"{est:>10}  {n_val:>6}  {bias:>+8.4f}  {sd:>7.4f}  "
                      f"{rmse:>7.4f}  {cond_str:>12}")
        print(sep)


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario S2: Diverse models (baseline)
# ═══════════════════════════════════════════════════════════════════════════

def run_S2(R_reps, sample_sizes, seed=42):
    """S2: Diverse models — structural variety keeps Chan stable."""
    ps_funcs = [_ps_quad, _ps_sin]  # PS1 correct
    or_funcs = [_or_exp, _or_linear]  # both wrong (trig DGP)

    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 1.0 + 2.0*X_large + 1.5*np.sin(2*X_large)
    true_EY = float(np.mean(mu_large))

    r2 = check_spanning(X_large, mu_large, or_funcs, ps_funcs)

    print(f"\n{'='*90}")
    print(f"S2: Diverse Models (baseline) — μ(X) = 1 + 2X + 1.5·sin(2X)")
    print(f"PS: logit on {{X,X²}} (correct), logit on {{sin(X)}} (wrong)")
    print(f"OR: OLS on {{1,exp(X)}} (wrong), OLS on {{1,X}} (wrong)")
    print(f"True E[Y] = {true_EY:.4f}")
    print(f"Spanning R² = {r2:.4f}")
    print(f"{'='*90}")

    header = (f"{'Est':>10}  {'n':>6}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}  "
              f"{'Cond_med':>12}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_val in sample_sizes:
        res = run_scenario_generic(dgp_diverse, ps_funcs, or_funcs,
                                   n_val, R_reps, seed)
        for est in ['IV', 'IV_int', 'Chan', 'DR_best', 'DR_worst']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            cond_key = f'{est}_cond' if est in ['IV', 'IV_int', 'Chan'] else None
            if cond_key and cond_key in res:
                conds = np.array(res[cond_key], dtype=float)
                conds = conds[~np.isnan(conds)]
                cond_med = float(np.median(conds)) if len(conds) > 0 else np.nan
            else:
                cond_med = np.nan

            if not np.isnan(mn):
                cond_str = f"{cond_med:.1f}" if not np.isnan(cond_med) else ""
                print(f"{est:>10}  {n_val:>6}  {bias:>+8.4f}  {sd:>7.4f}  "
                      f"{rmse:>7.4f}  {cond_str:>12}")
        print(sep)


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario S3: Similar models — high collinearity
# ═══════════════════════════════════════════════════════════════════════════

def run_S3(R_reps, sample_sizes, seed=42):
    """
    S3: High model similarity + simple DGP.

    True OR = 2 + 1.5X (linear). All working OR models include {X} plus
    a different extra term. Since the truth is linear, the extra terms are
    pure noise — all fitted values collapse to ~{1, X}, making Chan's
    regressor matrix nearly singular.

    This is the "easy DGP, hard for Chan" scenario.
    """
    # Similar PS models: all logit on {X, g(X)}
    ps_funcs = [_ps_sim_x2, _ps_sim_sinx, _ps_sim_logx, _ps_sim_expx]
    # Similar OR models: all OLS on {1, X, g(X)}
    or_funcs = [_or_sim_x2, _or_sim_sinx, _or_sim_logx, _or_sim_expx]

    # PS_sim_x2 = logit on {X, X²} is correct

    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 2.0 + 1.5*X_large
    true_EY = float(np.mean(mu_large))

    r2 = check_spanning(X_large, mu_large, or_funcs, ps_funcs)

    print(f"\n{'='*90}")
    print(f"S3: High-Similarity Models + Simple DGP — μ(X) = 2 + 1.5X")
    print(f"All OR models: {{1, X, g(X)}} where g varies. K=4.")
    print(f"All PS models: logit on {{X, g(X)}} where g varies. K=4.")
    print(f"PS1=logit(X,X²) is correct. All OR models correct (contain {{1,X}}).")
    print(f"True E[Y] = {true_EY:.4f}")
    print(f"Spanning R² = {r2:.4f}")
    print(f"Chan fits 2K+1 = 9 params. IV fits K = 4.")
    print(f"{'='*90}")

    header = (f"{'Est':>10}  {'n':>6}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}  "
              f"{'Cond_med':>12}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_val in sample_sizes:
        res = run_scenario_generic(dgp_simple_linear, ps_funcs, or_funcs,
                                   n_val, R_reps, seed)

        # Print regressor correlation diagnostics (first rep)
        if res['regressor_corr_max']:
            print(f"  Max off-diag |corr| in Chan regressors: "
                  f"{res['regressor_corr_max'][0]:.4f}")

        for est in ['IV', 'IV_int', 'Chan', 'DR_best', 'DR_worst']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            cond_key = f'{est}_cond' if est in ['IV', 'IV_int', 'Chan'] else None
            if cond_key and cond_key in res:
                conds = np.array(res[cond_key], dtype=float)
                conds = conds[~np.isnan(conds)]
                cond_med = float(np.median(conds)) if len(conds) > 0 else np.nan
            else:
                cond_med = np.nan

            if not np.isnan(mn):
                cond_str = f"{cond_med:.1f}" if not np.isnan(cond_med) else ""
                print(f"{est:>10}  {n_val:>6}  {bias:>+8.4f}  {sd:>7.4f}  "
                      f"{rmse:>7.4f}  {cond_str:>12}")
        print(sep)


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario S4: DGP complexity gradient
# ═══════════════════════════════════════════════════════════════════════════

def _dgp_parametric(rng, n, complexity):
    """
    DGP with tunable complexity.

    complexity ∈ {0, 1, 2, 3}:
      0: μ = 2 + 1.5X                          (linear — simplest)
      1: μ = 2 + 1.5X + 0.3X²                  (slightly nonlinear)
      2: μ = 1 + 2X + 1.5X²                    (quadratic)
      3: μ = 1 + 2X + 1.5sin(2X)               (trig — most complex)

    PS: logit(-0.5 + 0.3X - 0.2X²) for all.
    """
    X = rng.uniform(-2, 2, n)
    logit_e = -0.5 + 0.3*X - 0.2*X**2
    e_true = _clip_ps(_invlogit(logit_e))
    R = rng.binomial(1, e_true, n).astype(float)

    if complexity == 0:
        mu = 2.0 + 1.5*X
    elif complexity == 1:
        mu = 2.0 + 1.5*X + 0.3*X**2
    elif complexity == 2:
        mu = 1.0 + 2.0*X + 1.5*X**2
    elif complexity == 3:
        mu = 1.0 + 2.0*X + 1.5*np.sin(2*X)
    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    Y_full = mu + rng.normal(0, 1, n)
    Y_obs = np.where(R == 1, Y_full, np.nan)
    return Y_obs, R, X, mu, e_true


def _or_fitted_correlation(R, or_fitted_list):
    """Compute pairwise correlation of OR fitted values among R==1."""
    complete = (R == 1)
    K = len(or_fitted_list)
    vals = np.column_stack([o[complete] for o in or_fitted_list])
    corr = np.corrcoef(vals, rowvar=False)
    # Mean off-diagonal absolute correlation
    mask = ~np.eye(K, dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


def run_S4(R_reps, n=1000, seed=42):
    """
    S4: DGP complexity gradient.

    SAME working models for all DGPs. Vary the true DGP from simple (linear)
    to complex (trig). The simpler the DGP, the more similar the OR fitted
    values become (all collapse toward the simple truth), making Chan's
    regressor matrix more ill-conditioned.

    Hypothesis: Chan's RMSE is worst when the DGP is simplest, because
    that's when models produce the most similar fitted values.
    """
    # Same models for all
    ps_funcs = [_ps_sim_x2, _ps_sim_sinx, _ps_sim_logx, _ps_sim_expx]
    or_funcs = [_or_sim_x2, _or_sim_sinx, _or_sim_logx, _or_sim_expx]

    complexity_labels = {
        0: ('linear: 2+1.5X', lambda X: 2.0 + 1.5*X),
        1: ('slight quad: 2+1.5X+0.3X²', lambda X: 2.0 + 1.5*X + 0.3*X**2),
        2: ('quadratic: 1+2X+1.5X²', lambda X: 1.0 + 2.0*X + 1.5*X**2),
        3: ('trig: 1+2X+1.5sin(2X)', lambda X: 1.0 + 2.0*X + 1.5*np.sin(2*X)),
    }

    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)

    print(f"\n{'='*90}")
    print(f"S4: DGP Complexity Gradient — n={n}, K=4, R={R_reps}")
    print(f"SAME working models for all DGPs.")
    print(f"PS: logit on {{X,g(X)}} — PS1 (X,X²) always correct.")
    print(f"OR: OLS on {{1,X,g(X)}} — none correct for trig, all correct for linear.")
    print(f"Hypothesis: simpler DGP → more similar fitted values → worse for Chan")
    print(f"{'='*90}")

    header = (f"{'Complexity':>6}  {'DGP':>30}  {'Est':>6}  {'Bias':>8}  "
              f"{'SD':>7}  {'RMSE':>7}  {'Cond_med':>12}  {'OR_corr':>8}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    complexities = [0, 1, 2, 3]
    iv_rmses, chan_rmses, chan_conds_med = [], [], []

    for c in complexities:
        label, mu_func = complexity_labels[c]
        mu_large = mu_func(X_large)
        true_EY = float(np.mean(mu_large))

        dgp_func = lambda rng, n, _c=c: _dgp_parametric(rng, n, _c)

        res = run_scenario_generic(dgp_func, ps_funcs, or_funcs,
                                   n, R_reps, seed)

        # Compute OR fitted value correlation for one representative rep
        rng_rep = np.random.default_rng(
            np.random.default_rng(seed).integers(0, 2**31, size=1)[0])
        Y_obs, R, X, mu, e_true = dgp_func(rng_rep, n)
        Y_safe = np.where(np.isnan(Y_obs), 0.0, Y_obs)
        or_fitted_rep = [fit_or_model(Y_safe, X, R, f) for f in or_funcs]
        or_corr = _or_fitted_correlation(R, or_fitted_rep)

        for est in ['IV', 'IV_int', 'Chan']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            conds = np.array(res[f'{est}_cond'], dtype=float)
            conds = conds[~np.isnan(conds)]
            cond_med = float(np.median(conds)) if len(conds) > 0 else np.nan

            if not np.isnan(mn):
                print(f"{c:>6}  {label:>30}  {est:>6}  {bias:>+8.4f}  "
                      f"{sd:>7.4f}  {rmse:>7.4f}  {cond_med:>12.1f}  "
                      f"{or_corr:>8.4f}")
            else:
                print(f"{c:>6}  {label:>30}  {est:>6}  {'NaN':>8}  "
                      f"{'NaN':>7}  {'NaN':>7}  {'NaN':>12}  {or_corr:>8.4f}")

            if est == 'IV' and not np.isnan(rmse):
                iv_rmses.append(rmse)
            elif est == 'Chan' and not np.isnan(rmse):
                chan_rmses.append(rmse)
                chan_conds_med.append(cond_med)

        print(sep)

    return complexities, iv_rmses, chan_rmses, chan_conds_med


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario S5: Similar models + no spanning (worst case for Chan)
# ═══════════════════════════════════════════════════════════════════════════

def run_S5(R_reps, sample_sizes, seed=42):
    """
    S5: Similar models + no-spanning DGP.

    This is the critical combination: models are structurally similar
    (all include {1,X,g(X)}), BUT the true OR = 1+2sin(3X) is NOT in
    their span. PS1 is correct.

    This combines:
    - High condition number (similar models → near-identical fitted values)
    - Model misspecification (truth not spanned → fitted values are wrong)

    The user's hypothesis: Chan pays the BIGGEST price here because
    collinearity prevents OLS from effectively using the 1/π̂ terms to
    correct for misspecification.
    """
    ps_funcs = [_ps_sim_x2, _ps_sim_sinx, _ps_sim_logx, _ps_sim_expx]
    or_funcs = [_or_sim_x2, _or_sim_sinx, _or_sim_logx, _or_sim_expx]

    rng_large = np.random.default_rng(seed + 999)
    X_large = rng_large.uniform(-2, 2, 500_000)
    mu_large = 1.0 + 2.0*np.sin(3*X_large)
    true_EY = float(np.mean(mu_large))

    r2 = check_spanning(X_large, mu_large, or_funcs, ps_funcs)

    # Compare with diverse models on same DGP
    ps_diverse = [_ps_quad, _ps_sin]
    or_diverse = [_or_exp, _or_linear]
    r2_diverse = check_spanning(X_large, mu_large, or_diverse, ps_diverse)

    print(f"\n{'='*90}")
    print(f"S5: Similar Models + No-Spanning DGP — μ(X) = 1 + 2·sin(3X)")
    print(f"All OR: {{1, X, g(X)}} — K=4 similar models, none correct.")
    print(f"PS: logit on {{X, g(X)}} — PS1 (X,X²) correct. K=4.")
    print(f"True E[Y] = {true_EY:.4f}")
    print(f"Spanning R² (similar models) = {r2:.4f}")
    print(f"Spanning R² (diverse models, S1) = {r2_diverse:.4f}")
    print(f"This is the WORST CASE: high condition + misspecification.")
    print(f"{'='*90}")

    header = (f"{'Est':>10}  {'n':>6}  {'Bias':>8}  {'SD':>7}  {'RMSE':>7}  "
              f"{'Cond_med':>12}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for n_val in sample_sizes:
        res = run_scenario_generic(dgp_no_spanning, ps_funcs, or_funcs,
                                   n_val, R_reps, seed)

        if res['regressor_corr_max']:
            print(f"  Max off-diag |corr| in Chan regressors: "
                  f"{res['regressor_corr_max'][0]:.4f}")

        for est in ['IV', 'IV_int', 'Chan', 'DR_best', 'DR_worst']:
            mn, bias, sd, rmse = _stats(res[est], true_EY)
            cond_key = f'{est}_cond' if est in ['IV', 'IV_int', 'Chan'] else None
            if cond_key and cond_key in res:
                conds = np.array(res[cond_key], dtype=float)
                conds = conds[~np.isnan(conds)]
                cond_med = float(np.median(conds)) if len(conds) > 0 else np.nan
            else:
                cond_med = np.nan

            if not np.isnan(mn):
                cond_str = f"{cond_med:.1f}" if not np.isnan(cond_med) else ""
                print(f"{est:>10}  {n_val:>6}  {bias:>+8.4f}  {sd:>7.4f}  "
                      f"{rmse:>7.4f}  {cond_str:>12}")
        print(sep)

    # Print comparison with S1 (diverse models, same DGP)
    print(f"\n  Compare with S1 (diverse models, same DGP):")
    print(f"  If similar models are worse, Chan pays a premium for collinearity")
    print(f"  when combined with model misspecification.")


# ═══════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(complexities, iv_rmses, chan_rmses, chan_conds_med, R_reps):
    """Generate diagnostic figures."""

    labels = ['linear', 'slight\nquad', 'quad', 'trig']

    # ── Figure A: RMSE by DGP complexity ──
    if len(iv_rmses) == len(complexities) and len(chan_rmses) == len(complexities):
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x_pos = np.arange(len(complexities))
        width = 0.35
        ax1.bar(x_pos - width/2, chan_rmses, width, color='tab:orange',
                label='Chan RMSE', alpha=0.8)
        ax1.bar(x_pos + width/2, iv_rmses, width, color='tab:blue',
                label='IV RMSE', alpha=0.8)
        ax1.set_xlabel('DGP Complexity', fontsize=12)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # Condition number on secondary axis
        if len(chan_conds_med) == len(complexities):
            ax2 = ax1.twinx()
            ax2.plot(x_pos, chan_conds_med, 'x--', color='tab:red',
                     label="Chan cond(U'U)", markersize=10, linewidth=2)
            ax2.set_ylabel('Condition number (Chan)', color='tab:red',
                           fontsize=12)
            ax2.set_yscale('log')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.legend(loc='upper left', fontsize=11)

        ax1.set_title(f'RMSE by DGP Complexity (n=1000, K=4, R={R_reps})\n'
                      f'Simpler DGP → more similar fitted values → '
                      f'higher condition number',
                      fontsize=11)
        fig.tight_layout()
        fig.savefig('fig_complexity_rmse.png', dpi=150)
        print("\nSaved fig_complexity_rmse.png")
        plt.close(fig)

    # ── Figure B: RMSE ratio ──
    if len(iv_rmses) == len(complexities) and len(chan_rmses) == len(complexities):
        fig, ax = plt.subplots(figsize=(8, 5))
        ratios = [iv/ch if ch > 0 else np.nan
                  for iv, ch in zip(iv_rmses, chan_rmses)]
        ax.bar(np.arange(len(complexities)), ratios, color='tab:purple',
               alpha=0.8)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('DGP Complexity', fontsize=12)
        ax.set_ylabel('RMSE(IV) / RMSE(Chan)', fontsize=12)
        ax.set_xticks(np.arange(len(complexities)))
        ax.set_xticklabels(labels)
        ax.set_title(f'IV/Chan RMSE Ratio by DGP Complexity '
                     f'(n=1000, K=4, R={R_reps})', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig('fig_complexity_ratio.png', dpi=150)
        print("Saved fig_complexity_ratio.png")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    R_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample_sizes = [300, 1000, 5000]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Collinearity & Spanning — Chan vs IV Diagnostic Study     ║")
    print(f"║  R = {R_reps}                                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # S1: No accidental spanning
    run_S1(R_reps, sample_sizes)

    # S2: Diverse models baseline
    run_S2(R_reps, sample_sizes)

    # S3: Similar models + simple DGP
    run_S3(R_reps, sample_sizes)

    # S5: Similar models + no spanning (worst case)
    run_S5(R_reps, sample_sizes)

    # S4: DGP complexity gradient
    complexities, iv_rmses, chan_rmses, chan_conds = run_S4(R_reps, n=1000)

    # Figures
    if R_reps >= 5:
        make_figures(complexities, iv_rmses, chan_rmses, chan_conds, R_reps)

    print("\n" + "="*70)
    print("Summary of findings:")
    print("  1. IV stacking has OR-channel robustness only (not PS-channel).")
    print("     When correct PS is present but OR models don't span μ(X),")
    print("     IV has persistent bias. Chan converges (has PS channel).")
    print("  2. Adding a constant as instrument + regressor (IV_int) does NOT")
    print("     fix the PS-channel. IV_int bias is similar or worse than IV.")
    print("     The constant-on-constant moment pins E[Y|R=1], not E[Y].")
    print("     PS-channel correction requires 1/π̂ in the REGRESSORS")
    print("     (as Chan does), not just an intercept.")
    print("  3. Chan's enormous condition numbers (10¹⁰+) are harmless when")
    print("     the truth IS in the model span — lstsq handles it fine.")
    print("  4. High cond + misspecification (S5) is the problematic case.")
    print("     Collinearity prevents Chan from using the 1/π̂ terms")
    print("     effectively when the OR fitted values are near-identical")
    print("     but wrong.")
    print("  5. DGP complexity gradient: simpler DGP → higher OR correlation")
    print("     → higher condition number, but LOWER RMSE when models are")
    print("     correct. The condition number only hurts under misspec.")
    print("="*70)


if __name__ == '__main__':
    main()
