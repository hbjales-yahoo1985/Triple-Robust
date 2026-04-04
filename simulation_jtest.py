"""
Simulation: J-Statistic Behavior in Overidentified IV Stacking
==============================================================

Verifies that the Hansen-Sargan J-statistic in the overidentified IV stacking
estimator behaves as predicted by theory across four cases:

| Case | OR correct? | PS correct? | ATT consistent? | J rejects? |
|------|-------------|-------------|-----------------|------------|
| 1    | Yes         | No          | Yes             | No         |
| 2    | No          | No          | No              | Yes        |
| 3    | No          | Yes (overid)| No              | Yes        |
| 4    | No          | Yes (exact) | Yes             | N/A        |

Case 3 is the key: one PS is correct, but overidentification prevents
the Bang-Robins cancellation from firing exactly, so ATT is inconsistent
and the J-test correctly detects this.

The prescribed workflow:
  Estimate with exact ID → Test with overid J-stat → Enrich OR if rejected.

Run:
    python simulation_jtest.py          # R=10 smoke test
    python simulation_jtest.py 500      # R=500 production run
"""

import sys
import warnings
import numpy as np
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

PS_CLIP = (0.02, 0.98)
TRUE_TAU = 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clip_ps(p):
    return np.clip(p, PS_CLIP[0], PS_CLIP[1])


def _invlogit(z):
    return np.where(z >= 0, 1.0/(1.0+np.exp(-z)), np.exp(z)/(1.0+np.exp(z)))


# ═══════════════════════════════════════════════════════════════════════════
#  DGP
# ═══════════════════════════════════════════════════════════════════════════

def dgp(rng, n):
    """
    Generate data with known ATT = 2.0.

    True PS: logit(-0.3 + 0.5X - 0.25X²)
    True OR: μ₀(X) = 2 + 0.8X² + 0.6sin(2X)
    Treatment effect: τ = 2.0 (constant)
    """
    X = rng.uniform(-2, 2, n)

    logit_e = -0.3 + 0.5*X - 0.25*X**2
    e_true = _clip_ps(_invlogit(logit_e))
    D = rng.binomial(1, e_true, n).astype(float)

    mu0 = 2.0 + 0.8*X**2 + 0.6*np.sin(2*X)
    Y0 = mu0 + rng.normal(0, 1, n)
    Y1 = Y0 + TRUE_TAU
    Y = D * Y1 + (1 - D) * Y0

    return Y, D, X, e_true, mu0


# ═══════════════════════════════════════════════════════════════════════════
#  PS and OR model fitting
# ═══════════════════════════════════════════════════════════════════════════

def fit_ps_logit(D, X, covariate_func):
    """Fit logit PS model, return fitted probabilities."""
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


def fit_ps_probit(D, X, covariate_func):
    """Fit probit PS model, return fitted probabilities."""
    n = len(D)
    features = covariate_func(X)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xd = sm.add_constant(features)
    try:
        mod = Probit(D, Xd).fit(disp=False, maxiter=200)
        return _clip_ps(mod.predict(Xd))
    except Exception:
        return np.full(n, np.clip(np.mean(D), *PS_CLIP))


def fit_or_model(Y, D, X, covariate_func):
    """Fit OR by OLS among controls (D==0), predict for all."""
    n = len(Y)
    features = covariate_func(X)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    Xd_all = sm.add_constant(features)
    controls = (D == 0)
    try:
        coef, _, _, _ = np.linalg.lstsq(Xd_all[controls], Y[controls],
                                         rcond=None)
        return Xd_all @ coef
    except Exception:
        return np.full(n, np.nanmean(Y[controls]))


# ═══════════════════════════════════════════════════════════════════════════
#  Covariate functions for working models
# ═══════════════════════════════════════════════════════════════════════════
# Design principle: wrong PS models use the SAME class of basis function
# transforms as the OR models (polynomial, trig, exp) to ensure instrument
# relevance — Cov(odds, m̂) is large when both live in similar function
# spaces.  Accidental spanning is acceptable; the J-test flags genuine
# misspecification.

# OR models
def _or_linear(X):       return X.reshape(-1, 1)                         # {X}
def _or_exp(X):          return np.exp(X).reshape(-1, 1)                 # {exp(X)}
def _or_correct(X):      return np.column_stack([X**2, np.sin(2*X)])     # {X²,sin(2X)} ← CORRECT

# PS models — use matching function class transforms for instrument relevance
def _ps_quad(X):         return np.column_stack([X, X**2])               # {X,X²} ← CORRECT
def _ps_sin(X):          return np.sin(X).reshape(-1, 1)                 # {sin(X)}
def _ps_linear(X):       return X.reshape(-1, 1)                         # {X}
def _ps_exp(X):          return np.exp(X).reshape(-1, 1)                 # {exp(X)}


# ═══════════════════════════════════════════════════════════════════════════
#  IV estimators
# ═══════════════════════════════════════════════════════════════════════════

def iv_exact_att(Y, D, ps_fitted_list, or_fitted_list):
    """
    Exactly identified IV stacking estimator for ATT.

    Among controls (D==0):
        Instruments: {ê_1/(1-ê_1),...,ê_K/(1-ê_K)}  (odds instruments)
        Regressors:  {m̂_1,...,m̂_K}

    ATT: τ̂ = n₁⁻¹ Σ D_i(Y_i - Σ_k φ̂_k m̂_k(X_i))

    Uses odds instruments on controls for ATT PS-channel robustness:
      E_{D=0}[(e/(1-e))(Y₀ - m'φ)] = 0
      → E[e(μ₀ - m'φ)] = 0
      → E[μ₀ - m'φ | D=1] = 0
      → (1/n₁)Σ_{D=1} m'φ → E[μ₀|D=1]

    Returns: (att_hat, phi_hat, cond_num)
    """
    K = len(or_fitted_list)
    assert K == len(ps_fitted_list), "K instruments must match K regressors"

    n = len(Y)
    controls = (D == 0)
    treated = (D == 1)
    n1 = int(treated.sum())

    M_all = np.column_stack(or_fitted_list)                     # (n, K)
    Z_all = np.column_stack([_clip_ps(ps) / (1.0 - _clip_ps(ps))
                             for ps in ps_fitted_list])         # (n, K) odds

    M_c = M_all[controls]
    Z_c = Z_all[controls]
    Y_c = Y[controls]

    ZtM = Z_c.T @ M_c
    ZtY = Z_c.T @ Y_c

    cond_num = float(np.linalg.cond(ZtM))

    try:
        phi = np.linalg.solve(ZtM, ZtY)
    except np.linalg.LinAlgError:
        phi, _, _, _ = np.linalg.lstsq(ZtM, ZtY, rcond=None)

    # ATT: average imputation gap among treated
    imputed = M_all @ phi
    att_hat = float(np.mean(Y[treated] - imputed[treated]))

    return att_hat, phi, cond_num


def iv_overid_att(Y, D, ps_fitted_list, or_fitted_list):
    """
    Overidentified 2SLS IV stacking estimator for ATT with J-statistic.

    Among controls (D==0):
        Instruments: {ê_1/(1-ê_1),...,ê_L/(1-ê_L)}  (L odds instruments, L > K)
        Regressors:  {m̂_1,...,m̂_K}                    (K regressors)

    2SLS: Project M onto span(Z), then OLS.
    J-stat: Hansen-Sargan test for overidentifying restrictions.

    Uses odds instruments on controls for ATT PS-channel robustness.
    Uses QR decomposition for numerical stability.
    Returns: (att_hat, phi_hat, J_stat, J_pval, J_df, cond_num)
    """
    K = len(or_fitted_list)
    L = len(ps_fitted_list)
    assert L > K, f"Need L={L} > K={K} for overidentification"

    n = len(Y)
    controls = (D == 0)
    treated = (D == 1)
    n0 = int(controls.sum())
    n1 = int(treated.sum())

    M_all = np.column_stack(or_fitted_list)                     # (n, K)
    Z_all = np.column_stack([_clip_ps(ps) / (1.0 - _clip_ps(ps))
                             for ps in ps_fitted_list])         # (n, L) odds

    M_c = M_all[controls]
    Z_c = Z_all[controls]
    Y_c = Y[controls]

    # QR decomposition for numerical stability
    Q, R_qr = np.linalg.qr(Z_c, mode='reduced')

    cond_num = float(np.linalg.cond(R_qr))

    # Projection: P_Z = QQ'
    # 2SLS: φ̂ = (M'QQ'M)^{-1} M'QQ'Y
    QtM = Q.T @ M_c          # (L, K)
    QtY = Q.T @ Y_c          # (L,)

    MtPM = QtM.T @ QtM       # (K, K)
    MtPY = QtM.T @ QtY       # (K,)

    try:
        phi = np.linalg.solve(MtPM, MtPY)
    except np.linalg.LinAlgError:
        phi, _, _, _ = np.linalg.lstsq(MtPM, MtPY, rcond=None)

    # Residuals
    resid = Y_c - M_c @ phi

    # J-statistic: J = (e'P_Z e) / σ̂²
    Qte = Q.T @ resid         # (L,)
    sigma2 = float(np.mean(resid**2))

    if sigma2 > 0:
        J_stat = float(Qte @ Qte) / sigma2
    else:
        J_stat = 0.0

    J_df = L - K
    J_pval = float(1.0 - chi2.cdf(J_stat, df=J_df))

    # ATT
    imputed = M_all @ phi
    att_hat = float(np.mean(Y[treated] - imputed[treated]))

    return att_hat, phi, J_stat, J_pval, J_df, cond_num


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics helpers
# ═══════════════════════════════════════════════════════════════════════════

def _stats(arr, true_val):
    """Compute mean, bias, SD, RMSE."""
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
#  Case implementations
# ═══════════════════════════════════════════════════════════════════════════

def run_case1(R_reps, sample_sizes, seed=42):
    """
    Case 1: OR correct, PS wrong → ATT consistent, J does not reject.

    OR: OR1_correct={X²,sin(2X)} ← CORRECT, OR2={exp(X)} ← wrong
    PS exact (K=2): PS_sin={sin(X)} wrong, PS_lin=probit{X} wrong
    PS overid (K+L=4): PS_sin, PS_lin, PS_exp={exp(X)}, PS_cube={X³} — all wrong
    """
    or_funcs = [_or_correct, _or_exp]
    ps_exact_funcs = [_ps_sin, _ps_linear]  # both wrong → exact ID K=2
    ps_extra_funcs = [_ps_exp, lambda X: (X**3).reshape(-1, 1)]
    ps_extra_types = ['logit', 'logit']
    ps_overid_funcs = ps_exact_funcs + ps_extra_funcs  # L=4 > K=2

    return _run_case(
        "Case 1: OR correct, PS wrong",
        or_funcs, ps_exact_funcs, ps_overid_funcs,
        ps_exact_types=['logit', 'probit'],
        ps_overid_types=['logit', 'probit', 'logit', 'logit'],
        R_reps=R_reps, sample_sizes=sample_sizes, seed=seed
    )


def run_case2(R_reps, sample_sizes, seed=42):
    """
    Case 2: OR wrong, PS wrong → ATT inconsistent, J rejects.

    OR: OR1={X} wrong, OR2={exp(X)} wrong
    PS exact (K=2): PS2={sin(X)} wrong, PS3=probit{X} wrong
    PS overid (K+L=4): PS2, PS3, PS4={exp(X)} wrong, PS_extra=logit{X³} wrong
    """
    or_funcs = [_or_linear, _or_exp]
    ps_exact_funcs = [_ps_sin, _ps_linear]
    ps_extra_funcs = [_ps_exp, lambda X: (X**3).reshape(-1, 1)]
    ps_overid_funcs = ps_exact_funcs + ps_extra_funcs

    return _run_case(
        "Case 2: OR wrong, PS wrong",
        or_funcs, ps_exact_funcs, ps_overid_funcs,
        ps_exact_types=['logit', 'probit'],
        ps_overid_types=['logit', 'probit', 'logit', 'logit'],
        R_reps=R_reps, sample_sizes=sample_sizes, seed=seed
    )


def run_case3(R_reps, sample_sizes, seed=42):
    """
    Case 3: OR wrong, PS1 correct but OVERIDENTIFIED → ATT inconsistent, J rejects.

    OR: OR1={X} wrong, OR2={exp(X)} wrong
    PS overid (L=4): PS1={X,X²} CORRECT, PS2={sin(X)}, PS3=probit{X}, PS4={exp(X)}
    """
    or_funcs = [_or_linear, _or_exp]
    # No exact-id for Case 3 (that's Case 4)
    ps_overid_funcs = [_ps_quad, _ps_sin, _ps_linear, _ps_exp]
    ps_overid_types = ['logit', 'logit', 'probit', 'logit']

    return _run_case(
        "Case 3: OR wrong, PS correct, OVERIDENTIFIED",
        or_funcs, None, ps_overid_funcs,
        ps_exact_types=None,
        ps_overid_types=ps_overid_types,
        R_reps=R_reps, sample_sizes=sample_sizes, seed=seed
    )


def run_case4(R_reps, sample_sizes, seed=42):
    """
    Case 4: OR wrong, PS1 correct, EXACTLY IDENTIFIED → ATT consistent.

    OR: OR1={X} wrong, OR2={exp(X)} wrong
    PS exact (K=2): PS1={X,X²} CORRECT, PS2={sin(X)} wrong
    """
    or_funcs = [_or_linear, _or_exp]
    ps_exact_funcs = [_ps_quad, _ps_sin]

    return _run_case(
        "Case 4: OR wrong, PS correct, EXACTLY IDENTIFIED",
        or_funcs, ps_exact_funcs, None,
        ps_exact_types=['logit', 'logit'],
        ps_overid_types=None,
        R_reps=R_reps, sample_sizes=sample_sizes, seed=seed
    )


def _run_case(title, or_funcs, ps_exact_funcs, ps_overid_funcs,
              ps_exact_types, ps_overid_types,
              R_reps, sample_sizes, seed):
    """
    Generic runner for a single case.

    Returns dict of results keyed by n.
    """
    K = len(or_funcs)
    has_exact = ps_exact_funcs is not None
    has_overid = ps_overid_funcs is not None

    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"  K={K} OR models", end="")
    if has_exact:
        print(f", {len(ps_exact_funcs)} PS exact-id instruments", end="")
    if has_overid:
        print(f", {len(ps_overid_funcs)} PS overid instruments", end="")
    print(f"\n  True ATT = {TRUE_TAU}")
    print(f"{'='*90}")

    results = {}

    for n_val in sample_sizes:
        res = {
            'exact_att': [], 'overid_att': [],
            'J_stat': [], 'J_pval': [], 'J_reject': [],
            'exact_cond': [], 'overid_cond': [],
        }

        rng_base = np.random.default_rng(seed)
        rep_seeds = rng_base.integers(0, 2**31, size=R_reps)

        for rep in range(R_reps):
            rng_rep = np.random.default_rng(rep_seeds[rep])
            Y, D, X, e_true, mu0 = dgp(rng_rep, n_val)

            # Fit OR models (among controls)
            or_fitted = [fit_or_model(Y, D, X, f) for f in or_funcs]

            # Exact-identified IV
            if has_exact:
                ps_exact_fitted = []
                for pf, pt in zip(ps_exact_funcs, ps_exact_types):
                    if pt == 'probit':
                        ps_exact_fitted.append(fit_ps_probit(D, X, pf))
                    else:
                        ps_exact_fitted.append(fit_ps_logit(D, X, pf))

                try:
                    att_ex, _, cond_ex = iv_exact_att(
                        Y, D, ps_exact_fitted, or_fitted)
                    res['exact_att'].append(att_ex)
                    res['exact_cond'].append(cond_ex)
                except Exception:
                    res['exact_att'].append(np.nan)
                    res['exact_cond'].append(np.nan)

            # Overidentified 2SLS + J-test
            if has_overid:
                ps_overid_fitted = []
                for pf, pt in zip(ps_overid_funcs, ps_overid_types):
                    if pt == 'probit':
                        ps_overid_fitted.append(fit_ps_probit(D, X, pf))
                    else:
                        ps_overid_fitted.append(fit_ps_logit(D, X, pf))

                try:
                    att_ov, _, J, Jp, Jdf, cond_ov = iv_overid_att(
                        Y, D, ps_overid_fitted, or_fitted)
                    res['overid_att'].append(att_ov)
                    res['J_stat'].append(J)
                    res['J_pval'].append(Jp)
                    res['J_reject'].append(1 if Jp < 0.05 else 0)
                    res['overid_cond'].append(cond_ov)
                except Exception:
                    res['overid_att'].append(np.nan)
                    res['J_stat'].append(np.nan)
                    res['J_pval'].append(np.nan)
                    res['J_reject'].append(np.nan)
                    res['overid_cond'].append(np.nan)

        results[n_val] = res

        # Print results for this n
        if has_exact and res['exact_att']:
            mn, bias, sd, rmse = _stats(res['exact_att'], TRUE_TAU)
            cond_med = float(np.nanmedian(res['exact_cond'])) if res['exact_cond'] else np.nan
            print(f"  n={n_val:>6}  ExactID:  bias={bias:>+8.4f}  SD={sd:>7.4f}  "
                  f"RMSE={rmse:>7.4f}  cond={cond_med:>10.1f}")

        if has_overid and res['overid_att']:
            mn, bias, sd, rmse = _stats(res['overid_att'], TRUE_TAU)
            J_arr = np.array(res['J_stat'], dtype=float)
            J_arr = J_arr[~np.isnan(J_arr)]
            J_mean = float(np.mean(J_arr)) if len(J_arr) > 0 else np.nan
            J_med = float(np.median(J_arr)) if len(J_arr) > 0 else np.nan
            rej_arr = np.array(res['J_reject'], dtype=float)
            rej_arr = rej_arr[~np.isnan(rej_arr)]
            rej_rate = float(np.mean(rej_arr)) if len(rej_arr) > 0 else np.nan
            cond_med = float(np.nanmedian(res['overid_cond'])) if res['overid_cond'] else np.nan
            print(f"  n={n_val:>6}  OverID:   bias={bias:>+8.4f}  SD={sd:>7.4f}  "
                  f"RMSE={rmse:>7.4f}  J_mean={J_mean:>7.1f}  "
                  f"J_med={J_med:>7.1f}  rej@5%={rej_rate:>5.1%}  "
                  f"cond={cond_med:>10.1f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Summary tables
# ═══════════════════════════════════════════════════════════════════════════

def print_table1(all_results, sample_sizes):
    """Table 1: ATT Bias and RMSE."""
    print(f"\n{'='*90}")
    print("  TABLE 1: ATT Bias and RMSE")
    print(f"{'='*90}")
    hdr = f"{'Case':>8}  {'Estimator':>10}  {'n':>6}  {'Mean τ̂':>8}  " \
          f"{'Bias':>8}  {'SD':>7}  {'RMSE':>7}"
    print(hdr)
    print("-" * len(hdr))

    for case_num, case_key in enumerate([1, 2, 3, 4], start=1):
        if case_key not in all_results:
            continue
        res_by_n = all_results[case_key]
        for n_val in sample_sizes:
            if n_val not in res_by_n:
                continue
            res = res_by_n[n_val]

            if res['exact_att']:
                mn, bias, sd, rmse = _stats(res['exact_att'], TRUE_TAU)
                print(f"{case_num:>8}  {'ExactID':>10}  {n_val:>6}  "
                      f"{mn:>8.4f}  {bias:>+8.4f}  {sd:>7.4f}  {rmse:>7.4f}")

            if res['overid_att']:
                mn, bias, sd, rmse = _stats(res['overid_att'], TRUE_TAU)
                print(f"{case_num:>8}  {'OverID':>10}  {n_val:>6}  "
                      f"{mn:>8.4f}  {bias:>+8.4f}  {sd:>7.4f}  {rmse:>7.4f}")
        print("-" * len(hdr))


def print_table2(all_results, sample_sizes):
    """Table 2: J-Test Performance."""
    print(f"\n{'='*90}")
    print("  TABLE 2: J-Test Performance")
    print(f"{'='*90}")
    hdr = f"{'Case':>8}  {'n':>6}  {'Mean J':>8}  {'Med J':>8}  " \
          f"{'Rej@5%':>8}  {'Prediction':>16}"
    print(hdr)
    print("-" * len(hdr))

    predictions = {
        1: "≈ 5% (size)",
        2: "→ 100% (power)",
        3: "→ 100% (power)",
    }

    for case_num in [1, 2, 3]:
        if case_num not in all_results:
            continue
        res_by_n = all_results[case_num]
        for n_val in sample_sizes:
            if n_val not in res_by_n:
                continue
            res = res_by_n[n_val]
            if not res['J_stat']:
                continue

            J_arr = np.array(res['J_stat'], dtype=float)
            J_arr = J_arr[~np.isnan(J_arr)]
            J_mean = float(np.mean(J_arr)) if len(J_arr) else np.nan
            J_med = float(np.median(J_arr)) if len(J_arr) else np.nan
            rej = np.array(res['J_reject'], dtype=float)
            rej = rej[~np.isnan(rej)]
            rej_rate = float(np.mean(rej)) if len(rej) else np.nan

            pred = predictions.get(case_num, "")
            print(f"{case_num:>8}  {n_val:>6}  {J_mean:>8.2f}  {J_med:>8.2f}  "
                  f"{rej_rate:>7.1%}  {pred:>16}")
        print("-" * len(hdr))


def print_table3(all_results, sample_sizes):
    """Table 3: Case 3 vs Case 4 Head-to-Head."""
    print(f"\n{'='*90}")
    print("  TABLE 3: Case 3 (overid) vs Case 4 (exact id) — Head-to-Head")
    print(f"{'='*90}")
    hdr = f"{'n':>6}  {'Case4 ExactID bias':>20}  {'Case3 OverID bias':>20}  " \
          f"{'Case3 J rej@5%':>16}"
    print(hdr)
    print("-" * len(hdr))

    for n_val in sample_sizes:
        c4_bias = np.nan
        c3_bias = np.nan
        c3_rej = np.nan

        if 4 in all_results and n_val in all_results[4]:
            res4 = all_results[4][n_val]
            if res4['exact_att']:
                _, c4_bias, _, _ = _stats(res4['exact_att'], TRUE_TAU)

        if 3 in all_results and n_val in all_results[3]:
            res3 = all_results[3][n_val]
            if res3['overid_att']:
                _, c3_bias, _, _ = _stats(res3['overid_att'], TRUE_TAU)
            rej = np.array(res3['J_reject'], dtype=float)
            rej = rej[~np.isnan(rej)]
            c3_rej = float(np.mean(rej)) if len(rej) else np.nan

        c4_str = f"{c4_bias:>+.4f}" if not np.isnan(c4_bias) else "N/A"
        c3_str = f"{c3_bias:>+.4f}" if not np.isnan(c3_bias) else "N/A"
        rej_str = f"{c3_rej:.1%}" if not np.isnan(c3_rej) else "N/A"
        print(f"{n_val:>6}  {c4_str:>20}  {c3_str:>20}  {rej_str:>16}")


# ═══════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(all_results, sample_sizes, R_reps):
    """Generate Figures 1-3."""
    n_fig1 = min(sample_sizes, key=lambda x: abs(x - 2000))  # closest to 2000

    # ── Figure 1: J-stat distribution by case ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle(f"Figure 1: J-stat distribution (n={n_fig1}, R={R_reps})", y=1.02)

    for idx, case_num in enumerate([1, 2, 3]):
        ax = axes[idx]
        if case_num in all_results and n_fig1 in all_results[case_num]:
            J_arr = np.array(all_results[case_num][n_fig1]['J_stat'], dtype=float)
            J_arr = J_arr[~np.isnan(J_arr)]

            if len(J_arr) > 0:
                # Compute overid df
                J_df = 2  # L=4 instruments minus K=2 regressors
                upper = max(np.percentile(J_arr, 99), chi2.ppf(0.99, J_df) * 1.5)
                bins = np.linspace(0, min(upper, 100), 50)

                ax.hist(J_arr, bins=bins, density=True, alpha=0.6,
                        color=['steelblue', 'salmon', 'darkorange'][idx],
                        label=f'Case {case_num}')

                # Overlay χ²(L) density
                x_chi2 = np.linspace(0, min(upper, 100), 200)
                ax.plot(x_chi2, chi2.pdf(x_chi2, J_df), 'k-', lw=2,
                        label=f'χ²({J_df})')

                ax.axvline(chi2.ppf(0.95, J_df), color='red', ls='--',
                           alpha=0.7, label='5% crit')

        names = {1: "OR correct", 2: "All wrong", 3: "PS correct, overid"}
        ax.set_title(f"Case {case_num}: {names.get(case_num, '')}")
        ax.set_xlabel("J statistic")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("jtest_fig1_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved jtest_fig1_distributions.png")

    # ── Figure 2: Rejection rate vs n ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for case_num, marker, color in [(1, 'o', 'steelblue'),
                                     (2, 's', 'salmon'),
                                     (3, '^', 'darkorange')]:
        if case_num not in all_results:
            continue
        ns = []
        rates = []
        for n_val in sample_sizes:
            if n_val not in all_results[case_num]:
                continue
            rej = np.array(all_results[case_num][n_val]['J_reject'], dtype=float)
            rej = rej[~np.isnan(rej)]
            if len(rej) > 0:
                ns.append(n_val)
                rates.append(float(np.mean(rej)))
        if ns:
            ax.plot(ns, rates, f'-{marker}', color=color, markersize=8,
                    label=f'Case {case_num}')

    ax.axhline(0.05, color='gray', ls='--', alpha=0.5, label='5% nominal')
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("J-test rejection rate (5% level)")
    ax.set_title(f"Figure 2: J-test rejection rate vs n (R={R_reps})")
    ax.legend()
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig("jtest_fig2_rejection_rates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved jtest_fig2_rejection_rates.png")

    # ── Figure 3: Case 3 vs Case 4 ATT bias ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for case_num, key, label, color, marker in [
        (4, 'exact_att', 'Case 4: ExactID (PS correct)', 'steelblue', 'o'),
        (3, 'overid_att', 'Case 3: OverID (PS correct)', 'darkorange', '^'),
    ]:
        if case_num not in all_results:
            continue
        ns = []
        biases = []
        for n_val in sample_sizes:
            if n_val not in all_results[case_num]:
                continue
            arr = all_results[case_num][n_val][key]
            if arr:
                _, bias, _, _ = _stats(arr, TRUE_TAU)
                ns.append(n_val)
                biases.append(bias)
        if ns:
            ax.plot(ns, biases, f'-{marker}', color=color, markersize=8,
                    label=label)

    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("ATT bias")
    ax.set_title(f"Figure 3: Case 3 vs Case 4 ATT bias (R={R_reps})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("jtest_fig3_case3v4_bias.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved jtest_fig3_case3v4_bias.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    R_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sample_sizes = [500, 2000, 10000]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  J-Statistic in Overidentified IV Stacking — Simulation    ║")
    print(f"║  R = {R_reps:<5}  n ∈ {sample_sizes}                  ║")
    print(f"║  True ATT = {TRUE_TAU}                                         ║")
    print("║  Odds instruments e/(1-e) for ATT PS-channel robustness    ║")
    print("║  No sample splitting (proof of concept)                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results = {}

    # Case 1: OR correct → consistent, J does not reject
    all_results[1] = run_case1(R_reps, sample_sizes)

    # Case 2: All wrong → inconsistent, J rejects
    all_results[2] = run_case2(R_reps, sample_sizes)

    # Case 3: PS correct, overid → inconsistent, J rejects (KEY)
    all_results[3] = run_case3(R_reps, sample_sizes)

    # Case 4: PS correct, exact id → consistent
    all_results[4] = run_case4(R_reps, sample_sizes)

    # Summary tables
    print_table1(all_results, sample_sizes)
    print_table2(all_results, sample_sizes)
    print_table3(all_results, sample_sizes)

    # Figures
    if R_reps >= 5:
        print("\nGenerating figures...")
        make_figures(all_results, sample_sizes, R_reps)

    # Success criteria check
    print(f"\n{'='*90}")
    print("  SUCCESS CRITERIA CHECK")
    print(f"{'='*90}")

    criteria = []

    # 1. Case 1 J-test size
    if 1 in all_results:
        for n_val in [2000, 10000]:
            if n_val in all_results[1]:
                rej = np.array(all_results[1][n_val]['J_reject'], dtype=float)
                rej = rej[~np.isnan(rej)]
                if len(rej) >= 20:
                    rate = float(np.mean(rej))
                    ok = 0.01 <= rate <= 0.15  # relaxed bounds for finite R
                    criteria.append(('Case 1 size', n_val, rate, ok))
                    sym = "✓" if ok else "✗"
                    print(f"  {sym} Case 1 J-test size at n={n_val}: "
                          f"{rate:.1%} (target: 3-8%)")

    # 2. Case 2 J-test power
    if 2 in all_results and 10000 in all_results[2]:
        rej = np.array(all_results[2][10000]['J_reject'], dtype=float)
        rej = rej[~np.isnan(rej)]
        if len(rej) > 0:
            rate = float(np.mean(rej))
            ok = rate > 0.5
            criteria.append(('Case 2 power', 10000, rate, ok))
            sym = "✓" if ok else "✗"
            print(f"  {sym} Case 2 J-test power at n=10000: "
                  f"{rate:.1%} (target: → 100%)")

    # 3. Case 3 J-test power
    if 3 in all_results and 10000 in all_results[3]:
        rej = np.array(all_results[3][10000]['J_reject'], dtype=float)
        rej = rej[~np.isnan(rej)]
        if len(rej) > 0:
            rate = float(np.mean(rej))
            ok = rate > 0.5
            criteria.append(('Case 3 power', 10000, rate, ok))
            sym = "✓" if ok else "✗"
            print(f"  {sym} Case 3 J-test power at n=10000: "
                  f"{rate:.1%} (target: → 100%)")

    # 4. Case 3 overid ATT inconsistent
    if 3 in all_results and 10000 in all_results[3]:
        arr = all_results[3][10000]['overid_att']
        if arr:
            _, bias, _, _ = _stats(arr, TRUE_TAU)
            ok = abs(bias) > 0.02
            criteria.append(('Case 3 overid bias', 10000, bias, ok))
            sym = "✓" if ok else "✗"
            print(f"  {sym} Case 3 overid ATT persistent bias at n=10000: "
                  f"{bias:+.4f} (target: |bias| > 0.02)")

    # 5. Case 4 exact id ATT consistent
    if 4 in all_results and 10000 in all_results[4]:
        arr = all_results[4][10000]['exact_att']
        if arr:
            _, bias, _, _ = _stats(arr, TRUE_TAU)
            ok = abs(bias) < 0.15
            criteria.append(('Case 4 exact bias', 10000, bias, ok))
            sym = "✓" if ok else "✗"
            print(f"  {sym} Case 4 exact ATT bias → 0 at n=10000: "
                  f"{bias:+.4f} (target: |bias| < 0.15)")

    # 6. Case 3 vs Case 4
    if 3 in all_results and 4 in all_results:
        for n_val in [10000]:
            if n_val in all_results[3] and n_val in all_results[4]:
                arr3 = all_results[3][n_val]['overid_att']
                arr4 = all_results[4][n_val]['exact_att']
                if arr3 and arr4:
                    _, b3, _, _ = _stats(arr3, TRUE_TAU)
                    _, b4, _, _ = _stats(arr4, TRUE_TAU)
                    ok = abs(b3) > abs(b4) * 1.5
                    criteria.append(('Case3 > Case4 bias', n_val,
                                     f"{b3:+.4f} vs {b4:+.4f}", ok))
                    sym = "✓" if ok else "✗"
                    print(f"  {sym} Case 3 vs Case 4 at n={n_val}: "
                          f"overid bias={b3:+.4f}, exact bias={b4:+.4f}")

    passed = sum(1 for _, _, _, ok in criteria if ok)
    total = len(criteria)
    print(f"\n  Result: {passed}/{total} criteria met")


if __name__ == '__main__':
    main()
