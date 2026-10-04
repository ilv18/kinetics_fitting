"""Kinetic models and fits.

Solution-phase hybridization and strand displacement are treated as irreversible bimolecular reactions
A + B -> products with the initial concentrations fixed:

    a0 = b0 = c0 :  A(t) = c0 / (1 + k c0 t)
    a0 < b0      :  A(t) = a0 (b0 - a0) / (b0 exp[(b0 - a0) k t] - a0)

and the fluorescence is a linear combination of the unreacted and reacted states,
F(t) = alpha A + beta (a0 - A) = P A(t)/a0 + Q. The rate constant k is reported as k_on (hybridization) or
k_eff (displacement), in M^-1 s^-1. A reversible model A + B <=> C + D is fitted as a robustness check.

Capture of an incumbent by particle-bound substrate is sigmoidal and is fitted with the five-parameter logistic
f(t) = d + (a - d) / (1 + (t/c)^p)^s. The half-time t1/2 is the time at which the fitted curve reaches (a + d)/2,
and k_app = 1 / (t1/2 [Inc]0) is given for order-of-magnitude comparison with k_on only.
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, curve_fit


# ------------------------------------------------------------------------------------------------ models
def A_irrev(t, k, a0, b0):
    """Unreacted limiting species for irreversible A + B -> products (a0 <= b0)."""
    t = np.maximum(t, 0)
    if np.isclose(a0, b0, rtol=1e-9, atol=0.0):
        return a0 / (1.0 + k * a0 * t)
    x = np.clip((b0 - a0) * k * t, 0, 700)
    return a0 * (b0 - a0) / (b0 * np.exp(x) - a0)


def half_time_irrev(k, a0, b0):
    """Half-time of the limiting species for irreversible A + B -> products."""
    if np.isclose(a0, b0, rtol=1e-9, atol=0.0):
        return 1.0 / (k * a0)
    return np.log(2.0 - a0 / b0) / ((b0 - a0) * k)


def A_rev(t, kf, kr, a0, b0):
    """Unreacted limiting species for reversible A + B <=> C + D, integrated numerically."""
    def rhs(_, a):
        x = a0 - a[0]
        return [-kf * a[0] * (b0 - x) + kr * x * x]
    sol = solve_ivp(rhs, (0, t.max()), [a0], t_eval=t, method="LSODA", rtol=1e-8, atol=a0 * 1e-10)
    return sol.y[0] if sol.success and sol.y.shape[1] == len(t) else np.full_like(t, np.nan)


def fivepl(t, a, d, c, p, s):
    """Five-parameter logistic."""
    return d + (a - d) / (1.0 + (np.maximum(t, 0) / c) ** p) ** s


def r2(y, yf):
    return 1 - np.sum((y - yf) ** 2) / np.sum((y - np.mean(y)) ** 2)


def empirical_half_time(t, y):
    """Model-free t1/2: first time the 15-point running median crosses the midpoint of the start and end plateaus.
    Used as initial estimate for the fits."""
    ys = pd.Series(y).rolling(15, center=True, min_periods=1).median().to_numpy()
    y0, y1 = np.median(ys[:10]), np.median(ys[-20:])
    mid = (y0 + y1) / 2
    cross = np.where(np.sign(ys - mid) != np.sign(y0 - mid))[0]
    return t[cross[0]] if len(cross) else np.nan


# ------------------------------------------------------------------------------------------------ fits
def fit_second_order(tt, yy, a0, b0):
    """Irreversible bimolecular fit with a0, b0 fixed (M); fits log10 k, P and Q. Time tt (s) starts at t0.

    Returns (result dict, parameter vector [log10 k, P, Q])."""
    th_guess = max(empirical_half_time(tt, yy), 5.0)
    k0 = half_time_irrev(1.0, a0, b0) / th_guess
    f = lambda t_, lk, P, Q: P * A_irrev(t_, 10 ** lk, a0, b0) / a0 + Q
    p, cov = curve_fit(f, tt, yy, p0=[np.log10(k0), yy[0] - yy[-1], yy[-1]], maxfev=40000)
    k = 10 ** p[0]
    yf = f(tt, *p)
    th = half_time_irrev(k, a0, b0)
    res = dict(k_M_s=k, k_se_M_s=k * np.log(10) * np.sqrt(cov[0, 0]), P=p[1], Q=p[2], R2=r2(yy, yf),
               rmse=np.sqrt(np.mean((yy - yf) ** 2)), t_half_model_s=th, t_half_empirical_s=empirical_half_time(tt, yy),
               halftime_below_5s_deadtime=th < 5,
               # fraction of the limiting strand reacted at the end of the fit window, from the fitted k
               coverage_fraction_reacted_at_window_end=1 - A_irrev(np.array([tt[-1]]), k, a0, b0)[0] / a0)
    return res, p, yf


def fit_reversible(tt, yy, a0, b0, p_irrev, k_irrev):
    """Reversible A + B <=> C + D fit (log10 kf, log10 kr, P, Q), started from the irreversible fit."""
    fr = lambda t_, lkf, lkr, P, Q: P * A_rev(t_, 10 ** lkf, 10 ** lkr, a0, b0) / a0 + Q
    pr, _ = curve_fit(fr, tt, yy, p0=[p_irrev[0], -4.0, p_irrev[1], p_irrev[2]],
                      bounds=([p_irrev[0] - 3, -12, -np.inf, -np.inf], [p_irrev[0] + 3, 0, np.inf, np.inf]), maxfev=4000)
    yr = fr(tt, *pr)
    kf, kr = 10 ** pr[0], 10 ** pr[1]
    a_end = A_rev(tt, kf, kr, a0, b0)[-1]
    res = dict(rev_kf_M_s=kf, rev_kr_M_s=kr, rev_K_eq=kf / kr, rev_R2=r2(yy, yr), rev_kf_over_irrev_k=kf / k_irrev,
               # reverse rate kr x^2 relative to forward rate kf a b at the end of the fit window
               rev_reverse_to_forward_at_end=kr * (a0 - a_end) ** 2 / (kf * a_end * (b0 - a0 + a_end)))
    return res, yr


def fit_5pl(tt, yy, inc_M):
    """Five-parameter logistic fit of an on-particle capture trace; t1/2 at (a + d)/2 and k_app = 1/(t1/2 [Inc]0)."""
    th_emp = empirical_half_time(tt, yy)
    p0 = [yy[:5].mean(), yy[-20:].mean(), max(th_emp, 10.0), 2.0, 1.0]
    p, _ = curve_fit(fivepl, tt, yy, p0=p0, bounds=([-np.inf, -np.inf, 1e-3, 1e-3, 1e-3], [np.inf, np.inf, 1e5, 50, 50]),
                     maxfev=40000)
    yf = fivepl(tt, *p)
    a, d = p[0], p[1]
    th = brentq(lambda x: fivepl(x, *p) - (a + d) / 2, 1e-6, 1e6)
    res = dict(fivepl_a=p[0], fivepl_d=p[1], fivepl_c=p[2], fivepl_p=p[3], fivepl_s=p[4],
               R2=r2(yy, yf), rmse=np.sqrt(np.mean((yy - yf) ** 2)), t_half_model_s=th, t_half_empirical_s=th_emp,
               # share of the fitted amplitude (a -> d) covered by the data window
               coverage_fraction_of_amplitude_at_window_end=(yf[-1] - a) / (d - a),
               t_half_extrapolated=(yf[-1] - a) / (d - a) < 0.8 or th > tt[-1],
               k_app_M_s=1.0 / (th * inc_M))
    return res, yf


def fit_anchored(tt, yy, a0, b0, F_q, F_top):
    """Second-order fit with the amplitude fixed: F(t) = F_q + (F_top - F_q) (1 - A(t)/a0), only k fitted
    (irreversible) or kf, kr fitted (reversible). F_q = quenched duplex before injection, F_top = free incumbent."""
    amp = F_top - F_q
    fi = lambda t_, lk: F_q + amp * (1 - A_irrev(t_, 10 ** lk, a0, b0) / a0)
    pi, _ = curve_fit(fi, tt, yy, p0=[3.0], maxfev=20000)
    fr = lambda t_, lkf, lkr: F_q + amp * (1 - A_rev(t_, 10 ** lkf, 10 ** lkr, a0, b0) / a0)
    pr, _ = curve_fit(fr, tt, yy, p0=[pi[0], pi[0] - 8.0], bounds=([0, -6], [9, 9]), maxfev=4000)
    kf, kr = 10 ** pr[0], 10 ** pr[1]
    K = kf / kr
    # equilibrium fraction displaced for the reversible model: x^2 / ((a0 - x)(b0 - x)) = K
    xs = np.linspace(0, a0, 200001)[1:-1]
    x_eq = xs[np.argmin(np.abs(xs ** 2 - K * (a0 - xs) * (b0 - xs)))]
    return dict(anchored_irrev_k_M_s=10 ** pi[0], anchored_irrev_R2=r2(yy, fi(tt, *pi)),
                anchored_rev_kf_M_s=kf, anchored_rev_kr_M_s=kr, anchored_rev_K=K,
                anchored_rev_equilibrium_fraction_displaced=x_eq / a0, anchored_rev_R2=r2(yy, fr(tt, *pr)),
                fraction_displaced_at_window_end=(yy[-20:].mean() - F_q) / amp)


def fit_second_order_variant(tt, yy, a0, b0, k0, offset):
    """Second-order refit used for the start-time robustness check: bounded log10 k, optionally with a fitted
    start-time offset d (F = P A(t - d)/a0 + Q, d in [-30, 30] s). Returns (k, d, fraction reacted at window end)."""
    if offset:
        f = lambda t_, lk, P, Q, d: P * A_irrev(t_ - d, 10 ** lk, a0, b0) / a0 + Q
        p0, lb, ub = [np.log10(k0), yy[0] - yy[-1], yy[-1], 0.0], [0, -np.inf, -np.inf, -30], [9, np.inf, np.inf, 30]
    else:
        f = lambda t_, lk, P, Q: P * A_irrev(t_, 10 ** lk, a0, b0) / a0 + Q
        p0, lb, ub = [np.log10(k0), yy[0] - yy[-1], yy[-1]], [0, -np.inf, -np.inf], [9, np.inf, np.inf]
    p, _ = curve_fit(f, tt, yy, p0=p0, bounds=(lb, ub), maxfev=40000)
    k = 10 ** p[0]
    d = p[3] if offset else 0.0
    frac_end = 1 - A_irrev(np.array([tt[-1] - d]), k, a0, b0)[0] / a0
    return k, d, frac_end
