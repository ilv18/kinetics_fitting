"""Fit every trace, derive the reported values (Table S4) and the robustness checks."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import manifest as mf
from .data import all_injections, injection_t0, load_trace, pre_injection_level
from .fitting import (A_irrev, A_rev, empirical_half_time, fit_5pl, fit_anchored, fit_reversible, fit_second_order,
                      fit_second_order_variant, fivepl)
from scipy.optimize import brentq, curve_fit


def _trace_path(data_dir, m):
    return Path(data_dir) / m["folder"] / m["file"]


def fit_all(data_dir, diagnostics_dir=None):
    """Fit every trace in the manifest. Returns a DataFrame with one row per trace and channel.

    If ``diagnostics_dir`` is given, a data/fit/residual plot is saved for every fit.
    """
    rows = []
    for m in mf.traces():
        for ch_name, col in m["channels"]:
            t, y = load_trace(_trace_path(data_dir, m), col)
            t_spike, t0 = injection_t0(t, y, m["win_start"])
            start = max(m["win_start"], t0 + m["skip_after_t0_s"])
            sel = (t >= start) & (t < m["win_end"])
            tt, yy = t[sel] - t0, y[sel]
            cuv_nM, inj_nM = m["cuvette_stock_uM"] * 1000 * mf.DIL, m["injected_stock_uM"] * 1000 * mf.DIL
            # a second injection inside the window would invalidate a single-step fit: flag it
            dy = np.abs(np.diff(y[sel]))
            later_spike = bool(len(dy) > 20 and dy[10:].max() > 8 * np.median(dy) + 10)
            row = dict(system=m["system"], panel=m["panel"], file=m["file"], rep=m["rep"], channel=ch_name,
                       model=m["model"], cuvette_strand=m["cuvette_strand"], cuvette_final_nM=cuv_nM,
                       injected_strand=m["injected_strand"], injected_final_nM=inj_nM,
                       t_spike_s=t_spike, t0_s=t0, window_start_s=m["win_start"], window_end_s=m["win_end"],
                       fit_start_s=start, n_points=int(sel.sum()), possible_second_injection_in_window=later_spike,
                       note=m["note"])
            yf = yr = None
            if m["model"] == "2nd":
                a0, b0 = sorted([cuv_nM * 1e-9, inj_nM * 1e-9])
                row.update(a0_limiting_nM=a0 * 1e9, b0_excess_nM=b0 * 1e9)
                try:
                    res, p, yf = fit_second_order(tt, yy, a0, b0)
                    row.update(res)
                    try:
                        rres, yr = fit_reversible(tt, yy, a0, b0, p, res["k_M_s"])
                        row.update(rres)
                    except Exception as e:
                        row.update(rev_fit_error=str(e))
                except Exception as e:
                    row.update(fit_error=str(e))
            else:
                try:
                    res, yf = fit_5pl(tt, yy, inj_nM * 1e-9)
                    row.update(res)
                    row.update(spn_final_pM=cuv_nM * 1000, inc_final_nM=inj_nM)
                    # keep k_app as the last column, as in the published tables
                    row["k_app_M_s"] = row.pop("k_app_M_s")
                except Exception as e:
                    row.update(fit_error=str(e))
            rows.append(row)
            if diagnostics_dir is not None:
                from .plotting import diagnostic_plot
                diagnostic_plot(m, row, t, y, start, t0, tt, yy, yf, yr, cuv_nM, inj_nM, Path(diagnostics_dir))
    return pd.DataFrame(rows)


def anchored_invR21(data_dir):
    """InvR21 displacement with the amplitude fixed by independently measured levels.

    The quenched Sub21-BHQ2/Inc21-AF594 duplex is recorded before injection in each InvR21 trace (F_q); free
    Inc21-AF594 at the same concentration is recorded before injection in the same-day Sub21-BHQ2 + Inc21-AF594
    hybridization traces (F_free). Both levels are corrected for the injection volume. The fit is repeated with
    F_free at mean, mean - s.d. and mean + s.d. Returns (DataFrame, F_free mean, F_free s.d., N).
    """
    traces = mf.traces()
    free = []
    for m in traces:
        if m["folder"] == "20250524-b22af594":
            t, y = load_trace(_trace_path(data_dir, m), 1)
            ts, _ = injection_t0(t, y, m["win_start"])
            free.append(pre_injection_level(t, y, ts))
    F_free, F_sd = float(np.mean(free)), float(np.std(free, ddof=1))
    out = []
    for m in traces:
        if m["injected_strand"] != "InvR21":
            continue
        t, y = load_trace(_trace_path(data_dir, m), 1)
        ts, t0 = injection_t0(t, y, m["win_start"])
        F_q = pre_injection_level(t, y, ts)
        sel = (t >= max(m["win_start"], t0)) & (t < m["win_end"])
        tt, yy = t[sel] - t0, y[sel]
        a0, b0 = m["cuvette_stock_uM"] * 1e-6 * mf.DIL, m["injected_stock_uM"] * 1e-6 * mf.DIL
        for F_top, lab in [(F_free, "mean"), (F_free - F_sd, "mean-sd"), (F_free + F_sd, "mean+sd")]:
            res = fit_anchored(tt, yy, a0, b0, F_q, F_top)
            out.append(dict(file=m["file"], invader_final_nM=b0 * 1e9, F_quenched=F_q, F_free=F_top, F_free_choice=lab,
                            **res))
    return pd.DataFrame(out), F_free, F_sd, len(free)


def summary_by_condition(fits):
    """Mean +/- s.d. of k (solution) or k_app (particles) and t1/2 per reaction, channel and concentration."""
    fits = fits.copy()
    fits["k_or_kapp_M_s"] = np.where(fits["model"] == "2nd", fits.get("k_M_s"), fits.get("k_app_M_s"))
    g = fits.groupby(["system", "channel", "cuvette_final_nM", "injected_final_nM"], sort=False)
    s = g.agg(N=("k_or_kapp_M_s", "count"), mean=("k_or_kapp_M_s", "mean"), sd=("k_or_kapp_M_s", "std"),
              t_half_mean_s=("t_half_model_s", "mean"), t_half_sd_s=("t_half_model_s", "std")).reset_index()
    s["quantity"] = np.where(s["system"].str.startswith("F8BT"), "k_app = 1/(t1/2 [Inc]0)",
                             np.where(s["system"].str.contains(r"\+ Inv"), "k_eff", "k_on"))
    return s


def reported_values(fits, anchored):
    """Values reported in Table S4: free-amplitude second-order k for solution reactions, the amplitude-anchored
    k_eff for InvR21, and 5PL t1/2 with k_app for both channels of the particle reactions."""
    fits = fits.copy()
    fits["injected_final_nM_r"] = fits["injected_final_nM"].round(2)
    rows = []
    for (system, ch, cuv, inj), d in fits.groupby(["system", "channel", "cuvette_final_nM", "injected_final_nM_r"],
                                                  sort=False):
        if system.startswith("F8BT"):
            vals, th = d["k_app_M_s"], d["t_half_model_s"]
            qty, method = "k_app = 1/(t1/2 [Inc]0)", "5PL, t1/2 from fitted curve"
        elif "InvR21" in system:
            a = anchored[(anchored.F_free_choice == "mean") & np.isclose(anchored.invader_final_nM, inj, atol=0.05)]
            vals, th = a["anchored_irrev_k_M_s"], pd.Series(dtype=float)
            qty, method = "k_eff", "2nd order, amplitude fixed (pre-injection duplex and same-day free Inc21-AF594)"
        else:
            vals, th = d["k_M_s"], d["t_half_model_s"]
            qty = "k_eff" if "+ Inv" in system else "k_on"
            method = "2nd order, free amplitude" + ("; first 5 s skipped" if d["note"].str.contains("5 s").any() else "")
        rows.append(dict(system=system, channel=ch, quantity=qty, method=method,
                         cuvette_strand=d["cuvette_strand"].iloc[0], cuvette_final_nM=cuv,
                         injected_strand=d["injected_strand"].iloc[0], injected_final_nM=inj,
                         N=int(vals.notna().sum()), mean_M_s=vals.mean(), sd_M_s=vals.std(ddof=1),
                         t_half_mean_s=th.mean() if len(th) else np.nan,
                         t_half_sd_s=th.std(ddof=1) if len(th) else np.nan, panels=d["panel"].iloc[0]))
    return pd.DataFrame(rows)


def start_time_check(data_dir, fits):
    """Refit every second-order trace (except InvR21) from the published start, with a fitted start-time offset,
    and with the first 10 s or 20 s after injection excluded. Returns (per-trace table, per-condition table)."""
    out = []
    for m in mf.traces():
        if m["model"] != "2nd" or "InvR21" in m["system"]:
            continue
        ch, col = m["channels"][0]
        fr = fits[(fits.file == m["file"]) & (fits.channel == ch) & (fits.system == m["system"])].iloc[0]
        t, y = load_trace(_trace_path(data_dir, m), col)
        t0 = fr.t0_s
        a0, b0 = sorted([m["cuvette_stock_uM"] * 1000 * mf.DIL * 1e-9, m["injected_stock_uM"] * 1000 * mf.DIL * 1e-9])
        k0 = fr.k_M_s
        res = dict(system=m["system"], file=m["file"], cuvette_nM=round(m["cuvette_stock_uM"] * 1000 * mf.DIL, 1),
                   injected_nM=round(m["injected_stock_uM"] * 1000 * mf.DIL, 1), k_published=k0,
                   t_half_published=fr.t_half_model_s)
        for name, start, off in [("base", max(m["win_start"], t0 + m["skip_after_t0_s"]), False),
                                 ("offset", max(m["win_start"], t0 + m["skip_after_t0_s"]), True),
                                 ("skip10", max(m["win_start"], t0 + 10), False),
                                 ("skip20", max(m["win_start"], t0 + 20), False)]:
            sel = (t >= start) & (t < m["win_end"])
            try:
                k, d, fe = fit_second_order_variant(t[sel] - t0, y[sel], a0, b0, k0, off)
            except Exception:
                k, d, fe = np.nan, np.nan, np.nan
            res[f"k_{name}"] = k
            if name == "offset":
                res["offset_s"] = d
            if name == "base":
                res["fraction_reacted_at_window_end"] = fe
        out.append(res)
    tr = pd.DataFrame(out)
    summ = tr.groupby(["system", "cuvette_nM", "injected_nM"]).agg(
        N=("file", "size"), k_published=("k_published", "mean"), k_base=("k_base", "mean"),
        k_offset=("k_offset", "mean"), offset_s=("offset_s", "mean"), k_skip10=("k_skip10", "mean"),
        k_skip20=("k_skip20", "mean"), t_half_published=("t_half_published", "mean"),
        min_fraction_reacted=("fraction_reacted_at_window_end", "min")).reset_index()
    return tr, summ


def particle_channel_summary(fits, system="F8BT-Sub21 + Inc21-AF594"):
    """Both channels of a particle reaction per concentration: t1/2 and k_app (mean +/- s.d.), number of traces
    whose t1/2 is extrapolated beyond the data, and the smallest share of the fitted amplitude reached."""
    sub = fits[fits.system == system]
    rows = []
    for ch in ["F8BT 540 nm", "AF594 620 nm"]:
        for c, g in sub[sub.channel == ch].groupby("inc_final_nM"):
            rows.append(dict(channel=ch, inc_final_nM=round(c, 1), N=len(g), t_half_mean=g.t_half_model_s.mean(),
                             t_half_sd=g.t_half_model_s.std(ddof=1), k_app_mean=g.k_app_M_s.mean(),
                             k_app_sd=g.k_app_M_s.std(ddof=1),
                             n_extrapolated=int(g.t_half_extrapolated.astype(str).eq("True").sum()),
                             min_amplitude_reached=g.coverage_fraction_of_amplitude_at_window_end.min(),
                             per_trace_t_half=", ".join(f"{v:.0f}" for v in g.t_half_model_s)))
    return pd.DataFrame(rows).sort_values(["inc_final_nM", "channel"])


def fit_injection_steps(data_dir, files=None):
    """5PL fit of each step of a sequential-injection trace (both channels).

    Step n starts at the first recorded point after injection n and ends 2 s before injection n+1 (or at the end
    of the trace). [Inc]0 of step n is the incumbent added by that injection, final in cuvette:
    0.5 uM x 1.5 uL / V_n with V_n = 151.5 + 1.5 n uL. Steps with a fitted amplitude < 3 a.u. are reported as
    "no further capture" (particle-bound substrate saturated). Missing files are skipped.
    Returns (table, list of fitted curves).
    """
    files = files or mf.INJECTION_FILES
    channels = [("F8BT 540 nm", 1), ("AF594 620 nm", 3)]
    folder = Path(data_dir) / mf.INJECTION_FOLDER
    rows, curves = [], []
    for file, rep in files.items():
        path = folder / file
        if not path.exists():
            print(f"skipped (not found): {path}")
            continue
        inj = all_injections(path, channels)
        for ch, col in channels:
            t, y = load_trace(path, col)
            for n, t_spike in enumerate(inj, start=1):
                i0 = np.searchsorted(t, t_spike + 1.5)             # first recorded point after the spike
                while i0 < len(t) - 1 and abs(y[i0] - np.median(y[i0:i0 + 9])) > 8:   # skip residual spike points
                    i0 += 1
                t0 = t[i0]
                t_end = (inj[n] - 2.0) if n < len(inj) else t[-1]
                sel = (t >= t0) & (t < t_end) & (y > 1.0)           # y > 1 drops a lamp-off tail at the trace end
                tt, yy = t[sel] - t0, y[sel]
                V = 151.5 + 1.5 * n
                inc_nM = mf.INJECTION_STOCK_uM * 1000 * 1.5 / V
                row = dict(file=file, rep=rep, channel=ch, step=n, injection_s=round(t_spike, 1), t0_s=t0,
                           window_end_s=t_end, n_points=len(tt), inc_added_final_nM=inc_nM,
                           inc_total_final_nM=mf.INJECTION_STOCK_uM * 1000 * 1.5 * n / V, cuvette_volume_uL=V)
                if len(tt) < 60 or tt[-1] < 120:
                    row.update(status="not evaluable (< 120 s of data after injection)")
                    rows.append(row)
                    continue
                th_emp = empirical_half_time(tt, yy)
                try:
                    p, _ = curve_fit(fivepl, tt, yy, p0=[yy[:5].mean(), yy[-20:].mean(), max(th_emp, 10.0), 2.0, 1.0],
                                     bounds=([-np.inf, -np.inf, 1e-3, 1e-3, 1e-3], [np.inf, np.inf, 1e5, 50, 50]),
                                     maxfev=40000)
                except Exception as e:
                    row.update(status=f"fit failed: {e}")
                    rows.append(row)
                    continue
                yf = fivepl(tt, *p)
                a, d = p[0], p[1]
                amp = d - a
                if abs(amp) < 3.0:
                    row.update(status="no further capture (amplitude < 3 a.u.)", amplitude=amp)
                    rows.append(row)
                    continue
                th = brentq(lambda x: fivepl(x, *p) - (a + d) / 2, 1e-6, 1e6)
                row.update(status="fitted", amplitude=amp, fivepl_a=a, fivepl_d=d, fivepl_c=p[2], fivepl_p=p[3],
                           fivepl_s=p[4], R2=1 - np.sum((yy - yf) ** 2) / np.sum((yy - yy.mean()) ** 2),
                           t_half_model_s=th, t_half_empirical_s=th_emp,
                           coverage_fraction_of_amplitude_at_window_end=(yf[-1] - a) / (d - a),
                           k_app_M_s=1.0 / (th * inc_nM * 1e-9))
                rows.append(row)
                curves.append((file, ch, n, tt + t0, yf))
    return pd.DataFrame(rows), curves
