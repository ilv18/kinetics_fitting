"""Figure panels: main-figure and SI kinetics panels, the robustness figure, injection-step panels and per-fit
diagnostic plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import manifest as mf
from .data import all_injections, load_trace
from .fitting import A_irrev, fivepl
from .utils import conc_label, sci_mpl

PALETTE = sns.color_palette("viridis", n_colors=6)[::-1]
MAUVE = "#7a3756"


def _save(fig, path, png_dpi=150):
    fig.savefig(Path(path).with_suffix(".svg"), dpi=300, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


def fit_curve(r, m, t_raw, anchored):
    """Fitted curve on the raw time axis for fit row ``r``; InvR21 uses the amplitude-anchored fit."""
    t = t_raw - r["t0_s"]
    if r["model"] == "2nd":
        a0, b0 = r["a0_limiting_nM"] * 1e-9, r["b0_excess_nM"] * 1e-9
        if m["injected_strand"] == "InvR21":
            a = anchored[(anchored.file == r["file"]) & (anchored.F_free_choice == "mean")].iloc[0]
            k = a["anchored_irrev_k_M_s"]
            return a["F_quenched"] + (a["F_free"] - a["F_quenched"]) * (1 - A_irrev(t, k, a0, b0) / a0), k
        return r["P"] * A_irrev(t, r["k_M_s"], a0, b0) / a0 + r["Q"], r["k_M_s"]
    return fivepl(t, r["fivepl_a"], r["fivepl_d"], r["fivepl_c"], r["fivepl_p"], r["fivepl_s"]), r["t_half_model_s"]


def fit_panel(r, m, t, y, path, anchored, legend_detail=True, title=None):
    """Data (scatter) and fit (line) inside the fit window, on the raw time axis."""
    sel = (t >= r["fit_start_s"]) & (t < r["window_end_s"])
    tf = np.linspace(t[sel].min(), t[sel].max(), 400)
    yf, val = fit_curve(r, m, tf, anchored)
    if legend_detail:
        if r["model"] == "2nd":
            q = "k$_{eff}$" if "+ Inv" in r["system"] else "k$_{on}$"
            lab = f"Fit ({q} = {sci_mpl(val)} M$^{{-1}}$ s$^{{-1}}$)"
        else:
            lab = f"Fit (t$_{{1/2}}$ = {val:.0f} s; k$_{{app}}$ = {sci_mpl(r['k_app_M_s'])} M$^{{-1}}$ s$^{{-1}}$)"
    else:
        lab = "Fit"
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(t[sel], y[sel], alpha=0.5, label="Data", color=MAUVE)
    plt.plot(tf, yf, "r-", label=lab)
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (a.u.)")
    if title:
        plt.title(title, fontsize=10)
    plt.legend()
    _save(fig, path)


def raw_panel(m, data_dir, path, anchored=None, title=None, fits_for_channels=None):
    """Raw trace; for particle reactions both channels, optionally with the 5PL fits drawn on."""
    fpath = Path(data_dir) / m["folder"] / m["file"]
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(m["channels"]) == 1:
        t, y = load_trace(fpath, m["channels"][0][1])
        sns.lineplot(x=t, y=y, color=PALETTE[-1], ax=ax)
    else:
        frames = []
        for ch, col in m["channels"]:
            t, y = load_trace(fpath, col)
            frames.append(pd.DataFrame({"Time": t, "Intensity": y, "Group": ch}))
        sns.lineplot(data=pd.concat(frames), x="Time", y="Intensity", hue="Group", palette=PALETTE[:2], ax=ax)
        ax.set_ylim(0, 200)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:2], labels=["ex: 450 / em: 540", "ex: 450 / em: 620"])
        if fits_for_channels is not None:
            for r in fits_for_channels:
                tt = np.linspace(r["fit_start_s"], r["window_end_s"], 400)
                yf, _ = fit_curve(r, m, tt, anchored)
                ax.plot(tt, yf, "r-", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity (a.u.)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_box_aspect(1)
    _save(fig, path)


def trace_panels(fits, anchored, data_dir, out_dir):
    """For every trace: raw panel (i.) and fit panel per channel (ii.), titled with reaction and concentrations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in mf.traces():
        cuv, inj = m["cuvette_stock_uM"] * 1000 * mf.DIL, m["injected_stock_uM"] * 1000 * mf.DIL
        title = f"{m['cuvette_strand']} {conc_label(cuv)} + {m['injected_strand']} {conc_label(inj)}, rep {m['rep']}"
        stem = f"{m['folder']}__{Path(m['file']).stem}"
        raw_panel(m, data_dir, out_dir / f"{stem}__i_raw", title=title)
        for ch, col in m["channels"]:
            r = fits[(fits.file == m["file"]) & (fits.channel == ch)].iloc[0]
            t, y = load_trace(Path(data_dir) / m["folder"] / m["file"], col)
            fit_panel(r, m, t, y, out_dir / f"{stem}__ii_fit_{ch.split()[1]}", anchored, title=f"{title}\n{ch}")


def main_panels(fits, anchored, data_dir, out_dir):
    """Panels of Figures 2H, 2I, 3G and 3H."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = {m["file"]: m for m in mf.traces()}
    for label in ("Fig2H", "Fig3G", "Fig3H"):
        f = mf.MAIN_PANELS[label]
        m, r = traces[f], fits[fits.file == f].iloc[0]
        t, y = load_trace(Path(data_dir) / m["folder"] / m["file"], m["channels"][0][1])
        fit_panel(r, m, t, y, out_dir / f"{label}_{Path(f).stem}", anchored, legend_detail=False)
    f = mf.MAIN_PANELS["Fig2I"]
    raw_panel(traces[f], data_dir, out_dir / f"Fig2I_{Path(f).stem}", anchored=anchored,
              fits_for_channels=[r for _, r in fits[fits.file == f].iterrows()])


def robustness_figure(fits, anchored, out_dir, font_size=10):
    """Figure S12: (a) reversible k_f against irreversible k for every solution trace; (b) InvR21 k_eff with free
    amplitude and with the amplitude fixed (error bars: free-incumbent level at mean -/+ s.d.)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = font_size
    rc = {"font.family": "DejaVu Sans", "font.stretch": "condensed", "font.size": fs, "axes.labelsize": fs,
          "axes.titlesize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
          "legend.title_fontsize": fs, "svg.fonttype": "none", "mathtext.fontset": "custom",
          "mathtext.rm": "DejaVu Sans:stretch=condensed", "mathtext.it": "DejaVu Sans:italic:stretch=condensed",
          "mathtext.bf": "DejaVu Sans:bold:stretch=condensed"}
    sol = fits[fits.model == "2nd"].copy()
    systems = list(dict.fromkeys(sol.system))
    pal = sns.color_palette("viridis", n_colors=len(systems))[::-1]

    def panel_a(ax):
        for s, c in zip(systems, pal):
            d = sol[sol.system == s]
            ax.scatter(d.k_M_s, d.rev_kf_M_s, alpha=0.8, color=c, label=s, edgecolor="k", linewidth=0.3)
        lim = [1e2, 5e6]
        ax.plot(lim, lim, "k--", lw=1, label="k$_\\mathrm{f}$ = k")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("k, irreversible model (M$^{-1}$ s$^{-1}$)")
        ax.set_ylabel("k$_\\mathrm{f}$, reversible model (M$^{-1}$ s$^{-1}$)")
        ax.legend(fontsize=fs, loc="upper left")
        ax.set_box_aspect(1)

    def panel_b(ax):
        free = sol[sol.injected_strand == "InvR21"].set_index("file")["k_M_s"]
        rows = []
        for f, g in anchored.groupby("file", sort=False):
            mid = g[g.F_free_choice == "mean"].anchored_irrev_k_M_s.iloc[0]
            lo, hi = sorted([g[g.F_free_choice == "mean+sd"].anchored_irrev_k_M_s.iloc[0],
                             g[g.F_free_choice == "mean-sd"].anchored_irrev_k_M_s.iloc[0]])
            rows.append((f, g.invader_final_nM.iloc[0], free[f], mid, lo, hi))
        x = np.arange(len(rows))
        ax.scatter(x - 0.12, [r[2] for r in rows], color=MAUVE, alpha=0.8, label="free amplitude")
        ax.errorbar(x + 0.12, [r[3] for r in rows], yerr=[[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]],
                    fmt="o", color="r", capsize=3, label="amplitude anchored")
        ax.set_xticks(x, [f"{r[1]:.0f} nM\nrep {r[0].split('_')[-1][:2]}" for r in rows], fontsize=fs)
        ax.set_yscale("log"); ax.set_ylim(1e2, 2e4)
        ax.set_xlabel("InvR21 (final concentration, replicate)")
        ax.set_ylabel("k$_\\mathrm{eff}$ (M$^{-1}$ s$^{-1}$)")
        ax.legend(fontsize=fs)
        ax.set_box_aspect(1)

    with plt.rc_context(rc):
        for name, fn in [("a_reversible_vs_irreversible", panel_a), ("b_invR21_free_vs_anchored", panel_b)]:
            fig, ax = plt.subplots(figsize=(6, 6))
            fn(ax)
            _save(fig, out_dir / name)
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 6))
        panel_a(a); panel_b(b)
        a.set_title("a)", loc="left", fontweight="bold", fontsize=fs)
        b.set_title("b)", loc="left", fontweight="bold", fontsize=fs)
        fig.tight_layout()
        fig.savefig(out_dir / "kinetics_robustness.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "kinetics_robustness.svg", bbox_inches="tight")
        plt.close(fig)


def injection_panels(curves, data_dir, out_dir, files=None):
    """Sequential-injection traces (both channels) with the 5PL fit of each step and the injection times."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = [("F8BT 540 nm", 1), ("AF594 620 nm", 3)]
    for file in (files or mf.INJECTION_FILES):
        path = Path(data_dir) / mf.INJECTION_FOLDER / file
        if not path.exists():
            continue
        frames = []
        for ch, col in channels:
            t, y = load_trace(path, col)
            frames.append(pd.DataFrame({"Time": t, "Intensity": y, "Group": ch}))
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(data=pd.concat(frames), x="Time", y="Intensity", hue="Group", palette=PALETTE[:2], ax=ax)
        for f, ch, n, tx, yf in curves:
            if f == file:
                ax.plot(tx, yf, "r-", lw=1.5)
        for n, ts in enumerate(all_injections(path, channels)):
            ax.axvline(ts, color="0.3", ls=":", lw=1)
            ax.text(ts + 8, 190, f"t$_{n}$", fontsize=11)
        ax.set_ylim(0, 200)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:2], labels=["ex: 450 / em: 540", "ex: 450 / em: 620"], loc="center right")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Intensity (a.u.)")
        ax.set_box_aspect(1)
        _save(fig, out_dir / f"{Path(file).stem}__injections_withfits")


def diagnostic_plot(m, row, t, y, start, t0, tt, yy, yf, yr, cuv_nM, inj_nM, out_dir):
    """Data inside and outside the fit window, fit(s) and residuals, time from the injection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    pre = t < start
    ax.plot(t[pre] - t0, y[pre], ".", ms=2, color="0.75", label="outside fit window")
    ax.plot(tt, yy, ".", ms=2, color=MAUVE, label="data")
    if yf is not None:
        lab = (f"irreversible, k = {row['k_M_s']:.2e} M$^{{-1}}$s$^{{-1}}$" if m["model"] == "2nd"
               else f"5PL, t$_{{1/2}}$ = {row['t_half_model_s']:.0f} s")
        ax.plot(tt, yf, "-", color="#d62728", lw=1.4, label=lab)
        axr.plot(tt, yy - yf, ".", ms=1.5, color=MAUVE)
    if m["model"] == "2nd" and yr is not None:
        ax.plot(tt, yr, "--", color="#1f77b4", lw=1, label=f"reversible, k$_f$ = {row.get('rev_kf_M_s', np.nan):.2e}")
    axr.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{m['system']} ({row['channel']})\n{m['cuvette_strand']} {conc_label(cuv_nM)} + "
                 f"{m['injected_strand']} {conc_label(inj_nM)}, rep {m['rep']}", fontsize=8)
    ax.set_ylabel("Intensity (a.u.)"); axr.set_ylabel("resid."); axr.set_xlabel("Time after injection (s)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{m['folder']}__{Path(m['file']).stem}__{row['channel'].split()[0]}.png", dpi=150)
    plt.close(fig)
