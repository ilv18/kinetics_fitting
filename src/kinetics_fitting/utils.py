"""Formatting helpers and the LaTeX version of Table S4."""
import numpy as np
import pandas as pd


def conc_label(nM):
    """Concentration label: nM above 1 nM, pM below."""
    return f"{nM:.3g} nM" if nM >= 1 else f"{nM * 1000:.3g} pM"


def sci_mpl(x):
    """Scientific notation for matplotlib labels, e.g. 7.58x10^5."""
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.2f}×10$^{{{e}}}$"


def sci_tex(x):
    """Scientific notation for LaTeX, e.g. 7.58 \\times 10^{5}."""
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.2f} \\times 10^{{{e}}}"


def table_s4_tex(reported):
    """Table S4 rows (LaTeX tabular body plus header) from the reported values."""
    q = {"k_on": r"$k_{\mathrm{on}}$", "k_eff": r"$k_{\mathrm{eff}}$"}
    lines = [r"\begin{tabular}{llllcll}", r"\toprule",
             r"Reaction & Channel & [cuvette] & [injected] & $N$ & $t_{1/2}$ (s) & $k$ (M$^{-1}$\,s$^{-1}$) \\",
             r"\midrule"]
    for _, r in reported.iterrows():
        qty = q.get(r["quantity"], r"$k_{\mathrm{app}}$")
        th = "--" if pd.isna(r["t_half_mean_s"]) else f"{r['t_half_mean_s']:.0f} $\\pm$ {r['t_half_sd_s']:.0f}"
        sd = "" if pd.isna(r["sd_M_s"]) else f" \\pm {sci_tex(r['sd_M_s'])}"
        lines.append(f"{r['system']} & {r['channel']} & {conc_label(r['cuvette_final_nM'])} & "
                     f"{conc_label(r['injected_final_nM'])} & {r['N']} & {th} & {qty} = ${sci_tex(r['mean_M_s'])}{sd}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
