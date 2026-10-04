"""Command-line entry point: fit all traces and write tables and figures.

    kinetics-fitting DATA_DIR OUT_DIR [--no-figures] [--diagnostics]

DATA_DIR holds one folder per experiment (e.g. 20250205-b35/) with the Cary Eclipse .csv exports.
OUT_DIR must not exist yet; results are never overwritten.
"""
import argparse
from pathlib import Path

import pandas as pd

from . import analysis
from . import manifest as mf
from .utils import table_s4_tex


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kinetic fits of the F8BT-DNA probe paper (Table S4, Figs. 2H/I, 3G/H, "
                                             "S2-S6, S12).")
    ap.add_argument("data_dir", type=Path, help="folder with one subfolder per experiment")
    ap.add_argument("out_dir", type=Path, help="new output folder")
    ap.add_argument("--no-figures", action="store_true", help="tables only")
    ap.add_argument("--diagnostics", action="store_true", help="also save a data/fit/residual plot for every fit")
    a = ap.parse_args(argv)

    if a.out_dir.exists():
        raise SystemExit(f"{a.out_dir} already exists; choose a new output folder.")
    tables = a.out_dir / "tables"
    tables.mkdir(parents=True)

    mf.manifest_table().to_csv(tables / "trace_manifest.csv", index=False)
    fits = analysis.fit_all(a.data_dir, diagnostics_dir=(a.out_dir / "diagnostics") if a.diagnostics else None)
    fits.to_csv(tables / "fits_per_trace.csv", index=False)

    anchored, F_free, F_sd, n_free = analysis.anchored_invR21(a.data_dir)
    anchored.to_csv(tables / "invR21_amplitude_anchored.csv", index=False)
    print(f"Free Inc21-AF594 level (same day, N = {n_free}): {F_free:.1f} +/- {F_sd:.1f} a.u.")

    analysis.summary_by_condition(fits).to_csv(tables / "summary_by_condition.csv", index=False)
    reported = analysis.reported_values(fits, anchored)
    reported.to_csv(tables / "table_S4_reported_values.csv", index=False)
    (tables / "table_S4.tex").write_text(table_s4_tex(reported))

    per_trace, per_condition = analysis.start_time_check(a.data_dir, fits)
    per_trace.to_csv(tables / "start_time_check_per_trace.csv", index=False)
    per_condition.to_csv(tables / "start_time_check_by_condition.csv", index=False)
    analysis.particle_channel_summary(fits).to_csv(tables / "F8BT-Sub21_both_channels.csv", index=False)

    inj, curves = analysis.fit_injection_steps(a.data_dir)
    inj.to_csv(tables / "injection_steps.csv", index=False)

    if not a.no_figures:
        from . import plotting
        plotting.main_panels(fits, anchored, a.data_dir, a.out_dir / "figures" / "main")
        plotting.trace_panels(fits, anchored, a.data_dir, a.out_dir / "figures" / "traces")
        plotting.robustness_figure(fits, anchored, a.out_dir / "figures" / "robustness")
        plotting.injection_panels(curves, a.data_dir, a.out_dir / "figures" / "injections")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
        print(reported[["system", "channel", "quantity", "cuvette_final_nM", "injected_final_nM", "N", "mean_M_s",
                        "sd_M_s", "t_half_mean_s", "t_half_sd_s"]])
    print(f"{len(fits)} fits written to {a.out_dir}")


if __name__ == "__main__":
    main()
