"""Experimental design of every real-time kinetics trace analysed in the paper.

Each trace is a Cary Eclipse export ``<data_dir>/<experiment folder>/<file>.csv``. Concentrations are given as
stock concentrations (as in the file names); 1.5 uL of each stock is added to 150 uL PBST, so the final in-cuvette
concentration is stock x 1.5/153. ``win`` is the fit window in raw instrument time (s); points before the
injection are never used.
"""
import pandas as pd

DIL = 1.5 / 153.0          # stock -> final in-cuvette concentration
SPN_STOCK_nM = 2.0         # F8BT-Sub particle stock (nM) -> 19.6 pM in the cuvette

_TRACES = []


def _add(system, panel, folder, file, rep, cuv, cuv_uM, inj, inj_uM, channels, model, win, note="", skip_after_t0_s=0.0):
    _TRACES.append(dict(system=system, panel=panel, folder=folder, file=file, rep=rep,
                        cuvette_strand=cuv, cuvette_stock_uM=cuv_uM, injected_strand=inj, injected_stock_uM=inj_uM,
                        channels=channels, model=model, win_start=win[0], win_end=win[1], note=note,
                        skip_after_t0_s=skip_after_t0_s))


# Sub35-BHQ2 (1 uM stock, injected) into Inc35-AF594 (stock in file name); AF594 620 nm, decreasing
_w = {("1uM", 1): (56.75, 953.75), ("1uM", 2): (53.75, 907.25), ("1uM", 3): (52.75, 647.25),
      ("500nM", 1): (54.75, 477.75), ("500nM", 2): (54.75, 464.25), ("500nM", 3): (53.25, 589.25),
      ("2uM", 1): (55.75, 701.75), ("2uM", 2): (58.75, 607.25), ("2uM", 3): (58.75, 702.75)}
for (_c, _r), _win in _w.items():
    _add("Sub35-BHQ2 + Inc35-AF594", "Fig. 2H, S2a", "20250305-b35BHQ", f"{_c}-b25AF_add-b35BHQ-{_r:03d}.csv", _r,
         "Inc35-AF594", {"1uM": 1, "2uM": 2, "500nM": 0.5}[_c], "Sub35-BHQ2", 1.0, [("AF594 620 nm", 1)], "2nd", _win,
         note="Fig. 2H and S2a trace" if (_c, _r) == ("1uM", 1) else "")

# Sub35-BHQ2 (injected) into Inc35-FAM; FAM 520 nm, decreasing
_w = {("1uM", 1): (47.25, 700), ("1uM", 2): (50.75, 800), ("1uM", 3): (48.25, 704.75),
      ("500nM", 1): (35.75, 827.25), ("500nM", 2): (38.25, 900.75), ("500nM", 3): (50.75, 699.75),
      ("2uM", 1): (51.25, 435.75), ("2uM", 2): (34.25, 372.25), ("2uM", 3): (56.75, 557.25)}
for (_c, _r), _win in _w.items():
    _add("Sub35-BHQ2 + Inc35-FAM", "Fig. S2b", "20250305-b35BHQ", f"{_c}-b25FAM_add-b35BHQ-{_r:03d}.csv", _r,
         "Inc35-FAM", {"1uM": 1, "2uM": 2, "500nM": 0.5}[_c], "Sub35-BHQ2", 1.0, [("FAM 520 nm", 1)], "2nd", _win,
         note="Fig. S2b trace" if (_c, _r) == ("1uM", 3) else "")

# InvD35 (stock in file name, injected) into Sub35-BHQ2/Inc35-AF594 (1 uM stock); AF594 620 nm, increasing
_w = {("1uM", 1): (41.25, 650), ("1uM", 2): (42.75, 400), ("1uM", 3): (37.25, 650),
      ("2uM", 1): (38.75, 500), ("2uM", 2): (39.75, 550), ("2uM", 3): (43.75, 500),
      ("5uM", 1): (43.75, 580), ("5uM", 2): (34.25, 500), ("5uM", 3): (38.25, 450)}
for (_c, _r), _win in _w.items():
    _note = "first 5 s after injection skipped (mixing artifact)" if _c == "5uM" else (
        "Fig. S2c trace" if (_c, _r) == ("1uM", 1) else "")
    _add("Sub35-BHQ2/Inc35-AF594 + InvD35", "Fig. S2c", "20250314-b35invader", f"1uM-b35probe-{_c}invD-{_r:03d}.csv", _r,
         "Sub35-BHQ2/Inc35-AF594", 1.0, "InvD35", {"1uM": 1, "2uM": 2, "5uM": 5}[_c], [("AF594 620 nm", 1)], "2nd", _win,
         skip_after_t0_s=5.0 if _c == "5uM" else 0.0, note=_note)

# Inc35-AF594 (stock) injected into F8BT-Sub35 (2 nM stock); F8BT 540 nm and AF594 620 nm; 5PL
_b35spn = {  # file number -> (Inc stock uM, replicate, window)
    2: (1, 1, (61.45624924, 815.2212524)), 12: (1, 2, (65.56200244, 866.9324951)),
    4: (0.5, 1, (67.20500183, 1335.527466)), 5: (0.5, 2, (66.35874939, 1293.278809)),
    6: (2, 1, (67.66374969, 548.5437622)), 7: (2, 2, (77.26499939, 550)), 11: (2, 3, (65.36374664, 600)),
    8: (5, 1, (67.11374664, 347.573761)), 9: (5, 2, (59.46125031, 341.6620085)), 10: (5, 3, (65.56375122, 367.1187439)),
    13: (8, 1, (64.56200244, 242.1649933)), 14: (8, 2, (63.95500183, 242.9137573)), 15: (8, 3, (67.21125031, 280.6712646))}
for _n, (_c, _r, _win) in _b35spn.items():
    _add("F8BT-Sub35 + Inc35-AF594", "Fig. 2I, S3a", "20250205-b35", f"KINETICS-titration-b35-{_n:02d}.csv", _r,
         "F8BT-Sub35", SPN_STOCK_nM / 1000, "Inc35-AF594", _c, [("F8BT 540 nm", 1), ("AF594 620 nm", 3)], "5PL", _win,
         note="Fig. 2I and S3a trace" if _n == 11 else "")

# Sub21-BHQ2 (stock in file name, injected) into Inc21-AF594 (1 uM stock); AF594 620 nm, decreasing
_w = {(1, 1): (63.75, 786.75), (1, 2): (61.75, 801.25), (1, 3): (55.25, 797.75),
      (2, 1): (53.75, 451.75), (2, 2): (45.25, 506.25), (2, 3): (38.75, 400.75),
      (5, 1): (44.75, 200), (5, 2): (43.25, 231.75), (5, 3): (49.25, 160)}
for (_c, _r), _win in _w.items():
    _add("Sub21-BHQ2 + Inc21-AF594", "Fig. 3G, S5a", "20250524-b22af594", f"{_c}uM-b22-1uM-b15AF594_{_r:02d}.csv", _r,
         "Inc21-AF594", 1.0, "Sub21-BHQ2", _c, [("AF594 620 nm", 1)], "2nd", _win,
         note="Fig. 3G and S5a trace" if (_c, _r) == (5, 1) else "")

# InvD21 (stock, injected) into Sub21-BHQ2/Inc21-AF594 (1 uM stock); AF594 620 nm, increasing
_w = {(2, 1): (39.75, 773.25), (2, 2): (52.75, 1459.75), (2, 3): (45.25, 1650),
      (5, 1): (43.75, 959.25), (5, 2): (55.75, 703.75), (5, 3): (37.75, 1596.25)}
for (_c, _r), _win in _w.items():
    _add("Sub21-BHQ2/Inc21-AF594 + InvD21", "Fig. S5b", "20250317-b22invader", f"1uM-b22b15AF_{_c}uMinvD-{_r:03d}.csv", _r,
         "Sub21-BHQ2/Inc21-AF594", 1.0, "InvD21", _c, [("AF594 620 nm", 1)], "2nd", _win,
         note="Fig. S5b trace" if (_c, _r) == (5, 1) else "")

# InvR21 (stock, injected) into Sub21-BHQ2/Inc21-AF594 (1 uM stock); AF594 620 nm, increasing
_w = {(5, 1): (36.25, 4025.25), (5, 2): (36.25, 4239.75),
      (10, 1): (322.25, 2155.75), (10, 2): (480, 2153.75), (10, 3): (100, 2430.75)}
for (_c, _r), _win in _w.items():
    _add("Sub21-BHQ2/Inc21-AF594 + InvR21", "Fig. 3H, S5c", "20250524-b22invaderRNA", f"1uM-b22b15AF-{_c}uM-invR_{_r:02d}.csv", _r,
         "Sub21-BHQ2/Inc21-AF594", 1.0, "InvR21", _c, [("AF594 620 nm", 1)], "2nd", _win,
         note="Fig. 3H and S5c trace" if (_c, _r) == (10, 2) else "")

# Inc21-AF594 (stock) injected into F8BT-Sub21 (2 nM stock); F8BT 540 nm and AF594 620 nm; 5PL
_b22spn = {2: (8, 1, (81.41500092, 810)), 3: (8, 2, (62.81375122, 730.6737671)), 4: (8, 3, (58.10499954, 602.8637695)),
           5: (2, 1, (64.65750122, 774.7287598)), 6: (2, 2, (62.61500168, 600)), 7: (2, 3, (56.25500107, 450)),
           8: (5, 1, (63.06375122, 745.9237671)), 9: (5, 2, (45.15499878, 643.8150024)), 10: (5, 3, (77.31500244, 773.3800049))}
for _n, (_c, _r, _win) in _b22spn.items():
    _f = "KINETICS-titration-b22-06_orig.csv" if _n == 6 else f"KINETICS-titration-b22-{_n:02d}.csv"
    _add("F8BT-Sub21 + Inc21-AF594", "Fig. S6", "20250207-b22ncel39", _f, _r,
         "F8BT-Sub21", SPN_STOCK_nM / 1000, "Inc21-AF594", _c, [("F8BT 540 nm", 1), ("AF594 620 nm", 3)], "5PL", _win,
         note="Fig. S6 trace" if _n == 9 else "")

# Traces drawn in the main-figure panels
MAIN_PANELS = {
    "Fig2H": "1uM-b25AF_add-b35BHQ-001.csv",
    "Fig2I": "KINETICS-titration-b35-11.csv",
    "Fig3G": "5uM-b22-1uM-b15AF594_01.csv",
    "Fig3H": "1uM-b22b15AF-10uM-invR_02.csv",
}
# Sequential-injection traces (three 1.5 uL injections of 0.5 uM Inc35-AF594 into F8BT-Sub35); Fig. S3b = b35-05
INJECTION_FILES = {"KINETICS-titration-b35-04.csv": 1, "KINETICS-titration-b35-05.csv": 2,
                   "KINETICS-titration-b35-16.csv": 3}
INJECTION_FOLDER = "20250205-b35"
INJECTION_STOCK_uM = 0.5


def traces():
    """List of trace dictionaries (one per raw file)."""
    return [dict(t) for t in _TRACES]


def manifest_table():
    """Trace manifest as a DataFrame."""
    return pd.DataFrame(_TRACES)
