"""Reading Cary Eclipse kinetics exports and locating strand injections."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_trace(path, col):
    """Return (time, intensity) for one channel of a Cary Eclipse .csv export.

    The export has two header rows and one (time, intensity) column pair per channel; ``col`` is the 1-based index
    of the intensity column (1 = first channel, 3 = second channel).
    """
    raw = pd.read_csv(Path(path), header=None, skiprows=2).apply(pd.to_numeric, errors="coerce")
    d = raw.iloc[:, [col - 1, col]].dropna()
    return d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)


def injection_t0(t, y, win_start):
    """Injection spike = largest deviation from the pre-injection baseline before the fit-window start.

    Returns (t_spike, t0), with t0 the first recorded time point after the spike (time zero of the fit).
    """
    search = t <= min(win_start, 200.0) + 1.0
    ts, ys = t[search], y[search]
    base = np.median(ys[: max(5, len(ys) // 5)])
    i = int(np.argmax(np.abs(ys - base)))
    return ts[i], (t[i + 1] if i + 1 < len(t) else t[i])


def pre_injection_level(t, y, t_spike):
    """Median signal 30-2 s before the injection, corrected from 151.5 uL to 153 uL (dilution by the injection)."""
    return np.median(y[(t < t_spike - 2) & (t > t_spike - 30)]) * 151.5 / 153


def find_injections(t, y):
    """Injection spikes in a multi-injection trace: points deviating strongly from a running median, grouped
    within 10 s. Returns a list of (spike time, index of last spike point)."""
    med = pd.Series(y).rolling(9, center=True, min_periods=1).median().to_numpy()
    dev = np.abs(y - med)
    idx = np.where(dev > max(8.0, 8 * np.median(dev)))[0]
    groups = []
    for i in idx:
        if groups and t[i] - t[groups[-1][-1]] <= 10:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [(t[g[0]], g[-1]) for g in groups if t[g[0]] > 20]


def all_injections(path, channels):
    """Union of the injections found in all channels of one file, merged within 10 s (earliest time kept)."""
    ts = sorted(tt for _, col in channels for tt, _ in find_injections(*load_trace(path, col)))
    merged = []
    for x in ts:
        if not merged or x - merged[-1] > 10:
            merged.append(x)
    return merged
