# kinetics_fitting

Kinetic analysis of DNA hybridization and toehold-mediated strand displacement (TMSD), in solution and on
DNA-conjugated F8BT semiconducting polymer nanoparticles, as used for the real-time fluorescence kinetics in
Van den Bossche *et al.*, bioRxiv, 2026. (Table S4; Figures 2H, 2I, 3G, 3H, S2–S6 and S12).

## Models

**Solution-phase hybridization and displacement.** Each reaction is treated as an irreversible bimolecular
reaction A + B → products, with the initial concentrations *a*<sub>0</sub> ≤ *b*<sub>0</sub> fixed at the final
in-cuvette values:

- *a*<sub>0</sub> = *b*<sub>0</sub> = *c*<sub>0</sub>: [A](*t*) = *c*<sub>0</sub> / (1 + *k c*<sub>0</sub> *t*)
- *a*<sub>0</sub> < *b*<sub>0</sub>: [A](*t*) = *a*<sub>0</sub>(*b*<sub>0</sub> − *a*<sub>0</sub>) / (*b*<sub>0</sub> e<sup>(*b*<sub>0</sub> − *a*<sub>0</sub>)*kt*</sup> − *a*<sub>0</sub>)

The fluorescence is a linear combination of the unreacted and reacted states,
*F*(*t*) = α[A] + β(*a*<sub>0</sub> − [A]); *k*, α and β are fitted by nonlinear least squares, with *k* on a
logarithmic scale. *k* is reported as *k*<sub>on</sub> (hybridization) or *k*<sub>eff</sub> (displacement) in
M<sup>−1</sup> s<sup>−1</sup>. Time zero is the first recorded point after strand injection.

- **Amplitude-fixed fit (InvR21).** RNA-invader traces do not approach completion within the recording time, so the
  amplitude is fixed by independently measured levels: the quenched duplex before injection in each trace, and free
  Inc21-AF594 at the same concentration recorded the same day. Only *k* is fitted.
- **Robustness check.** All solution traces are also fitted with a reversible model A + B ⇌ C + D
  (*k*<sub>f</sub>, *k*<sub>r</sub>), and every second-order trace is refitted with a fitted start-time offset and
  with the first 10 s or 20 s after injection excluded.

**Capture on particle-bound substrate.** Incumbent capture by F8BT–substrate particles is sigmoidal with a lag
phase and is fitted with the five-parameter logistic (5PL) [1, 2]

*f*(*t*) = *d* + (*a* − *d*) / (1 + (*t*/*c*)<sup>*p*</sup>)<sup>*s*</sup>

where *a* and *d* are the intensities at *t* = 0 and *t* → ∞. Donor (540 nm) and acceptor (620 nm) channels are
fitted separately. The half-time *t*<sub>1/2</sub> is the time at which the fitted curve reaches (*a* + *d*)/2.
Because the effective concentration of particle-bound substrate is not known, no rate constant is fitted; an
apparent constant *k*<sub>app</sub> = 1/(*t*<sub>1/2</sub>[Inc]<sub>0</sub>) is given for order-of-magnitude
comparison with *k*<sub>on</sub> only. Sequential-injection traces are fitted step by step in the same way.

## Installation

```bash
pip install .
```

Requires Python ≥ 3.9 with numpy, pandas, scipy, matplotlib and seaborn.

## Data

The raw traces are not included in this repository. They are provided with the paper's data repository, in
`Appendix/TableS4/Raw Data`, with one folder per experiment (e.g. `20250205-b35/`) containing the Cary Eclipse
`.csv` exports. Concentrations in file names are stock concentrations; 1.5 µL of each stock is added to 150 µL
PBST, so final in-cuvette concentrations are stock × 1.5/153. The experimental design of every trace (strands,
concentrations, channels, fit window) is listed in `src/kinetics_fitting/manifest.py`.

## Usage

```bash
kinetics-fitting "path/to/Appendix/TableS4/Raw Data" results
```

or `python -m kinetics_fitting DATA_DIR OUT_DIR`. The output folder must not exist yet. Options: `--no-figures`
(tables only) and `--diagnostics` (data, fit and residual plot for every fit).

| Output | Content |
|---|---|
| `tables/table_S4.tex`, `table_S4_reported_values.csv` | Table S4 |
| `tables/fits_per_trace.csv` | every fit (91 fits from 69 traces) |
| `tables/invR21_amplitude_anchored.csv` | InvR21 amplitude-fixed fits |
| `tables/start_time_check_*.csv` | start-time robustness check |
| `tables/F8BT-Sub21_both_channels.csv` | F8BT-Sub21 capture, both channels |
| `tables/injection_steps.csv` | sequential-injection steps (Figure S3b) |
| `figures/main/` | Figures 2H, 2I, 3G, 3H |
| `figures/traces/` | raw (i) and fitted (ii) panel of every trace (Figures S2, S3a, S5, S6) |
| `figures/robustness/` | Figure S12 |
| `figures/injections/` | Figure S3b |

## References

1. Gottschalk, P. G. & Dunn, J. R. The five-parameter logistic: a characterization and comparison with the
   four-parameter logistic. *Anal. Biochem.* **343**, 54–65 (2005).
2. Ghotra, G., Nguyen, B. K. & Chen, J. I. L. DNA-functionalized gold nanoparticles with toehold-mediated strand
   displacement for nucleic acid sensors. *ACS Appl. Nano Mater.* **3**, 10123–10132 (2020).
3. Srinivas, N. *et al.* On the biophysics and kinetics of toehold-mediated DNA strand displacement.
   *Nucleic Acids Res.* **41**, 10641–10658 (2013).
