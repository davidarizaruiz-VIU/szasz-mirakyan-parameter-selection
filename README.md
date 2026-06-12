# Optimal selection of the preserved exponential in Szász–Mirakyan operators

Reproducibility code for the manuscript

> **Optimal selection of the preserved exponential in Szász–Mirakyan operators: a leading-term criterion with finite-*n* guarantees**
> David Ariza-Ruiz (Valencian International University, VIU) and Dionisio F. Yáñez (Universitat de València)
> Submitted to *Dolomites Research Notes on Approximation* (DRNA).

This repository contains the code, data and figures needed to reproduce the numerical
results of the paper. The manuscript itself is not included here; it will be available
from DRNA. The official DRNA LaTeX template can be obtained from the journal website.

## What this is about

For the **known** exponential-preserving Szász–Mirakyan operators
*R*<sub>*n*,*a*</sub>*f* = *S*<sub>*n*</sub>(*f*; α<sub>*n*,*a*</sub>), which reproduce {1, *e*<sup>2*ax*</sup>}
(Acar–Aral–Gonska 2017; Aral–Inoan–Raşa 2019), we do **not** modify the operators.
We treat the preserved rate *a* as a *design parameter* and choose it by minimizing the
weighted *L*² norm of the leading Voronovskaya term. The code reproduces every numerical
claim in the paper:

- a closed-form optimal parameter *a*\* and a finite-*n* justification
  *n*² *E*<sub>*n*</sub>(*a*)² = ¼ *J*(*a*) + *O*(1/*n*), hence *a*<sub>*n*</sub>\* → *a*\*;
- a saturation reading: the trivial class is span{1, *e*<sup>2*ax*</sup>}, so *a*\* minimizes the
  **saturation constant** (the 1/*n* order itself being classical, Becker 1978; Ditzian–Totik 1987);
- a **computable a posteriori selector** that recovers *a*\*, its sensitivity and the loss of any
  admissible parameter from three error evaluations, using no derivative of *f*, and that satisfies
  an **oracle inequality** (within a factor 1 + *O*(1/*n*²) of the finite-*n* optimum);
- explicit optima for structured classes (single exponentials, positive mixtures, layers);
- a **uniform-in-ε relative-error bound** for the perturbed boundary layer (1+*x*)*e*<sup>−*x*/ε</sup>,
  extended in closed form (via Touchard polynomials) to any polynomial-modulated layer
  *g*(*x*)*e*<sup>−*x*/ε</sup>, with the bound holding exactly when *g*(0) ≠ 0;
- a like-for-like comparison with the mixed two-exponential operator of Ulusoy Ada–Aral (2026),
  generalized to a **domination** statement over all convex two-exponential mixtures.

The operators, their Voronovskaya-type asymptotics and the 1/*n* saturation order are not new and
are cited as such.

## Repository structure

```
.
├── code/                       Python reproducibility scripts + precomputed data
│   ├── num_core.py             operator core; Table 1 + data for F1, F2, F4
│   ├── num_eoc.py              L∞ errors and EOC (Table 2)
│   ├── enrich_C.py             perturbed-layer data (F5; Theorem 5.5 setting)
│   ├── make_figures_pdf.py     builds the four vector-PDF figures into ../figs/
│   ├── posteriori_selector.py  a posteriori selector (self-contained)
│   ├── comparison_and_gain.py  gain law + Ulusoy Ada–Aral comparison (self-contained)
│   ├── robustness_and_sharp.py uniform-in-ε robustness + first-order term (self-contained)
│   └── results_f*.npz, rows_f*.json, eoc_results.json, enrichC_rows.json   precomputed data
├── figs/                       vector-PDF figures used by the paper
│   ├── F1_n2En2_vs_J_f3.pdf
│   ├── F2_anstar_vs_invn.pdf
│   ├── F4_mixtures_pred_vs_meas.pdf
│   └── F5_layer_scale.pdf
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `matplotlib`:

```bash
pip install -r requirements.txt
```

No LaTeX is needed to run the code.

## Reproduce the numerics

Run everything from the `code/` directory:

```bash
cd code

# Table 1 and per-function data (also feeds figures F1, F2, F4)
python3 num_core.py 1
python3 num_core.py 2
python3 num_core.py 3
python3 num_core.py 4

# Table 2: uniform-norm errors and experimental order of convergence
python3 num_eoc.py

# Perturbed boundary-layer data (figure F5; Theorem 5.5 setting)
python3 enrich_C.py

# Build the four figures as vector PDF into ../figs/
python3 make_figures_pdf.py

# Self-contained checks of the main analytic claims (no data files needed)
python3 posteriori_selector.py
python3 comparison_and_gain.py
python3 robustness_and_sharp.py
```

The repository ships the precomputed data (`results_f*.npz`, `enrichC_rows.json`, …), so
`make_figures_pdf.py` runs immediately; the generator scripts above regenerate it from scratch.

## Script → result map

| Script | Reproduces in the paper |
|---|---|
| `num_core.py` | Table 1 (*a*\*, *a*<sub>*n*</sub>\*, errors, gain); data for Figures F1, F2, F4 |
| `num_eoc.py` | Table 2 (*L*<sup>∞</sup> errors and EOC) |
| `enrich_C.py` | Figure F5 data; perturbed-layer rate *a*\*<sub>ε</sub> → −1/(2ε); §6 layer paragraphs |
| `make_figures_pdf.py` | Figures F1, F2, F4, F5 (vector PDF) |
| `posteriori_selector.py` | A posteriori selector (parabola-vertex rule) |
| `comparison_and_gain.py` | Gain law *E*<sub>*n*</sub>(0)/*E*<sub>*n*</sub>(*a*\*) → 1/√(1−ρ²) and the comparison with Ulusoy Ada–Aral (2026) |
| `robustness_and_sharp.py` | Uniform-in-ε relative robustness (Theorem 5.5) and the explicit first-order term *G*(*a*) |

## Reproducibility notes

- Operators are evaluated with a numerically stable log-space Poisson kernel; the self-contained
  scripts additionally use exact closed forms (no series truncation) for exponential mixtures and
  for the perturbed layer.
- Fixed setup: domain [0, *X*] with *X* = 10, weight *w* ≡ 1, mesh of 2001 nodes (graded near 0 for
  sharp layers), *n* ∈ {20, 40, 80, 160, 320}.
- Cross-platform last-digit differences (BLAS, rounding) are possible; the reported convergence
  orders, ratios and bounds are stable.

## License

- **Code** (`code/`): MIT License — see [`LICENSE`](LICENSE).
- **Figures** (`figs/`): © the authors. Upon publication, the article and its figures are
  distributed by DRNA under a Creative Commons **CC BY-NC-ND** license.

## Citation

See [`CITATION.cff`](CITATION.cff). Provisional BibTeX:

```bibtex
@unpublished{ArizaRuizYanez2026exp,
  author = {Ariza-Ruiz, David and Y\'a\~nez, Dionisio F.},
  title  = {Optimal selection of the preserved exponential in
            Sz\'asz--Mirakyan operators: a leading-term criterion
            validated at finite $n$},
  note   = {Submitted to Dolomites Research Notes on Approximation},
  year   = {2026}
}
```
