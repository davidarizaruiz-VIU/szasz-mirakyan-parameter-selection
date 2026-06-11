"""
Supplementary numerical check: uniform-norm (L^infty) error and experimental
order of convergence (EOC) for the parameter-selection study.

Reuses the SAME operator, mesh, n-grid and stable log-space Poisson kernel as
num_core.py (X=10, w=1, n in {20,40,80,160,320}).  This script ONLY adds:

    E_{n,2}(a;f)   = ( int_0^X |R_{n,a}f - f|^2 dx )^{1/2}        (already in paper)
    E_{n,inf}(a;f) = max_{x in [0,X]} |R_{n,a}f - f|              (new, supplementary)

and the experimental orders of convergence

    EOC_2(n)   = log2( E_{n,2}   / E_{2n,2}   )
    EOC_inf(n) = log2( E_{n,inf} / E_{2n,inf} )

for a in {0, -1/2, a*}.  No theory, operator or claim is modified.

Run:  python3 num_eoc.py            (prints tables for f1..f4, writes eoc_results.json)
"""
import json
import numpy as np
from scipy.integrate import simpson

# --- identical setup to num_core.py ---
from num_core import X, NX, NS, xs, make_fun, Sn, a_star


def errors(f, n, a):
    """Return (E_{n,2}, E_{n,inf}) for R_{n,a} f on the fixed mesh."""
    d = Sn(f, n, a) - f(xs)
    e2 = float(np.sqrt(simpson(d**2, xs)))
    einf = float(np.max(np.abs(d)))
    return e2, einf


def eoc(seq):
    """EOC between successive (doubled-n) entries; returns list aligned to NS[:-1]."""
    out = []
    for i in range(len(seq) - 1):
        if seq[i] > 0 and seq[i + 1] > 0:
            out.append(np.log2(seq[i] / seq[i + 1]))
        else:
            out.append(float("nan"))
    return out


def run(fi):
    f, fp, fpp, _, label = make_fun(fi)
    ast, Q, R = a_star(fp, fpp)
    avals = {"0": 0.0, "-1/2": -0.5, "a*": ast}

    data = {}
    for tag, a in avals.items():
        e2s, einfs = [], []
        for n in NS:
            e2, einf = errors(f, n, a)
            e2s.append(e2)
            einfs.append(einf)
        data[tag] = dict(a=a, E2=e2s, Einf=einfs,
                         EOC2=eoc(e2s), EOCinf=eoc(einfs))

    # ---- print a compact block ----
    print(f"\n=== {label}   (a* = {ast:.5f}) ===")
    hdr = f"{'a':>6} {'n':>5} {'E_n2':>11} {'EOC2':>6} {'E_ninf':>11} {'EOCinf':>7}"
    print(hdr)
    for tag, a in avals.items():
        d = data[tag]
        for i, n in enumerate(NS):
            o2 = "" if i == len(NS) - 1 else f"{d['EOC2'][i]:.3f}"
            oi = "" if i == len(NS) - 1 else f"{d['EOCinf'][i]:.3f}"
            atag = tag if i == 0 else ""
            print(f"{atag:>6} {n:>5} {d['E2'][i]:>11.4e} {o2:>6} {d['Einf'][i]:>11.4e} {oi:>7}")
    return label, ast, data


if __name__ == "__main__":
    allres = {}
    for fi in (1, 2, 3, 4):
        label, ast, data = run(fi)
        allres[f"f{fi}"] = dict(label=label, a_star=ast, NS=NS, data=data)
    json.dump(allres, open("eoc_results.json", "w"), indent=1)
    print("\n[written] eoc_results.json")
