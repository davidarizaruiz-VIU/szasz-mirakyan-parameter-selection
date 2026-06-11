"""Regenerate the four manuscript figures as vector PDF into ../figs/.

F1, F2, F4 are built from results_f{1..4}.npz (produced by num_core.py);
F5 is built from enrichC_rows.json (produced by enrich_C.py).
All inputs are read from this script's directory, so it works from any cwd.

Run:  python3 make_figures_pdf.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(BASE, "..", "figs"))
os.makedirs(OUT, exist_ok=True)
def load_npz(fi): return dict(np.load(os.path.join(BASE, f"results_f{fi}.npz"), allow_pickle=True))
def save(name): plt.tight_layout(); plt.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close()

NS = [20, 40, 80, 160, 320]
D = {fi: load_npz(fi) for fi in [1, 2, 3, 4]}
def lab(fi): return str(D[fi]['label'])
cols = ['C0', 'C1', 'C2', 'C3']

# ---------- F1: n^2 E_n^2(a) vs J/4, mixture f3 ----------
fi = 3; ag = D[fi]['agrid']; J4 = D[fi]['Jgrid'] / 4
plt.figure(figsize=(7, 5))
for n in NS:
    plt.plot(ag, n*n*D[fi][f'En2_{n}'], lw=1.3, label=f"$n^2E_n^2$, n={n}")
plt.plot(ag, J4, 'k--', lw=2.2, label=r"$\frac{1}{4} J(a)$")
plt.axvline(float(D[fi]['a_star']), color='r', ls=':', label=r"$a^\ast$")
plt.xlabel("a"); plt.ylabel(r"$n^2E_n(a;f)^2$  and  $\frac{1}{4} J(a)$")
plt.title(r"$n^2E_n^2(a)\to\frac{1}{4} J(a)$ — " + lab(3)); plt.legend(fontsize=8); plt.grid(alpha=.3)
save("F1_n2En2_vs_J_f3.pdf")

# ---------- F2: a_n* vs 1/n, all four ----------
plt.figure(figsize=(7, 5))
for j, fi in enumerate([1, 2, 3, 4]):
    invn = 1.0/np.array(NS); an = D[fi]['an_stars']; ast = float(D[fi]['a_star'])
    plt.plot(invn, an, 'o-', color=cols[j], label=f"{lab(fi)}: $a_n^*$")
    plt.axhline(ast, color=cols[j], ls='--', lw=1)
plt.xlabel("1/n"); plt.ylabel(r"$a_n^\ast$  (dashed = $a^\ast$)")
plt.title(r"$a_n^\ast\to a^\ast$ as $1/n\to0$"); plt.legend(fontsize=7); plt.grid(alpha=.3)
save("F2_anstar_vs_invn.pdf")

# ---------- F4: positive mixtures, a* vs a_n* + convex-hull bands ----------
plt.figure(figsize=(7, 5))
info = {3: (-1.5, -0.5), 4: (-2.0, -0.5)}
for j, fi in enumerate([3, 4]):
    invn = 1.0/np.array(NS); an = D[fi]['an_stars']; ast = float(D[fi]['a_star'])
    plt.plot(invn, an, 'o-', color=cols[j+2], label=f"{lab(fi)}: $a_n^*$")
    plt.axhline(ast, color=cols[j+2], ls='--', lw=1.2, label=f"   $a^*$={ast:.4f}")
    lo, hi = info[fi]; plt.axhspan(lo, hi, color=cols[j+2], alpha=0.06)
plt.xlabel("1/n"); plt.ylabel(r"$a_n^\ast$ vs predicted $a^\ast$ (shaded: convex hull of $-\lambda_j/2$)")
plt.title("Positive mixtures: predicted vs measured optimal parameter")
plt.legend(fontsize=8); plt.grid(alpha=.3)
save("F4_mixtures_pred_vs_meas.pdf")

# ---------- F5: perturbed layer (1+x) e^{-x/eps} ----------
R = json.load(open(os.path.join(BASE, "enrichC_rows.json")))
eps = np.array([r['eps'] for r in R]); lam = np.array([r['lam'] for r in R])
a_pred = np.array([r['a_star'] for r in R]); a_emp = np.array([r['an_320'] for r in R])
scale = 1.0/(2*eps)
ratio = np.array([r['ratio_astar_to_scale'] for r in R])
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.7))
axL.loglog(lam, scale, 'k--', lw=2, label=r"$1/(2\varepsilon)$")
axL.loglog(lam, -a_pred, 'o-', label=r"$-a^\ast_\varepsilon$ (predicted)")
axL.loglog(lam, -a_emp, 's', ms=5, label=r"$-a^\ast_{320,\varepsilon}$ (empirical)")
axL.set_xlabel(r"$1/\varepsilon$"); axL.set_ylabel(r"$-a^\ast_\varepsilon$")
axL.set_title("Optimal rate tracks the layer scale"); axL.legend(fontsize=8); axL.grid(alpha=.3, which='both')
axR.semilogx(1.0/eps, ratio, 'o-', color='C3')
axR.axhline(1.0, color='k', ls=':', lw=1)
axR.set_xlabel(r"$1/\varepsilon$"); axR.set_ylabel(r"$a^\ast_\varepsilon/(-1/(2\varepsilon))$")
axR.set_title(r"Ratio $\to 1$ as $\varepsilon\to0$"); axR.grid(alpha=.3, which='both')
save("F5_layer_scale.pdf")

print("Saved vector PDFs F1,F2,F4,F5 into", OUT)
