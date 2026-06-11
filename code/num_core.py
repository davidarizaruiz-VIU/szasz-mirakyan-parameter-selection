"""
Numerical verification for parameter selection in the exponential-preserving
Szász-Mirakyan family  R_{n,a}(f;x) = S_n(f; alpha_{n,a}(x)),
alpha_{n,a}(x) = 2 a x / ( n (e^{2a/n}-1) ).

Goal: verify  n^2 E_n(a;f)^2 ~ (1/4) J(a)  and  a_n^* -> a^*.

Family and Voronovskaya are KNOWN (Acar-Aral-Gonska 2017; Aral-Inoan-Rasa 2019).
This script only checks the parameter-selection claim numerically.

Reproducible: fixed mesh, fixed n-grid, stable log-space Poisson kernel.
Run:  python3 num_core.py <fi>   with fi in {1,2,3,4}; writes results_f<fi>.npz + prints a CSV row block.
"""
import sys, json
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson

X = 10.0
NX = 2001
NS = [20, 40, 80, 160, 320]
xs = np.linspace(0.0, X, NX)

# ---- test functions: f, f', f'' (analytic), plus a-range and label ----
def make_fun(fi):
    if fi == 1:   # e^{-2x}, a*=-1
        f   = lambda x: np.exp(-2*x)
        fp  = lambda x: -2*np.exp(-2*x)
        fpp = lambda x: 4*np.exp(-2*x)
        return f, fp, fpp, (-2.5, 0.5), "f1 = e^{-2x}"
    if fi == 2:   # e^{-x/0.1}=e^{-10x}, a*=-5
        f   = lambda x: np.exp(-10*x)
        fp  = lambda x: -10*np.exp(-10*x)
        fpp = lambda x: 100*np.exp(-10*x)
        return f, fp, fpp, (-9.0, -1.0), "f2 = e^{-10x} (layer eps=0.1)"
    if fi == 3:   # 0.7 e^{-x}+0.3 e^{-3x}
        f   = lambda x: 0.7*np.exp(-x)+0.3*np.exp(-3*x)
        fp  = lambda x: -0.7*np.exp(-x)-0.9*np.exp(-3*x)
        fpp = lambda x: 0.7*np.exp(-x)+2.7*np.exp(-3*x)
        return f, fp, fpp, (-2.0, 0.3), "f3 = 0.7 e^{-x}+0.3 e^{-3x}"
    if fi == 4:   # e^{-x}+e^{-4x}
        f   = lambda x: np.exp(-x)+np.exp(-4*x)
        fp  = lambda x: -np.exp(-x)-4*np.exp(-4*x)
        fpp = lambda x: np.exp(-x)+16*np.exp(-4*x)
        return f, fp, fpp, (-3.0, 0.3), "f4 = e^{-x}+e^{-4x}"
    raise ValueError(fi)

def cfac(n, a):
    """alpha_{n,a}(x) = c(n,a) * x with c = 2a/(n*expm1(2a/n)); c->1 as a->0."""
    if abs(a) < 1e-12:
        return 1.0
    return 2*a/(n*np.expm1(2*a/n))

def Sn(f, n, a, L=13):
    """Stable evaluation of S_n(f; alpha) on the whole mesh xs, alpha=c*xs."""
    c = cfac(n, a)
    alpha = c*xs
    mu = n*alpha
    mumax = float(mu.max())
    kmax = int(np.ceil(mumax + L*np.sqrt(mumax+1.0) + 5))
    k = np.arange(0, kmax+1)
    fk = f(k/n)
    with np.errstate(divide='ignore'):
        logmu = np.log(np.where(mu > 0, mu, 1.0))
    lp = -mu[:, None] + k[None, :]*logmu[:, None] - gammaln(k+1)[None, :]
    P = np.exp(lp)
    zero = (mu == 0)
    if np.any(zero):
        P[zero, :] = 0.0
        P[zero, 0] = 1.0
    return P @ fk

def En2(f, n, a):
    """E_n(a;f)^2 = int_0^X |R_{n,a}f - f|^2 dx  (w=1)."""
    S = Sn(f, n, a)
    return float(simpson((S - f(xs))**2, xs))

def Jfun(fp, fpp, a):
    integ = xs**2 * (fpp(xs) - 2*a*fp(xs))**2
    return float(simpson(integ, xs))

def a_star(fp, fpp):
    Q = simpson(xs**2 * fp(xs)*fpp(xs), xs)
    R = simpson(xs**2 * fp(xs)**2, xs)
    return float(Q/(2*R)), float(Q), float(R)

def main(fi):
    f, fp, fpp, (alo, ahi), label = make_fun(fi)
    ast, Q, R = a_star(fp, fpp)
    agrid = np.linspace(alo, ahi, 61)
    Jgrid = np.array([Jfun(fp, fpp, a) for a in agrid])

    rows = []
    En2_curves = {}
    an_stars = {}
    for n in NS:
        En2_grid = np.array([En2(f, n, a) for a in agrid])
        En2_curves[n] = En2_grid
        res = minimize_scalar(lambda a: En2(f, n, a), bounds=(alo, ahi), method='bounded',
                              options={'xatol': 1e-4})
        an = float(res.x)
        an_stars[n] = an
        relerr = abs(an - ast)/(1+abs(ast))
        En_0  = np.sqrt(En2(f, n, 0.0))
        En_h  = np.sqrt(En2(f, n, -0.5))
        En_as = np.sqrt(En2(f, n, ast))
        rows.append(dict(n=n, a_star=ast, a_n_star=an, relerr=relerr,
                         En0=En_0, En_m05=En_h, En_astar=En_as,
                         ratio0_astar=En_0/En_as))
    # save
    np.savez(f"results_f{fi}.npz", label=label, alo=alo, ahi=ahi, a_star=ast, Q=Q, R=R,
             agrid=agrid, Jgrid=Jgrid, NS=np.array(NS),
             **{f"En2_{n}": En2_curves[n] for n in NS},
             an_stars=np.array([an_stars[n] for n in NS]))
    print(f"=== {label} ===  a* = {ast:.6f}  (Q={Q:.4e}, R={R:.4e})")
    print(f"{'n':>5} {'a_n*':>10} {'relerr':>10} {'E_n(0)':>11} {'E_n(-1/2)':>11} {'E_n(a*)':>11} {'E0/Ea*':>9}")
    for r in rows:
        print(f"{r['n']:>5} {r['a_n_star']:>10.5f} {r['relerr']:>10.3e} {r['En0']:>11.4e} {r['En_m05']:>11.4e} {r['En_astar']:>11.4e} {r['ratio0_astar']:>9.3f}")
    json.dump(rows, open(f"rows_f{fi}.json","w"), indent=1)

if __name__ == "__main__":
    main(int(sys.argv[1]))
