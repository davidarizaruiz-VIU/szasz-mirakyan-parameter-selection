# Companion script for Section 4 (sharp first-order term) and Section 5 (layer robustness).
#
# (4)  D_n(a) := n*( n^2 E_n(a;f)^2 - J(a)/4 )  ->  G(a) = 2 * int_0^X L_a[f] Phi_a[f] w,
#      with L_a[f]=(x/2)(f''-2a f') and Phi_a[f] the second-order operator of Lemma 4.x.
#      (This also numerically validates the Phi_a formula.)
#
# (5)  For f_eps=(1+x)e^{-x/eps} and the scale-matched a=-1/(2eps):
#         R_{n,a} f_eps - f_eps = x e^{-x/eps} * phi(1/(n eps)),   phi(b)=b/(e^b-1)-1,
#      hence  ||.||_2 / ||f_eps||_2  <=  1/(2 sqrt(2) n)   uniformly in eps,
#      whereas the classical operator (a=0) has relative error bounded away from 0 as eps->0.

import numpy as np
X = 10.0
xs = np.linspace(0.0, X, 80001)
def tz(y): return np.trapezoid(y, xs) if hasattr(np, "trapezoid") else np.trapz(y, xs)
def c_na(n, a): return 1.0 if a == 0 else 2*a/(n*np.expm1(2*a/n))

# ---------- (4) ----------
def Rna_mix(n, a, cs, lams):
    al = c_na(n, a)*xs; out = np.zeros_like(xs)
    for cj, lj in zip(cs, lams): out += cj*np.exp(n*al*np.expm1(-lj/n))
    return out
def fder(cs, lams, k): return sum(cj*((-lj)**k)*np.exp(-lj*xs) for cj, lj in zip(cs, lams))
def En2_mix(n, a, cs, lams):
    r = Rna_mix(n, a, cs, lams) - fder(cs, lams, 0); return tz(r*r)
def La(a, cs, lams):  return 0.5*xs*(fder(cs,lams,2) - 2*a*fder(cs,lams,1))
def Phi(a, cs, lams):
    f1,f2,f3,f4 = (fder(cs,lams,k) for k in (1,2,3,4))
    return (a*a*xs/3)*f1 + (a*a*xs*xs/2 - a*xs/2)*f2 + (xs/6 - a*xs*xs/2)*f3 + (xs*xs/8)*f4
def quarterJ(a, cs, lams): return tz(La(a,cs,lams)**2)
def G(a, cs, lams):        return 2*tz(La(a,cs,lams)*Phi(a,cs,lams))

print("==== (4) D_n(a) -> G(a) ====")
for name,(cs,lams) in {"f3":([0.7,0.3],[1,3]), "f4":([1,1],[1,4])}.items():
    astar = tz(xs*xs*fder(cs,lams,1)*fder(cs,lams,2))/(2*tz(xs*xs*fder(cs,lams,1)**2))
    for a,tag in [(0.0,"a=0"), (-0.5,"a=-1/2"), (astar,"a=a*")]:
        Gp, qJ = G(a,cs,lams), quarterJ(a,cs,lams)
        Dn = [n*(n*n*En2_mix(n,a,cs,lams) - qJ) for n in (320,1280,2560)]
        print(f"  {name} {tag:7s}: G={Gp:+.5f}   D_n(320,1280,2560)="
              f"[{Dn[0]:+.5f},{Dn[1]:+.5f},{Dn[2]:+.5f}]")

# ---------- (5) ----------
def phi(b): return b/np.expm1(b) - 1.0
def Rna_player(n, a, lam):
    al = c_na(n,a)*xs; mu = n*al; z = np.exp(-lam/n)
    return np.exp(mu*(z-1))*(1 + al*z)
def fplayer(lam): return (1+xs)*np.exp(-lam*xs)
print("\n==== (5) relative L2 error: matched a=-1/(2eps) vs classical a=0 ====")
print("    uniform bound 1/(2*sqrt(2)*n):  n=20 ->%.4f  n=80 ->%.4f" %
      (1/(2*np.sqrt(2)*20), 1/(2*np.sqrt(2)*80)))
for n in (20, 80):
    print(f"  n={n}:  eps : rel_matched (closed form) | rel_classical")
    for eps in (0.5, 0.1, 0.05, 0.02, 0.01):
        lam = 1/eps; nrm = np.sqrt(tz(fplayer(lam)**2))
        rm = np.sqrt(tz((Rna_player(n,-lam/2,lam)-fplayer(lam))**2))/nrm
        rc = np.sqrt(tz((Rna_player(n,0.0,lam)-fplayer(lam))**2))/nrm
        cf = np.sqrt(tz((xs*np.exp(-lam*xs)*phi(lam/n))**2))/nrm
        print(f"     {eps:5.3f} (n*eps={n*eps:5.2f}) : {rm:.4e} ({cf:.4e}) | {rc:.4e}")
