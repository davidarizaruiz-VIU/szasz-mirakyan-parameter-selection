# Companion script for Section 6:
#   (i) asymptotic gain over the classical operator,  E_n(0)/E_n(a*) -> 1/sqrt(1-rho^2);
#   (ii) like-for-like comparison with the mixed operator of Ulusoy Ada & Aral (2026),
#        L_n = theta S_n^{(1)} + (1-theta) S_n^{(2)}.
#
# Exponential-preserving Szasz operator S_n^{(c)} (reproduces e^{-c t}), c=-2a:
#   S_n^{(c)}(g;x) = E[g(K/n)],  K ~ Poisson(mu),  mu = n*alpha,  alpha = c x/(n(1-e^{-c/n})),
#   c = 0 -> classical Szasz (alpha = x).
# Exact closed forms (no truncation):
#   S_n^{(c)}(e^{-lam t};x)      = exp(mu(z-1)),              z = e^{-lam/n}
#   S_n^{(c)}((1+t)e^{-lam t};x) = exp(mu(z-1))*(1 + alpha*z)

import numpy as np
from scipy.optimize import minimize_scalar

X = 10.0
xs = np.linspace(0.0, X, 40001)

def trapz(y): return np.trapezoid(y, xs) if hasattr(np, "trapezoid") else np.trapz(y, xs)

def alpha_c(n, c, x):
    return x.copy() if c == 0.0 else c * x / (n * (1.0 - np.exp(-c / n)))

def Sc_mix(n, c, cs, lams, x):
    mu = n * alpha_c(n, c, x); out = np.zeros_like(x)
    for cj, lj in zip(cs, lams):
        out += cj * np.exp(mu * (np.exp(-lj / n) - 1.0))
    return out

def Sc_player(n, c, lam, x):
    al = alpha_c(n, c, x); mu = n * al; z = np.exp(-lam / n)
    return np.exp(mu * (z - 1.0)) * (1.0 + al * z)

def f_mix(cs, lams, x): return sum(cj*np.exp(-lj*x) for cj, lj in zip(cs, lams))
def f_player(lam, x):   return (1.0 + x) * np.exp(-lam * x)

def En_ours_mix(n, a, cs, lams):
    r = Sc_mix(n, -2*a, cs, lams, xs) - f_mix(cs, lams, xs); return np.sqrt(trapz(r*r))
def En_Ln_mix(n, t, cs, lams):
    r = t*Sc_mix(n,1.,cs,lams,xs) + (1-t)*Sc_mix(n,2.,cs,lams,xs) - f_mix(cs,lams,xs)
    return np.sqrt(trapz(r*r))
def En_ours_player(n, a, lam):
    r = Sc_player(n, -2*a, lam, xs) - f_player(lam, xs); return np.sqrt(trapz(r*r))
def En_Ln_player(n, t, lam):
    r = t*Sc_player(n,1.,lam,xs) + (1-t)*Sc_player(n,2.,lam,xs) - f_player(lam, xs)
    return np.sqrt(trapz(r*r))

def I2(s): return 2/s**3 - np.exp(-s*X)*(X**2/s + 2*X/s**2 + 2/s**3)

def rho2_gain(cs, lams):                      # weighted correlation and asymptotic gain
    cs, lams = np.array(cs), np.array(lams); P=Q=R=0.0
    for i in range(len(cs)):
        for j in range(len(cs)):
            M = I2(lams[i]+lams[j])
            P += cs[i]*cs[j]*lams[i]**2*lams[j]**2*M
            Q += -cs[i]*cs[j]*lams[i]*lams[j]**2*M
            R += cs[i]*cs[j]*lams[i]*lams[j]*M
    r2 = Q*Q/(P*R); return r2, 1.0/np.sqrt(1.0-r2)

def compare(name, En_ours, En_Ln, n=320):
    a  = minimize_scalar(lambda a: En_ours(n,a), bounds=(-20,0.5), method="bounded").x
    th = minimize_scalar(lambda t: En_Ln(n,t),  bounds=(0,1),     method="bounded").x
    E0, Eo, EL = En_ours(n,0.), En_ours(n,a), En_Ln(n,th)
    print(f"{name}  (n={n})")
    print(f"   our a*={a:+.4f}   their theta*={th:.4f} (eff. a={-(1-th/2):+.4f}; proj on [-1,-1/2]={min(-0.5,max(-1.,a)):+.4f})")
    print(f"   gain vs classical:  ours {E0/Eo:7.2f}x    L_n {E0/EL:7.2f}x    ours/L_n {EL/Eo:7.2f}x")

if __name__ == "__main__":
    for nm,(cs,lams) in {"f3=0.7e^-x+0.3e^-3x":([0.7,0.3],[1,3]),
                         "f4=e^-x+e^-4x":([1,1],[1,4])}.items():
        r2,g = rho2_gain(cs,lams)
        print(f"[gain] {nm}:  rho^2={r2:.4f}  ->  1/sqrt(1-rho^2)={g:.3f}")
    print()
    compare("f3 = 0.7 e^-x + 0.3 e^-3x", lambda n,a: En_ours_mix(n,a,[0.7,0.3],[1,3]),
                                         lambda n,t: En_Ln_mix(n,t,[0.7,0.3],[1,3]))
    compare("f4 = e^-x + e^-4x",         lambda n,a: En_ours_mix(n,a,[1,1],[1,4]),
                                         lambda n,t: En_Ln_mix(n,t,[1,1],[1,4]))
    for eps in [0.25, 0.1, 0.05]:
        lam = 1.0/eps
        compare(f"perturbed layer (1+x)e^(-x/{eps})",
                lambda n,a,lam=lam: En_ours_player(n,a,lam),
                lambda n,t,lam=lam: En_Ln_player(n,t,lam))
