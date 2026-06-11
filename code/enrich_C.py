"""Controlled-improvement test, Part C: perturbed boundary layers f_eps=(1+x) e^{-x/eps}.
Not reproduced exactly by any R_{n,a}; checks a*_eps ~ -1/(2eps), a_{n,eps}*, and error ratios.
Also Part B numeric cross-check (exponential weight) and Part A restricted optimum.
Stable log-space Poisson kernel. Graded mesh resolving the layer; weight w=1 on [0,X], X=10."""
import numpy as np, json
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

X=10.0
# graded mesh: dense near 0 to resolve sharp layers
xs=np.unique(np.concatenate([np.linspace(0,1,1200,endpoint=False), np.linspace(1,X,601)]))

def cfac(n,a):
    return 1.0 if abs(a)<1e-12 else 2*a/(n*np.expm1(2*a/n))

def Sn(fk_of_t, n, a, L=13):
    c=cfac(n,a); alpha=c*xs; mu=n*alpha; mmax=float(mu.max())
    kmax=int(np.ceil(mmax+L*np.sqrt(mmax+1)+5)); k=np.arange(0,kmax+1)
    fk=fk_of_t(k/n)
    with np.errstate(divide='ignore'):
        logmu=np.log(np.where(mu>0,mu,1.0))
    P=np.exp(-mu[:,None]+k[None,:]*logmu[:,None]-gammaln(k+1)[None,:])
    z=(mu==0)
    if np.any(z): P[z,:]=0.0; P[z,0]=1.0
    return P@fk

def En(fk, fxs, n, a):
    return float(np.sqrt(np.trapz((Sn(fk,n,a)-fxs)**2, xs)))

# ---------- Part C ----------
def make_layer(lam):
    f  = lambda x:(1+x)*np.exp(-lam*x)
    fp = lambda x:(1-lam*(1+x))*np.exp(-lam*x)
    fpp= lambda x:lam*(lam*(1+x)-2)*np.exp(-lam*x)
    return f,fp,fpp

def astar_finite(fp,fpp):
    Q,_=quad(lambda x:x**2*fp(x)*fpp(x),0,X,limit=200)
    R,_=quad(lambda x:x**2*fp(x)**2,0,X,limit=200)
    return Q/(2*R)

def astar_closed(lam):  # X=inf, w=1
    return -(lam*(2*lam**2+1))/(4*(lam**2+lam+1))

eps_list=[1.0,0.5,0.25,0.1,0.05,0.025]
NS=[160,320]
rowsC=[]
for eps in eps_list:
    lam=1.0/eps
    f,fp,fpp=make_layer(lam)
    fxs=f(xs)
    ast=astar_finite(fp,fpp); astc=astar_closed(lam); scale=-1.0/(2*eps)
    lo=-(lam+2.0); hi=0.3
    rec=dict(eps=eps, lam=lam, a_star=ast, a_star_closedXinf=astc, neg_half_over_eps=scale,
             ratio_astar_to_scale=ast/scale)
    for n in NS:
        res=minimize_scalar(lambda a:En(f,fxs,n,a), bounds=(lo,hi), method='bounded', options={'xatol':1e-4})
        an=float(res.x)
        E0=En(f,fxs,n,0.0); Eh=En(f,fxs,n,-0.5); Eas=En(f,fxs,n,ast)
        aK=min(0.0,max(-5.0,ast))   # restricted to K=[-5,0]
        EK=En(f,fxs,n,aK)
        rec[f'an_{n}']=an
        rec[f'E0_{n}']=E0; rec[f'Eh_{n}']=Eh; rec[f'Eas_{n}']=Eas
        rec[f'aK_{n}']=aK; rec[f'EK_{n}']=EK
        rec[f'r0_{n}']=E0/Eas; rec[f'rh_{n}']=Eh/Eas
    rowsC.append(rec)
    print(f"eps={eps:5.3f} lam={lam:4.1f} a*={ast:8.4f} (Xinf {astc:8.4f}) -1/(2eps)={scale:8.4f} a*/scale={ast/scale:.4f} "
          f"a*_320={rec['an_320']:8.4f} E0/Eas(320)={rec['r0_320']:.3f} Eh/Eas(320)={rec['rh_320']:.3f} aK={rec['aK_320']:.3f}")

json.dump(rowsC, open("enrichC_rows.json","w"), indent=1)

# ---------- Part B numeric cross-check: mixture 0.7 e^{-x}+0.3 e^{-3x}, weight e^{-gamma x} on [0,inf) ----------
print("\n=== Part B: exponential-weight a*_gamma (mixture 0.7 e^{-x}+0.3 e^{-3x}) ===")
c=[0.7,0.3]; lam=[1.0,3.0]
def astar_gamma_formula(gamma):
    M=lambda i,j:2.0/(lam[i]+lam[j]+gamma)**3
    Q=-sum(c[i]*c[j]*lam[i]*lam[j]**2*M(i,j) for i in range(2) for j in range(2))
    R= sum(c[i]*c[j]*lam[i]*lam[j]   *M(i,j) for i in range(2) for j in range(2))
    return Q/(2*R)
def astar_gamma_quad(gamma):
    f1=lambda x:-(c[0]*lam[0]*np.exp(-lam[0]*x)+c[1]*lam[1]*np.exp(-lam[1]*x))
    f2=lambda x: (c[0]*lam[0]**2*np.exp(-lam[0]*x)+c[1]*lam[1]**2*np.exp(-lam[1]*x))
    w=lambda x:np.exp(-gamma*x)
    Q,_=quad(lambda x:x**2*f1(x)*f2(x)*w(x),0,np.inf,limit=200)
    R,_=quad(lambda x:x**2*f1(x)**2*w(x),0,np.inf,limit=200)
    return Q/(2*R)
for g in [0.0,1.0,3.0,10.0,100.0]:
    print(f" gamma={g:6.1f}  a*_gamma(formula)={astar_gamma_formula(g):.5f}  (quad {astar_gamma_quad(g):.5f})")
print(" gamma->inf analytic limit  sum c_i l_i (-l_i/2)/sum c_i l_i =",
      sum(c[i]*lam[i]*(-lam[i]/2) for i in range(2))/sum(c[i]*lam[i] for i in range(2)))
