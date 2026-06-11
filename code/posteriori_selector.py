# A posteriori selector for the exponential-preserving Szasz-Mirakyan operators.
#
# Reproduces the values reported in Section 6 (paragraph "A posteriori selector")
# and verifies Proposition 4.x: the vertex of the parabola interpolating
#   phi_n(a) = n^2 E_n(a;f)^2
# at three FIXED nodes converges to a* = Q/(2R) with order O(1/n), and the fitted
# leading coefficient A_n estimates the sensitivity constant R.
#
# For mixtures f(t) = sum_j c_j e^{-lam_j t} the operator has the exact closed form
#   S_n(e^{-lam t}; y) = exp( n y (e^{-lam/n} - 1) ),  R_{n,a}(f;x) = S_n(f; c_{n,a} x),
# so no Poisson truncation is needed.

import numpy as np

X = 10.0
xs = np.linspace(0.0, X, 8001)          # trapezoidal quadrature mesh, w == 1


def trapz(y):
    return np.trapezoid(y, xs) if hasattr(np, "trapezoid") else np.trapz(y, xs)


def c_na(n, a):
    return 1.0 if a == 0.0 else 2.0 * a / (n * np.expm1(2.0 * a / n))


def Rna_mix(n, a, cs, lams, x):
    alpha = c_na(n, a) * x
    out = np.zeros_like(x)
    for cj, lj in zip(cs, lams):
        out += cj * np.exp(n * alpha * np.expm1(-lj / n))   # exact closed form
    return out


def f_mix(cs, lams, x):
    return sum(cj * np.exp(-lj * x) for cj, lj in zip(cs, lams))


def En2(n, a, cs, lams):
    r = Rna_mix(n, a, cs, lams, xs) - f_mix(cs, lams, xs)
    return trapz(r * r)


def I2(s):                               # int_0^X x^2 e^{-s x} dx
    return 2.0 / s**3 - np.exp(-s * X) * (X**2 / s + 2 * X / s**2 + 2.0 / s**3)


def a_star_exact(cs, lams):
    cs, lams = np.array(cs), np.array(lams)
    Q = R = 0.0
    for i in range(len(cs)):
        for j in range(len(cs)):
            M = I2(lams[i] + lams[j])
            Q += -cs[i] * cs[j] * lams[i] * lams[j] ** 2 * M
            R += cs[i] * cs[j] * lams[i] * lams[j] * M
    return Q / (2 * R), R


def selector(n, nodes, cs, lams):
    phi = np.array([n * n * En2(n, a, cs, lams) for a in nodes])
    A, B, C = np.polyfit(nodes, phi, 2)
    return -B / (2 * A), A            # vertex (a_hat) and curvature (estimates R)


if __name__ == "__main__":
    cases = {
        "f3 = 0.7 e^-x + 0.3 e^-3x": ([0.7, 0.3], [1.0, 3.0]),
        "f4 = e^-x + e^-4x":         ([1.0, 1.0], [1.0, 4.0]),
    }
    nodes = [-2.0, -1.0, 0.0]            # fixed, no prior knowledge of a*
    ns = [20, 40, 80, 160, 320]
    for name, (cs, lams) in cases.items():
        astar, R = a_star_exact(cs, lams)
        print(f"{name}:  a* = {astar:.6f},  R = {R:.4f}  (sensitivity 4R = {4*R:.4f})")
        print(f"  {'n':>4} {'a_hat':>12} {'|a_hat-a*|':>12} {'ratio':>6} {'A_hat(~R)':>10}")
        prev = None
        for n in ns:
            ahat, Ahat = selector(n, nodes, cs, lams)
            err = abs(ahat - astar)
            ratio = (prev / err) if prev else float("nan")
            print(f"  {n:>4} {ahat:>12.6f} {err:>12.3e} {ratio:>6.2f} {Ahat:>10.4f}")
            prev = err
        print()
