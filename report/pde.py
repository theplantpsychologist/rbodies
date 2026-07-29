"""
Numerical solution of the friction-coupled stress system and F(x) plot.

Reproduces the following symbolic MATLAB model for numerical parameters:

    R  = -((L^2-x^2)^(3/2) + theta*x^2) / (x*(theta*sqrt(L^2-x^2) - L^2 + x^2))
    p  = ((L^2-x^2)*sigma_ss - 2*x*sqrt(L^2-x^2)*sigma_st) / L^2
    ft = mu*p / (1 + 1/R^2)
    fs = mu*p / (1 + R^2)

    eq1: d(sigma_st)/ds            = -ft
    eq2: d(sigma_ss)/ds + d(sigma_st)/dt = -fs

    F(x) = int_{-W/2}^{W/2} [ sigma_ss(L,t,x)*sin(phi) + sigma_st(L,t,x)*cos(phi) ] dt
           where sin(phi) = x/L, cos(phi) = sqrt(1-(x/L)^2)

ASSUMPTIONS MADE TO CLOSE THE PROBLEM (the original snippet references
sigma_ss_sol / sigma_st_sol without showing how eq1/eq2 were solved --
i.e. the boundary conditions were not given):

  1. No t-dependence in the solution. R, ft, fs are algebraic in
     sigma_ss, sigma_st (no derivatives on the RHS), and the final F(x)
     integral over t is a plain definite integral of a t-independent
     integrand in the pasted code (it just multiplies by W). That is
     only consistent if d(sigma_st)/dt = 0, which also makes eq2 a
     clean ODE in s. This reduces the "PDE" to a constant-coefficient
     linear ODE system in s (for each fixed x):

         d/ds [sigma_ss; sigma_st] = M(x) @ [sigma_ss; sigma_st]

  2. Boundary/initial condition at s=0 (entry to the contact):
         sigma_ss(0,x) = 1   (normalized applied stress/tension)
         sigma_st(0,x) = 0   (no initial shear)
     Because the system is linear and homogeneous in the state, this
     choice only sets an overall scale factor on F(x); the *shape* of
     the semilogy curve is independent of it. If you have the actual
     boundary condition from the original derivation, rescale
     accordingly (F scales linearly with sigma_ss(0,x)).

R(x) itself has removable singularities (denominator -> 0 while
numerator stays finite) at x=0, x=L, and x=sqrt(L^2-theta^2). To avoid
0/0 or inf/inf, kt = mu/(1+1/R^2) and ks = mu/(1+R^2) are rewritten
directly in terms of R's numerator/denominator (N_R, D_R):

    kt = mu * N_R^2 / (N_R^2 + D_R^2)
    ks = mu * D_R^2 / (N_R^2 + D_R^2)

which are bounded in [0, mu] everywhere and singularity-free.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# Numerical parameters
# ----------------------------
L = 200.0      # Total length of the strip (cm)
r0 = 2.0
W = 10.0
theta = L / r0
mu = 0.3       # Coefficient of friction


def coeffs(x, L, theta, mu):
    """Stable kt, ks, c1, c2 for a given x (no 0/0 from forming R directly)."""
    Lx2 = L**2 - x**2
    sq = np.sqrt(max(Lx2, 0.0))
    N_R = Lx2**1.5 + theta * x**2
    D_R = x * (theta * sq - L**2 + x**2)
    denom = N_R**2 + D_R**2
    if denom == 0:
        kt, ks = mu, 0.0
    else:
        kt = mu * N_R**2 / denom
        ks = mu * D_R**2 / denom
    c1 = Lx2 / L**2
    c2 = 2 * x * sq / L**2
    return kt, ks, c1, c2


def solve_F(x, L, W, theta, mu):
    """Solve the ODE system in s (closed form via matrix exponential) for one x,
    then evaluate the width integral to get F(x)."""
    kt, ks, c1, c2 = coeffs(x, L, theta, mu)

    # d/ds [sigma_ss; sigma_st] = M @ [sigma_ss; sigma_st]
    M = np.array([[-ks * c1,  ks * c2],
                  [-kt * c1,  kt * c2]])

    y0 = np.array([1.0, 0.0])          # sigma_ss(0)=1, sigma_st(0)=0
    sigma_ss_L, sigma_st_L = expm(M * L) @ y0

    sinphi = x / L
    cosphi = np.sqrt(max(1 - (x / L) ** 2, 0.0))

    integrand = sigma_ss_L * sinphi + sigma_st_L * cosphi  # constant in t
    F = integrand * W                                       # int_{-W/2}^{W/2} dt
    return F, sigma_ss_L, sigma_st_L


# ----------------------------
# Sweep over x and plot
# ----------------------------
N = 4000
xs = np.linspace(0.0, L, N)
Fs = np.array([solve_F(x, L, W, theta, mu)[0] for x in xs])

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.semilogy(xs, np.abs(Fs), lw=1.8, color="#B03A2E")
ax.set_xlabel("x  (cm)")
ax.set_ylabel(r"$|F(x)|$")
ax.set_title(r"$|F(x)|$  (F is negative over nearly the whole domain;"
             "\nsign flips to positive only very close to x = L)")
ax.grid(True, which="both", ls=":", alpha=0.6)
fig.tight_layout()
fig.savefig("F_semilogy.png", dpi=150)
print("Saved plot. F(0) =", Fs[0], " F(L) =", Fs[-1], " max|F| =", np.max(np.abs(Fs)),
      " at x =", xs[np.argmax(np.abs(Fs))])