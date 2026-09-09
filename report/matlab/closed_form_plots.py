"""
closed_form_plots.py -- plots of the closed-form solution built by
energy_closed_form.m:  the minimising radius, the energy landscape, the residual
strain, F(x), the through-thickness profiles, and a map of where the uniform
interior solution is actually valid.

The closed form is
    U/(E L W t) = tau^2/(24(1-nu^2)) [ (1/rho-1)^2 + 2(1-nu) chi^2/(rho lam^2)
                                       + Gamma (lam-1)^2 ],
which is identical to the two-mode energy of the report's Coupling I section --
that identity is the main result, not a coincidence.  Writes two PDFs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import brentq

OUT = "../figs/"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

L, W, t, thL, E, nu = 200.0, 8.0, 0.01, 100.0, 3.0e5, 0.3
rnat = L / thL
D    = E * t**3 / (12 * (1 - nu**2))
tau  = t / rnat
GAM  = 12 * (1 - nu**2) / tau**2
RJ   = t * (thL / (2*np.pi)) / (2 * rnat)          # packing scale
K    = tau**2 / (24 * (1 - nu**2))                 # common prefactor


def U_terms(rho, chi):
    """The three terms of U/(E L W t)."""
    lam = np.hypot(rho, chi)
    return (K*(1/rho - 1)**2, K*2*(1-nu)*chi**2/(rho*lam**2), K*GAM*(lam-1)**2)


def U(rho, chi):
    return sum(U_terms(rho, chi))


def rho_min(chi):
    """dU/drho = 0.  Solved in eps and converted, since eps = lam-1 is a
    difference of nearly equal quantities and is badly conditioned in rho."""
    f = lambda e: U(np.sqrt(max((1+e)**2 - chi**2, 1e-300)), chi)
    d = lambda e: (f(e*np.exp(-2e-3)) - 8*f(e*np.exp(-1e-3))
                   + 8*f(e*np.exp(1e-3)) - f(e*np.exp(2e-3))) / (12e-3*e)
    g = np.geomspace(1e-13, (np.sqrt(1+chi**2)-1)*0.99, 500)
    v = np.array([d(q) for q in g])
    k = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    eps = brentq(d, g[k[-1]], g[k[-1]+1], xtol=1e-20, rtol=8.9e-16)
    return np.sqrt((1+eps)**2 - chi**2), eps


def F_terms(rho, chi):
    """F = dU/dx at fixed rho (envelope theorem), in newtons, term by term."""
    lam = np.hypot(rho, chi)
    twist = E*W*t*K*4*(1-nu)*chi*rho/lam**4
    memb  = E*W*t*K*2*GAM*(lam-1)*chi/lam
    return twist, memb


def profiles(rho, chi):
    """Through-thickness fields of the minimising state, at zeta/t in [-1/2,1/2]."""
    r, lam = rho*rnat, np.hypot(rho, chi)
    s2, c2, sc = chi**2/lam**2, rho**2/lam**2, chi*rho/lam**2
    kuu, kuv, kvv = c2/r, sc/r, s2/r
    Dkuu, Dkuv, Dkvv = kuu - 1/rnat, kuv, kvv
    Muu = D*(Dkuu + nu*Dkvv); Mvv = D*(Dkvv + nu*Dkuu); Muv = D*(1-nu)*Dkuv
    P   = 12/t**3 * (kuu*Muu - 2*kuv*Muv + kvv*Mvv)   # r-moment of r_eq fixes P
    z   = np.linspace(-t/2, t/2, 201)
    srr = P/2*(z**2 - t**2/4)
    euu = lam - 1
    evv = -nu*euu - nu*(1+nu)*(-P*t**2/12)/E           # free edge, resultant form
    Euu, Evv = euu + z*Dkuu, evv + z*Dkvv
    suu = E*(Euu + nu*Evv)/(1-nu**2) + nu*srr/(1-nu)
    svv = E*(Evv + nu*Euu)/(1-nu**2) + nu*srr/(1-nu)
    return z/t, suu, svv, srr, P


if __name__ == "__main__":
    ch = np.linspace(0.02, 0.995, 400)
    rm, em = np.array([rho_min(c) for c in ch]).T
    tw, mb = F_terms(rm, ch)

    # ---------------- figure 1: the solution ----------------
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 6.6))

    a = ax[0, 0]
    a.plot(ch, rm, "k-", lw=2, label=r"$\rho_*(\chi)$, closed form")
    a.plot(ch, np.sqrt(np.maximum(1-ch**2, 0)), "--", color="0.55", lw=1.4,
           label=r"inextensible, $\sqrt{1-\chi^2}$")
    a.axhline(RJ, color="crimson", ls=":", lw=1.4,
              label=rf"$\rho_{{\rm jam}}={RJ:.3f}$")
    a.set_xlabel(r"$x/L$"); a.set_ylabel(r"$\rho = r/r_{\rm nat}$")
    a.set_ylim(0, 1.03); a.legend(fontsize=7.5)
    a.set_title("(a) minimising radius", fontsize=9)

    b = ax[0, 1]
    rr = np.geomspace(0.05, 1.4, 600)
    for c, col in zip([0.2, 0.5, 0.8, 0.95, 0.99],
                      ["0.75", "0.6", "0.42", "0.22", "k"]):
        ok = rr < np.sqrt(1 + c**2)
        b.semilogy(rr[ok], U(rr[ok], c), "-", color=col, lw=1.5,
                   label=rf"$\chi={c}$")
        r0, _ = rho_min(c)
        b.plot([r0], [U(r0, c)], "o", color=col, ms=5, mec="crimson", mew=1.3)
    b.set_xlabel(r"$\rho$"); b.set_ylabel(r"$U/(E\,L\,W\,t)$")
    b.set_xlim(0, 1.4); b.legend(fontsize=7.5, loc="upper right")
    b.set_title("(b) energy landscape at frozen $\\chi$ (minima circled)",
                fontsize=9)

    c_ = ax[1, 0]
    c_.semilogy(ch, em, "k-", lw=2, label=r"$\epsilon(\chi)=\lambda-1$")
    c_.axhline(1e-4, color="0.7", lw=0.8, ls="--")
    c_.text(0.04, 1.25e-4, r"$10^{-4}$", fontsize=7, color="0.45")
    xstar = 1 - 2**(-4/3)*GAM**(-1/3)
    c_.axvline(xstar, color="crimson", ls=":", lw=1.4,
               label=rf"$x_*/L={xstar:.3f}$ (crossover)")
    c_.set_xlabel(r"$x/L$"); c_.set_ylabel(r"$\epsilon$")
    c_.legend(fontsize=7.5, loc="upper left")
    c_.set_title("(c) residual membrane strain", fontsize=9)

    d = ax[1, 1]
    d.semilogy(ch, tw + mb, "k-", lw=2, label=r"$F$ total")
    d.semilogy(ch, tw, "--", color="darkgreen", lw=1.5, label="twist bending")
    d.semilogy(ch, mb, "-.", color="crimson", lw=1.5, label="membrane")
    i = np.argmin(np.abs(tw - mb))
    d.plot([ch[i]], [tw[i]], "k*", ms=11)
    d.annotate(rf"terms cross, $x/L={ch[i]:.2f}$", xy=(ch[i], tw[i]),
               xytext=(0.30, 3e-3), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
    d.set_xlabel(r"$x/L$"); d.set_ylabel(r"$F$  [N]")
    d.legend(fontsize=7.5, loc="upper left")
    d.set_title("(d) force, friction-free limit", fontsize=9)

    fig.tight_layout(); fig.savefig(OUT + "closed_form_solution.pdf"); plt.close(fig)
    print("wrote closed_form_solution.pdf")

    # ---------------- figure 2: fields and validity ----------------
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))
    cols = ["0.72", "0.5", "0.28", "k"]
    xs   = [0.4, 0.8, 0.95, 0.99]

    a = ax[0]
    for c, col in zip(xs, cols):
        r0, e0 = rho_min(c); zt, suu, svv, srr, P = profiles(r0, c)
        pk = np.max(np.abs(suu))
        a.plot(suu/pk, zt, "-", color=col, lw=1.7,
               label=rf"$x/L={c}$:  $\pm${pk:.0f},  memb $\pm${abs(E*e0/pk):.2f}")
    a.axvline(0, color="0.6", lw=0.8)
    a.set_xlabel(r"$\sigma_{uu}(\zeta)\;/\;\max|\sigma_{uu}|$")
    a.set_ylabel(r"$\zeta/t$")
    a.legend(fontsize=6.6, loc="lower left", title="peak [N cm$^{-2}$], membrane share",
             title_fontsize=6.6)
    a.set_title(r"(a) $\sigma_{uu}$: affine; slope is bending, offset is membrane",
                fontsize=9)

    b = ax[1]
    for c, col in zip(xs, cols):
        r0, _ = rho_min(c); zt, suu, svv, srr, P = profiles(r0, c)
        sb_ = np.max(np.abs(suu - suu[::-1]))/2          # bending stress scale
        b.plot(srr/sb_, zt, "-", color=col, lw=1.7,
               label=rf"$x/L={c}$:  {srr[100]:+.2f}")
    b.axvline(0, color="0.6", lw=0.8)
    b.set_xlabel(r"$\sigma_{rr}(\zeta)\;/\;\sigma_{uu}^{\rm bend}(t/2)$")
    b.set_ylabel(r"$\zeta/t$")
    b.legend(fontsize=6.6, loc="center right",
             title=r"$\sigma_{rr}(0)$ [N cm$^{-2}$]", title_fontsize=6.6)
    b.set_title(r"(b) $\sigma_{rr}$: quadratic, zero at both faces", fontsize=9)

    # (c) contact gates and Saint-Venant validity over (u,v)
    cx = 0.8
    r0, _ = rho_min(cx); r = r0*rnat; lam = np.hypot(r0, cx)
    off  = 2*np.pi*r*cx/lam                              # 2 pi r sin(phi)
    uu   = np.linspace(0, L, 600); vv = np.linspace(-0.5, 0.5, 500)*W
    Ug, Vg = np.meshgrid(uu, vv)
    end  = np.exp(-4.212*Ug/W) + np.exp(-4.212*(L-Ug)/W)
    layer = (end > 0.01) | (np.abs(Vg) > W/2 - np.sqrt(r*t))
    n_nb = (Vg <= W/2 - off).astype(int) + (Vg >= -W/2 + off).astype(int)
    code = np.where(layer, 3, 2 - n_nb)      # 0 both, 1 one, 2 none, 3 layer
    cc = ax[2]
    cc.pcolormesh(Ug/L, Vg/W, code, shading="auto", rasterized=True, vmin=0, vmax=3,
                  cmap=matplotlib.colors.ListedColormap(
                      ["#dcecdc", "#f6e3bf", "#efc0c0", "#b9b9b9"]))
    cc.set_xlabel(r"$u/L$"); cc.set_ylabel(r"$v/W$"); cc.grid(False)
    cc.legend(handles=[Patch(facecolor="#dcecdc", label="both neighbours"),
                       Patch(facecolor="#f6e3bf", label="one neighbour"),
                       Patch(facecolor="#efc0c0", label="no neighbour"),
                       Patch(facecolor="#b9b9b9", label="end / free-edge layer")],
              fontsize=6.6, loc="center", framealpha=0.95)
    cc.set_title(rf"(c) contact and validity at $x/L={cx}$,  "
                 rf"$2\pi r\sin\phi/W={off/W:.2f}$", fontsize=8.5)

    fig.tight_layout(); fig.savefig(OUT + "closed_form_fields.pdf"); plt.close(fig)
    print("wrote closed_form_fields.pdf")

    # ---------------- printed summary ----------------
    print(f"\nGamma = {GAM:.4e},  rho_jam = {RJ:.4f},  x_*/L = {xstar:.4f}")
    print(f"{'x/L':>6} {'rho*':>9} {'eps':>11} {'F [N]':>10} {'twist':>10} "
          f"{'memb':>10} {'P':>11} {'srr(0)':>9}")
    for c in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        r0, e0 = rho_min(c); tw0, mb0 = F_terms(r0, c)
        _, _, _, srr, P = profiles(r0, c)
        print(f"{c:6.2f} {r0:9.5f} {e0:11.4e} {tw0+mb0:10.4f} {tw0:10.4f} "
              f"{mb0:10.4f} {P:11.4g} {srr[100]:9.4f}")
    print(f"  F terms cross at x/L = {ch[np.argmin(np.abs(tw-mb))]:.3f}")
    # contact-gate fraction along the pull
    print("\n  fraction of the ribbon with BOTH contact neighbours:")
    for c in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99]:
        r0, _ = rho_min(c); rr = r0*rnat; lam = np.hypot(r0, c)
        off = 2*np.pi*rr*c/lam
        both = max(0.0, (W - 2*off))/W
        none = max(0.0, (2*off - W))/W
        print(f"    x/L={c:4.2f}:  2 pi r sin(phi)/W = {off/W:5.3f},  "
              f"both = {both:5.3f},  at least one = {1-none:5.3f}")
    # exactly one interior stationary point in rho, at every chi?
    def n_stat(c):
        g = np.geomspace(0.05, np.sqrt(1+c**2)*0.999, 4000)
        d = np.diff(U(g, c))
        return len(np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0])
    print("  interior stationary points in rho: " +
          ", ".join(f"chi={c}: {n_stat(c)}" for c in [0.1,0.3,0.5,0.7,0.9,0.99]))
