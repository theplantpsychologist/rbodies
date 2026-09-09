"""
energy_3d.py -- the closed-form elastic energy of the coil, from first principles.

Supports the report's "elastic energy in closed form" section.  The chain is a
ledger of eliminations starting from u_el = (1/2) sigma_ij eps_ij (twelve unknown
fields) and ending with an energy that is a function of the geometry alone:

  normality        -> eps_ur = eps_vr = 0, and sigma_ur, sigma_vr are workless
  thickness ansatz -> fields of (u,v,zeta) become fields of (u,v)
  constitutive law -> sigma_uu, sigma_vv, sigma_uv, eps_rr eliminated
  |zeta| << r      -> curved volume/face measures become du dv dzeta
  moment balance   -> exact shear ODE; Dsigma_rr must be a function of theta only
  homogeneity      -> sigma_ur = sigma_vr = 0 exactly
  equilibrium      -> Dsigma_rr = t sigma_theta / r  (over-determined; bounded)
  free edges + arc length -> the last three strains

What this script computes:
  (a) the two ways of closing the transverse pressure, and the bound each puts on
      the energy and on F -- the quantitative content of the flagged step;
  (b) the three terms of the closed-form energy density along the pull.
Writes ../figs/energy_3d.pdf.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import energy_coupled_solver as m

OUT = "../figs/"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

nu, tau, RJ, GAM = m.nu, m.t / m.r_nat, m.RHO_JAM, m.GAMMA
NTURN = m.theta_L / (2 * np.pi)

GX, GW = np.array([-1, 1]) / (2 * np.sqrt(3)), np.array([0.5, 0.5])  # exact to cubic
_p, _w = np.polynomial.legendre.leggauss(32)
PS, PW = (_p + 1) / 2, _w / 2                       # depth fraction through the stack


def rho_of(eps, chi):
    return np.sqrt(max((1 + eps) ** 2 - chi ** 2, 1e-300))


def curvatures(rho, lam, chi):
    """r_nat * Delta_kappa_ij, from the report's curvature-mismatch equation."""
    return rho / lam**2 - 1, chi / lam**2, chi**2 / (lam**2 * rho)


def U_closed(eps, chi):
    """The closed-form energy per unit reference volume, divided by E.
    Hoop bending + twist bending + membrane, in one nondimensional group."""
    rho, lam = rho_of(eps, chi), 1 + eps
    hoop  = (1 / rho - 1) ** 2
    twist = 2 * (1 - nu) * chi**2 / (rho * lam**2)
    memb  = GAM * eps**2
    return tau**2 * (hoop + twist + memb) / (24 * (1 - nu**2))


def U_terms(eps, chi):
    rho, lam = rho_of(eps, chi), 1 + eps
    k = tau**2 / (24 * (1 - nu**2))
    return (k * (1 / rho - 1) ** 2,
            k * 2 * (1 - nu) * chi**2 / (rho * lam**2),
            k * GAM * eps**2)


def U_with_pressure(eps, chi, per_turn=False):
    """Same energy, but carrying a transverse pressure through the stack.
    per_turn=False: the recursion-informed closure, |sigma_rr| up to Pi*sig_uu.
    per_turn=True : the homogeneous closure, only the single-layer drop Pi/n."""
    rho, lam = rho_of(eps, chi), 1 + eps
    kuu, kuv, kvv = curvatures(rho, lam, chi)
    Pi  = 2 * RJ * rho / lam**2
    if per_turn:
        Pi = Pi / NTURN
    suu = eps / (1 + nu * Pi / 2)
    ds  = tau * rho / lam**2 * suu
    U = 0.0
    for psi, pw in zip(PS, PW):
        sb  = -psi * Pi * suu
        euu = suu - nu * sb
        evv = -nu * euu - nu * (1 + nu) * sb        # free long edge, resultant form
        for x, w in zip(GX, GW):
            Euu, Evv, Euv = euu + tau*x*kuu, evv + tau*x*kvv, tau*x*kuv
            s = sb + x * ds
            Suu = (Euu + nu*Evv) / (1-nu**2) + nu*s / (1-nu)
            Svv = (Evv + nu*Euu) / (1-nu**2) + nu*s / (1-nu)
            Suv = Euv / (1 + nu)
            Err = (1+nu)*(1-2*nu)*s / (1-nu) - nu*(Euu + Evv) / (1-nu)
            U += pw * w * 0.5 * (Suu*Euu + Svv*Evv + 2*Suv*Euv + s*Err)
    return U


def _d(fn, e, chi, h=1e-3):
    f = [fn(e * np.exp(k * h), chi) for k in (-2, -1, 1, 2)]
    return (f[0] - 8*f[1] + 8*f[2] - f[3]) / (12 * h * e)


def eps_min(fn, chi):
    """Stationary point in eps.  Solving in eps rather than rho is essential:
    eps = sqrt(rho^2+chi^2) - 1 is a difference of nearly equal quantities."""
    d = lambda e: _d(fn, e, chi)
    g = np.geomspace(1e-13, (np.sqrt(1 + chi**2) - 1) * 0.99, 500)
    v = np.array([d(e) for e in g])
    k = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return brentq(d, g[k[-1]], g[k[-1] + 1], xtol=1e-20, rtol=8.9e-16)


def F_of(fn, eps, chi, h=1e-4):
    """Envelope theorem: F = dU/dx at fixed rho, in newtons."""
    rho = rho_of(eps, chi)
    g = lambda c: fn(np.sqrt(rho**2 + c**2) - 1, c)
    f = [g(chi + k*h) for k in (-2, -1, 1, 2)]
    return m.E * m.W * m.t * (f[0] - 8*f[1] + 8*f[2] - f[3]) / (12 * h)


if __name__ == "__main__":
    ch = np.linspace(0.05, 0.995, 60)
    e0 = np.array([eps_min(U_closed, c) for c in ch])
    r0 = np.array([rho_of(e, c) for e, c in zip(e0, ch)])
    U0 = np.array([U_closed(e, c) for e, c in zip(e0, ch)])
    F0 = np.array([F_of(U_closed, e, c) for e, c in zip(e0, ch)])
    lam = 1 + e0
    Pi_acc = 2 * RJ * r0 / lam**2                 # recursion-informed closure
    Pi_hom = Pi_acc / NTURN                       # homogeneous closure

    out = {}
    for nm, kw in [("acc", dict(per_turn=False)), ("hom", dict(per_turn=True))]:
        f = lambda e, c, kw=kw: U_with_pressure(e, c, **kw)
        ee = np.array([eps_min(f, c) for c in ch])
        out[nm] = dict(
            dU=np.abs(np.array([f(e, c) for e, c in zip(e0, ch)]) - U0) / U0,
            dF=np.abs(np.array([F_of(f, e, c) for e, c in zip(ee, ch)]) / F0 - 1))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ax.semilogy(ch, Pi_acc, "k-", lw=1.7,
                label=r"$|\sigma_{rr}|/\bar\sigma_{uu}$, recursion")
    ax.semilogy(ch, Pi_hom, "k--", lw=1.4,
                label=r"$|\sigma_{rr}|/\bar\sigma_{uu}$, homogeneous")
    ax.semilogy(ch, out["acc"]["dU"], "-", color="crimson", lw=1.7,
                label=r"$|\Delta U/U|$, recursion")
    ax.semilogy(ch, out["hom"]["dU"], "--", color="crimson", lw=1.4,
                label=r"$|\Delta U/U|$, homogeneous")
    FLOOR = 5e-8                      # finite-difference floor of this sweep
    mF = out["acc"]["dF"] > FLOOR
    ax.semilogy(ch[mF], out["acc"]["dF"][mF], ":", color="royalblue", lw=2.0,
                label=r"$|\Delta F/F|$, recursion")
    ax.axhline(FLOOR, color="0.85", lw=0.8, ls="--")
    ax.text(0.99, FLOOR*1.4, "resolution floor", fontsize=6.5, color="0.5",
            ha="right")
    ax.axhline(1e-2, color="0.85", lw=0.8)
    ax.text(0.07, 1.25e-2, "1%", fontsize=7, color="0.5")
    ax.set_xlabel(r"$x/L$"); ax.set_ylabel("relative magnitude")
    ax.set_ylim(1e-12, 1e0)
    ax.legend(fontsize=6.6, loc="lower left", ncol=1, framealpha=0.95,
              handlelength=1.9)
    ax.set_title("(a) the one step that does not close, bounded", fontsize=9)

    hoop, twist, memb = np.array([U_terms(e, c) for e, c in zip(e0, ch)]).T
    bx.semilogy(ch, hoop,  "-",  color="0.25",     lw=1.8, label="bending, hoop")
    bx.semilogy(ch, twist, "--", color="darkgreen", lw=1.8, label="bending, twist")
    bx.semilogy(ch, memb,  "-.", color="crimson",   lw=1.8, label="membrane")
    bx.semilogy(ch, U0,    ":",  color="royalblue", lw=2.2, label="total")
    d = hoop - twist              # which bending mode dominates
    k = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(k):
        j = k[-1]; xc = np.interp(0.0, d[j:j+2], ch[j:j+2])
        yc = np.interp(xc, ch, hoop)
        bx.plot([xc], [yc], "k*", ms=11, zorder=5)
        bx.annotate(rf"hoop overtakes twist, $x/L={xc:.2f}$",
                    xy=(xc, yc), xytext=(0.42, 4e-6), fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
        print(f"  hoop overtakes twist at x/L = {xc:.4f}; "
              f"membrane/bending at x/L=0.995 = "
              f"{memb[-1]/(hoop[-1]+twist[-1]):.3f}")
    bx.set_xlabel(r"$x/L$")
    bx.set_ylabel(r"energy per unit volume $/E$")
    bx.legend(fontsize=7, loc="lower right")
    bx.set_title("(b) the closed-form energy, term by term", fontsize=9)

    fig.tight_layout(); fig.savefig(OUT + "energy_3d.pdf"); plt.close(fig)
    print("wrote energy_3d.pdf")

    print(f"\nn = {NTURN:.3f}, tau = {tau}, rho_jam = {RJ:.4f}, Gamma = {GAM:.4e}")
    print(f"pressure ratio:  recursion {Pi_acc.min():.2e}..{Pi_acc.max():.2e}   "
          f"homogeneous {Pi_hom.min():.2e}..{Pi_hom.max():.2e}")
    for nm in ("acc", "hom"):
        print(f"  {nm}: |dU/U| <= {out[nm]['dU'].max():.2e}   "
              f"|dF/F| <= {out[nm]['dF'].max():.2e}")
    print(f"\nfields at the energy-minimizing state "
          f"(D = {m.E*m.t**3/(12*(1-nu**2)):.4g} N cm):")
    print(f"{'x/L':>6} {'rho':>7} {'eps':>10} {'sig_uu':>9} {'rn*Dk_uu':>9} "
          f"{'rn*Dk_uv':>9} {'rn*Dk_vv':>9} {'F [N]':>9}")
    for c in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        i = np.argmin(np.abs(ch - c))
        kuu, kuv, kvv = curvatures(r0[i], lam[i], ch[i])
        print(f"{ch[i]:6.3f} {r0[i]:7.4f} {e0[i]:10.3e} {m.E*e0[i]:9.2f} "
              f"{kuu:9.4f} {kuv:9.4f} {kvv:9.4f} {F0[i]:9.3f}")

