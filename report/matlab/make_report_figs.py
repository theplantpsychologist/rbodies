"""
make_report_figs.py -- generate the figures used by report3d_claude.tex.

Writes into ../figs/:
    force_decomposition.pdf   F(x): analytic bending, numeric bending, friction route
    radius_energy.pdf         energy-minimizing rho(x), energy landscape, geometry map
    stress_fields.pdf         sigma_rr^-, S_rr, sigma_uu heatmaps + contact gate
    strain_closure.pdf        eps(x) exact vs closed form, and its error
    capstan_gain.pdf          capstan gain A(x) against the exponent lambda_+ L
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import energy_coupled_solver as m

OUT = "../figs/"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def F_bend_analytic(chi):
    """Inextensible bending force, closed form (report Sec. 'coil radius')."""
    phi = np.arcsin(np.clip(chi, 0, 1))
    c, s = np.cos(phi), np.sin(phi)
    return (m.W * m.D_plate / m.r_nat**2) * (
        s * (1 - c) / c**4 + (1 - m.nu) * (2 * s * c**2 + s**3) / c**3)


from scipy.optimize import brentq


def stat_resid(rho, chi):
    """Left minus right of the exact stationarity condition."""
    lam = np.sqrt(rho**2 + chi**2)
    return ((1 - rho) / rho**3
            + (1 - m.nu) * chi**2 * (3 * rho**2 + chi**2) / (rho**2 * lam**4)
            - m.GAMMA * rho * (lam - 1) / lam)


def H(r0):
    return (1 - r0) + (1 - m.nu) * r0 * (1 - r0**2) * (1 + 2 * r0**2)


def rho_exact(chi):
    return brentq(stat_resid, 1e-6, 1.5, args=(chi,), xtol=1e-14, rtol=1e-15)


def U_unified(rho, chi, G, rj):
    """Nondimensional energy with the bending density integrated over the true
    radius distribution rho_loc = rho + 2*rj*sigma, sigma in [-1/2,1/2].
    Reduces to U_total_nondim as rj -> 0; diverges as rho -> rj."""
    rho = np.asarray(rho, dtype=float)
    out = np.full(rho.shape, np.inf)
    ok = rho > rj * (1 + 1e-12)
    r, lam = rho[ok], np.sqrt(rho[ok]**2 + chi**2)
    Lc = (np.log((r + rj) / (r - rj)) / (2 * rj)) if rj > 1e-12 else 1.0 / r
    out[ok] = (1.0 / (r**2 - rj**2) - 2 * Lc + 1.0
               + 2 * (1 - m.nu) * chi**2 / lam**2 * Lc + G * (lam - 1)**2)
    return out


def rho_star_of(G):
    """Crossover radius: the 50/50 split of the pull increment."""
    return (4.0 / G) ** (1 / 6)


def rho_star_of_tau(tau):
    return (tau**2 / (3.0 * (1 - m.nu**2))) ** (1 / 6)


# ---------------------------------------------------------------- sweep
chis = np.linspace(0.10, 0.985, 28)
res = m.run_sweep(chis)
xs = np.array([s["x"] for s in res])
rho = np.array([s["rho"] for s in res])
F_fr = np.array([s["F_end"] for s in res])
Ub = np.array([s["U_bend"] for s in res])
Fb_num = np.gradient(Ub, xs)
Fb_an = F_bend_analytic(chis)

# ---------------------------------------------------- fig 1: force
twist = np.array([2 * (1 - m.nu) * m.W * m.D_plate / m.r_nat**2
                  * s_["rho"] * s_["chi"] / s_["lam"]**4 for s_ in res])
memb = np.array([s_["F_end"] for s_ in res])          # membrane end reaction
F_tot = twist + memb

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax[0].plot(xs, F_tot, "k-", lw=1.8, label=r"$F$ total")
ax[0].plot(xs, twist, "--", color="C0", label=r"twist term")
ax[0].plot(xs, memb, "-.", color="C3", label=r"membrane term")
ax[0].set_xlabel(r"$x$ [cm]"); ax[0].set_ylabel(r"$F$ [N]")
ax[0].set_ylim(0, 90); ax[0].legend(fontsize=7); ax[0].set_title("(a) linear")

ax[1].semilogy(xs, F_tot, "k-", lw=1.8, label=r"$F$ total")
ax[1].semilogy(xs, twist, "--", color="C0", label=r"twist term")
ax[1].semilogy(xs, memb, "-.", color="C3", label=r"membrane term")
ax[1].semilogy(xs, Fb_an, ":", color="C2",
               label=r"$F$ friction-free, inextensible")
ax[1].set_xlabel(r"$x$ [cm]"); ax[1].set_ylabel(r"$F$ [N]")
ax[1].legend(fontsize=6.5); ax[1].set_title("(b) semilog")
fig.tight_layout(); fig.savefig(OUT + "force_decomposition.pdf"); plt.close(fig)
print("wrote force_decomposition.pdf")

# ------------------------------------------- fig 2: radius / energy
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
ax[0].plot(chis, rho, "o-", ms=3, label=r"$\rho$ minimizing $U$")
ax[0].plot(chis, np.sqrt(np.maximum(1 - chis**2, 0)), "k:",
           label=r"inextensible $\cos\phi$")
ax[0].axhline(m.RHO_JAM, color="C3", ls="--", label=r"$\rho_{\rm jam}$")
ax[0].set_xlabel(r"$x/L$"); ax[0].set_ylabel(r"$\rho=r/r_{\rm nat}$")
ax[0].legend(fontsize=6.5); ax[0].set_title("(a) coil radius")

rr = np.linspace(0.02, 1.15, 1500)
for chi, cc in zip([0.7, 0.95, 1.0], ["C0", "C1", "C2"]):
    U = U_unified(rr, chi, m.GAMMA, 0.0)
    ax[1].semilogy(rr, U - np.nanmin(U) + 1e-6, color=cc, label=rf"$x/L={chi}$")
U = U_unified(rr, 1.0, m.GAMMA, 0.25)
ax[1].semilogy(rr, U - np.nanmin(U) + 1e-6, "k--", lw=1.2,
               label=r"$x/L=1$, $\rho_{\rm jam}=0.25$")
ax[1].axvline(0.25, color="k", ls=":", lw=0.8)
ax[1].set_xlim(0, 1.15); ax[1].set_ylim(1e-6, 1e4)
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$\tilde U-\tilde U_{\min}$")
ax[1].legend(fontsize=6); ax[1].set_title("(b) energy landscape")

# grid computed wider than the plotted window so the contour does not
# terminate on the domain edge and draw a spurious boundary segment
tau = np.logspace(-3.6, -0.6, 400)          # t / r_nat
nn = np.logspace(-0.4, 2.7, 400)            # number of turns
TAU, NN = np.meshgrid(tau, nn, indexing="ij")
ratio = (0.5 * NN * TAU) / rho_star_of_tau(TAU)
pc = ax[2].pcolormesh(TAU, NN, np.log10(ratio), cmap="coolwarm",
                      vmin=-1.5, vmax=1.5, shading="auto")
ax[2].contour(TAU, NN, ratio, levels=[1.0], colors="k", linewidths=1.4)
ax[2].set_xlim(10**-3.2, 10**-1.0)
ax[2].set_ylim(1.0, 10**2.3)
ax[2].plot(m.t / m.r_nat, m.theta_L / (2 * np.pi), "k*", ms=9,
           label="this coil")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel(r"$t/r_{\rm nat}$"); ax[2].set_ylabel(r"turns $n$")
ax[2].legend(fontsize=6.5, loc="lower left")
cb = plt.colorbar(pc, ax=ax[2], pad=0.02)
cb.set_label(r"$\log_{10}(\rho_{\rm jam}/\rho_*)$", fontsize=7)
cb.ax.tick_params(labelsize=6)
ax[2].set_title("(c) which mechanism sets the scale")
fig.tight_layout(); fig.savefig(OUT + "radius_energy.pdf"); plt.close(fig)
print("wrote radius_energy.pdf")

# ------------------------------------------- fig 3: stress fields
pick = [8, 18, 25]
fig, axs = plt.subplots(3, len(pick), figsize=(7.4, 5.0))
for col, k in enumerate(pick):
    s = res[k]
    ext = [0, m.L, -m.W / 2, m.W / 2]
    for row, (key, lab) in enumerate([("sm", r"$\sigma_{rr}^-$"),
                                      ("sig_uu", r"$\bar\sigma_{uu}$")]):
        f = s[key]
        vmax = np.percentile(np.abs(f), 99.0) or 1.0
        im = axs[row, col].imshow(
            f.T, origin="lower", aspect="auto", cmap="RdBu_r", extent=ext,
            norm=SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax))
        plt.colorbar(im, ax=axs[row, col], pad=0.02)
        axs[row, col].set_title(rf"{lab},  $x/L={s['chi']:.2f}$", fontsize=7.5)
    g = np.where(s["has_inner"] & s["has_outer"], 0,
                 np.where(s["has_outer"], 1, np.where(s["has_inner"], 2, 3)))
    im = axs[2, col].imshow(g.T, origin="lower", aspect="auto", cmap="viridis",
                            vmin=0, vmax=3, extent=ext)
    plt.colorbar(im, ax=axs[2, col], ticks=[0, 1, 2, 3], pad=0.02)
    axs[2, col].set_title(rf"contact gate,  $x/L={s['chi']:.2f}$", fontsize=7.5)
for a in axs.ravel():
    a.set_xlabel(r"$u$ [cm]", fontsize=7); a.set_ylabel(r"$v$ [cm]", fontsize=7)
    a.tick_params(labelsize=6); a.grid(False)
fig.tight_layout(); fig.savefig(OUT + "stress_fields.pdf"); plt.close(fig)
print("wrote stress_fields.pdf")

# ------------------------------------------- fig 4: strain / closure
ch = np.linspace(0.05, 0.995, 400)
r0 = np.sqrt(1 - ch**2)
eps_as = H(r0) / (m.GAMMA * r0**4)
rex = np.array([rho_exact(c) for c in ch])
eps_ex = np.sqrt(rex**2 + ch**2) - 1

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax[0].semilogy(ch, eps_ex, "k-", lw=1.8, label=r"exact root")
ax[0].semilogy(ch, eps_as, "--", color="C3", lw=1.5, label=r"closed form")
ax[0].set_xlabel(r"$x/L$"); ax[0].set_ylabel(r"$\epsilon$")
ax[0].legend(fontsize=7); ax[0].set_title("(a) mean axial strain")

ax[1].semilogy(ch, np.abs(eps_as - eps_ex) / eps_ex, "--", color="C3", lw=1.5)
ax[1].set_xlabel(r"$x/L$")
ax[1].set_ylabel(r"rel. error in $\epsilon$")
ax[1].set_title("(b) closed-form accuracy")
fig.tight_layout(); fig.savefig(OUT + "strain_closure.pdf"); plt.close(fig)
print("wrote strain_closure.pdf")

# ------------------------------------------- fig 5: capstan gain
A_gain = np.array([s_["sig_uu"][-1].mean() / (m.E * (s_["lam"] - 1))
                   for s_ in res])
lpL = np.array([s_["lam_plus_L"] for s_ in res])

fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(chis, A_gain, "o-", ms=3.5, color="C3", label=r"$\mathcal{A}$")
ax.set_xlabel(r"$x/L$"); ax.set_ylabel(r"capstan gain $\mathcal{A}$")
a2 = ax.twinx(); a2.plot(chis, lpL, "s--", ms=3.5, color="C0",
                         label=r"$\lambda_+ L$")
a2.set_ylabel(r"$\lambda_+ L$"); a2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "capstan_gain.pdf"); plt.close(fig)
print("wrote capstan_gain.pdf")

print("\nchecks:")
ok = ~np.isclose(Fb_an, 0)
print("  Fb_num/Fb_an over x/L in [0.2,0.8]:",
      np.array2string((Fb_num / Fb_an)[(chis > 0.2) & (chis < 0.8)], precision=4))
print(f"  F_bend range {Fb_an[0]:.4g} -> {Fb_an[-1]:.4g} N, monotonic "
      f"{bool(np.all(np.diff(Fb_an) > 0))}")
print(f"  F_total {F_tot[0]:.4g} -> {F_tot[-1]:.4g} N; twist=membrane near "
      f"x/L={chis[np.argmin(np.abs(twist-memb))]:.3f}")
print(f"  capstan gain A: {A_gain.min():.2f} .. {A_gain.max():.2f}")
print(f"  rho_* = {rho_star_of(m.GAMMA):.4f}, rho_jam = {m.RHO_JAM:.4f}, "
      f"ratio = {m.RHO_JAM/rho_star_of(m.GAMMA):.3f}")
print(f"  closed-form eps rel err at x/L=0.9: "
      f"{np.interp(0.9, ch, np.abs(eps_as-eps_ex)/eps_ex):.2e}")
