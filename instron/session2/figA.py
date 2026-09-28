"""
WHAT IS A?     F_s(chi) = A chi + B ( e^{mu theta_L chi} - 1 )

The exponent is settled (figC.py): it is mu theta_L, with nothing fitted, so
it is written that way from here on and the only unknowns are the two
amplitudes.  This script goes after A.

THE DIAGNOSTIC, following the collapse idea of figC panel (b).  Subtract the
capstan term and divide by chi:

    Y(chi) = [ F_s(chi) - B ( e^{mu theta_L chi} - 1 ) ] / chi .

If the slow term really is A chi then Y is a horizontal line at height A, and
two questions separate cleanly:
    (i)  is Y flat?                  -> is the slow term linear in chi at all?
    (ii) what sets its height?       -> what is A?

A STRUCTURAL POINT that governs (ii).  Across these twelve pulls the only
quantities that change are L and theta_L -- and hence rbar = L/theta_L.  W, t,
E, nu, mu and r_nat are the same for every run.  So any candidate for A,
however it is built, is testable only through

    A  =  K * L^a * theta_L^b ,

with K absorbing every constant.  In particular the data CANNOT say whether
mu belongs in A: mu never varied.  The classical candidates are

    bending   A_bend = W D [ (2nu-1) dk_uu/rbar + 2(1-nu)/rbar^2 ]   -> (-2, 2)
    friction  A_fric = mu D (1-nu) W theta_L^3 / L^2                 -> (-2, 3)

and neither works: the measured exponent is far steeper than 3.

THE PACKING FRACTION.  What does work is not a power law at all.  Write the
radial pitch of the wound spiral, s0, from rbar = r0 + N s0/2 with
N = theta_L/2pi turns:

    s0 = 2 (rbar - r0) / N ,        rho = t / s0 = t theta_L / (4 pi (rbar - r0)) .

rho is the fraction of the radial pitch occupied by paper: rho -> 1 is close
packing, layers touching.  Empirically

    A  =  A0 / (1 - rho) ,          A0 ~ 0.53 N ,

collapses the nine clean runs to a factor of 1.18 -- the replicate floor is
1.14 -- with no fitted exponent, and it survives the parameter uncertainty.
A0 itself is NOT testable here (nothing that could set it varied).

Run:  python figA.py      ->  figs/A_diagnostic.png
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from processing import (parse, load, nominal, draw, theta_L, usable, fit, colors,
                        bending_D, W_EXACT, MU_STATIC, QUANT, HERE, OUT, RANGES)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

CUT = 2 * QUANT / 4        # N, the usable() threshold
NMC = 2000

# s1r5 returns A = 4.48 N where its two siblings at identical nominal geometry
# give 1.32 and 1.33.  Every conclusion about the HEIGHT of A is reported both
# with and without it.
OUTLIER = "s1r5"
# jammed from the start: the whole pull is in the exponential, A is not
# observable at all
UNUSABLE = ("s2r1", "s2r2")


def _e(z):
    return np.exp(np.clip(z, -700, 700))


def m_frozen(p, chi, C):
    """A chi + B(e^{mu theta_L chi} - 1), exponent held fixed.  Free: A, B."""
    A, B = _e(p)
    return A * chi + B * (_e(C * chi) - 1.0)


def packing(L, thetaL, t, r0):
    """Radial pitch s0 of the wound spiral and the packing fraction rho = t/s0."""
    N = thetaL / (2 * np.pi)
    s0 = 2 * (L / thetaL - r0) / N
    return s0, t / s0


# ---------------------------------------------------------------------------
PLAIN = {}


def library(L, thetaL, t, r0, E, nu, rnat):
    """Every candidate for A, as (name, value, exponent a on L, exponent b on
    theta_L).  a and b are None for the ones that are not power laws."""
    D = bending_D(E, nu, t)
    W, rbar = W_EXACT, L / thetaL
    dk = 1.0 / rbar - 1.0 / rnat
    s0, rho = packing(L, thetaL, t, r0)
    one = np.ones_like(np.asarray(L, float))
    return [
        (r"bending  $WD[(2\nu-1)\Delta\kappa/\bar r+2(1-\nu)/\bar r^{2}]$",
         W * D * ((2 * nu - 1) * dk / rbar + 2 * (1 - nu) / rbar**2), -2, 2),
        (r"twist only  $2(1-\nu)WD/\bar r^{2}$",
         2 * (1 - nu) * W * D / rbar**2, -2, 2),
        (r"friction  $\mu D(1-\nu)W\theta_L^{3}/L^{2}$",
         MU_STATIC * D * (1 - nu) * W * thetaL**3 / L**2, -2, 3),
        (r"curl cross term  $WD/(\bar r\,r_{\rm nat})$", W * D / (rbar * rnat), -1, 1),
        (r"natural curl  $WD/r_{\rm nat}^{2}$", W * D / rnat**2 * one, 0, 0),
        (r"membrane  $EtW(t/\bar r)^{2}$", E * t * W * (t / rbar)**2, -2, 2),
        (r"free power  $\propto\theta_L^{6}/L^{2}$", D * W * thetaL**6 / L**2, -2, 6),
        (r"\textbf{packing}  $1/(1-\rho)$", 1.0 / np.maximum(1 - rho, 1e-6), None, None),
        (r"packing  $1/(1-\rho^{2})=1/\chi_{\rm jam}^{2}$",
         1.0 / np.maximum(1 - rho**2, 1e-6), None, None),
        (r"packing  $A_{\rm bend}/(1-\rho)$",
         W * D * ((2 * nu - 1) * dk / rbar + 2 * (1 - nu) / rbar**2)
         / np.maximum(1 - rho, 1e-6), None, None),
        (r"packing  $A_{\rm fric}/(1-\rho)$",
         MU_STATIC * D * (1 - nu) * W * thetaL**3 / L**2 / np.maximum(1 - rho, 1e-6),
         None, None),
        (r"gap  $WD/(\bar r\,(s_0-t))$", W * D / (rbar * np.maximum(s0 - t, 1e-9)),
         None, None),
    ]


PLAIN = {
    r"bending  $WD[(2\nu-1)\Delta\kappa/\bar r+2(1-\nu)/\bar r^{2}]$":
        "bending   WD[(2nu-1)dk/rbar + 2(1-nu)/rbar^2]",
    r"twist only  $2(1-\nu)WD/\bar r^{2}$":  "twist only   2(1-nu)WD/rbar^2",
    r"friction  $\mu D(1-\nu)W\theta_L^{3}/L^{2}$":
        "friction   mu D(1-nu) W thetaL^3/L^2",
    r"curl cross term  $WD/(\bar r\,r_{\rm nat})$": "curl cross   WD/(rbar rnat)",
    r"natural curl  $WD/r_{\rm nat}^{2}$": "natural curl   WD/rnat^2",
    r"membrane  $EtW(t/\bar r)^{2}$": "membrane   E t W (t/rbar)^2",
    r"free power  $\propto\theta_L^{6}/L^{2}$": "free power   ~ thetaL^6/L^2",
    r"\textbf{packing}  $1/(1-\rho)$": "PACKING   1/(1-rho)",
    r"packing  $1/(1-\rho^{2})=1/\chi_{\rm jam}^{2}$": "packing   1/(1-rho^2) = 1/chi_jam^2",
    r"packing  $A_{\rm bend}/(1-\rho)$": "packing   A_bend/(1-rho)",
    r"packing  $A_{\rm fric}/(1-\rho)$": "packing   A_fric/(1-rho)",
    r"gap  $WD/(\bar r\,(s_0-t))$": "gap   WD/(rbar (s0-t))",
}


# ---------------------------------------------------------------------------
def main():
    metas = [parse(p) for p in sorted(glob.glob(os.path.join(HERE, "set*.csv")))]
    metas.sort(key=lambda m: (m["set"], m["run"]))
    nom = nominal()
    rng = np.random.default_rng(1)

    print("=" * 104)
    print("A, with the exponent fixed at mu theta_L.  "
          "Y(chi) = [F_s - B(e^{mu theta_L chi}-1)]/chi")
    print("=" * 104)
    print(f"{'run':<7}{'L':>6}{'thetaL':>8}{'N':>6}{'pitch mm':>10}{'rho':>7}"
          f"{'chimax':>8}{'A (N)':>9}{'90% CI':>15}{'slope':>8}")

    for m in metas:
        x, F, Fs, tare = load(m)
        m.update(chi=x / m["L"], Fs=Fs, short=m["label"].split()[0])
        m["thetaL"] = theta_L(m, m["L"], nom["t"], nom["r0"])
        m["C"] = MU_STATIC * m["thetaL"]
        m["s0"], m["rho"] = packing(m["L"], m["thetaL"], nom["t"], nom["r0"])
        k = usable(Fs)
        c, y = m["chi"][k], Fs[k]
        p, r = fit(m_frozen, [np.log(1.0), np.log(1e-3)], c, y, args=(m["C"],))
        A, B = np.exp(p)
        ex = B * (_e(m["C"] * c) - 1.0)
        Y = (y - ex) / c
        lin = (Y > 0) & (A * c > ex)
        # the window where A is observable: off the quantization floor, and
        # before the capstan term has grown enough to matter
        cln = (Y > 0) & (y > 2 * CUT) & (ex < 0.25 * y) & (c > 0.02)
        res = np.log(y) - np.log(np.maximum(m_frozen(p, c, m["C"]), 1e-300))
        bs = [np.exp(fit(m_frozen, p, c, np.exp(np.log(y) + rng.choice(res, len(res))),
                         args=(m["C"],))[0][0]) for _ in range(150)]
        lo, hi = np.percentile(bs, [5, 95])
        sl = (np.polyfit(np.log(c[cln]), np.log(Y[cln]), 1)[0] if cln.sum() > 30 else np.nan)
        m.update(A=A, B=B, Alo=lo, Ahi=hi, c=c, y=y, Y=Y, lin=lin, cln=cln,
                 slope=sl, rms=r)
        print(f"{m['short']:<7}{m['L']:6.2f}{m['thetaL']:8.1f}{m['thetaL']/2/np.pi:6.1f}"
              f"{1e3*m['s0']:10.3f}{m['rho']:7.3f}{c.max():8.3f}{A:9.3f}"
              f"  [{lo:5.2f},{hi:5.2f}]"
              + (f"{sl:8.2f}" if np.isfinite(sl) else f"{'--':>8}"))

    good = [m for m in metas if m["short"] not in UNUSABLE and m["A"] > 1e-3]
    print(f"\n  dropped {list(UNUSABLE)}: jammed from the start, the whole pull is "
          "already in the\n  exponential, so A is not observable (the fit returns "
          "A = 0 or an unconstrained value).")

    # ---------------- (i) is Y flat? ------------------------------------
    sl = np.array([m["slope"] for m in good if np.isfinite(m["slope"])])
    print(f"\n  (i) IS Y FLAT?   slope of log Y vs log chi over the clean window "
          f"(F_s > {2*CUT:.2f} N,\n      capstan term < 25% of F_s, chi > 0.02):")
    print("      " + "  ".join(f"{m['short']}:{m['slope']:+.2f}"
                               for m in good if np.isfinite(m["slope"])))
    print(f"      median {np.median(sl):+.2f}; seven of ten within 0.21 of zero.")
    print("      -> YES.  Where A can be seen at all, the slow term is linear in chi\n"
          "         to about +/-20%.  Note how narrow that window is: the force is\n"
          "         either sitting on the 0.05 N floor or already capstan-amplified.\n"
          "         The two exceptions (s1r6, s1r7) have the narrowest windows.")

    scaling(good, nom)
    figure(metas, good, nom)


# ---------------------------------------------------------------------------
def fit_ab(runs, fix_a=None):
    y = np.log([m["A"] for m in runs])
    lL = np.log([m["L"] for m in runs]); lT = np.log([m["thetaL"] for m in runs])
    if fix_a is None:
        X = np.column_stack([np.ones(len(y)), lL, lT])
    else:
        y = y - fix_a * lL
        X = np.column_stack([np.ones(len(y)), lT])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    se = np.sqrt((res @ res / max(len(y) - X.shape[1], 1))
                 * np.diag(np.linalg.pinv(X.T @ X)))
    return beta, se, np.sqrt(res @ res / len(y))


def replicate_floor(runs):
    out, ss, dof = [], 0.0, 0
    for g in [("s1r2", "s1r3", "s1r4"), ("s1r5", "s1r6", "s1r7")]:
        v = np.log([m["A"] for m in runs if m["short"] in g])
        if len(v) < 2:
            continue
        ss += np.sum((v - v.mean())**2); dof += len(v) - 1
        out.append((g, np.exp(v), np.exp(v).max() / np.exp(v).min()))
    return np.sqrt(ss / dof), out


def rank(runs, nom, tag):
    A = np.array([m["A"] for m in runs])
    L = np.array([m["L"] for m in runs]); th = np.array([m["thetaL"] for m in runs])
    print(f"\n  candidates, each rescaled by its single best prefactor k  [{tag}]")
    print(f"    {'expression':<52}{'(a,b)':>9}{'k':>12}{'scatter':>10}")
    out = {}
    for nm, P, a, b in library(L, th, nom["t"], nom["r0"], nom["E"], nom["nu"],
                               nom["rnat"]):
        P = np.asarray(P, float) * np.ones_like(L)
        clean = PLAIN.get(nm, nm)
        if np.any(P <= 0):
            print(f"    {clean[:50]:<52}{'':>9}{'  (sign)':>12}")
            continue
        r = A / P
        k = np.exp(np.mean(np.log(r))); sc = np.exp(np.std(np.log(r)))
        out[nm] = (k, sc)
        ab = f"({a},{b})" if a is not None else "--"
        print(f"    {clean[:50]:<52}{ab:>9}{k:12.4g}{'  x%.3f' % sc:>10}")
    print(f"    {'(A with no formula at all)':<52}{'':>9}{'':>12}"
          f"{'  x%.3f' % np.exp(np.std(np.log(A))):>10}")
    return out


def scaling(good, nom):
    print("\n" + "=" * 104)
    print("(ii) WHAT SETS THE HEIGHT?   only L and theta_L vary between runs")
    print("=" * 104)
    A = np.array([m["A"] for m in good])
    print(f"  A runs {A.min():.2f} to {A.max():.2f} N, a scatter of "
          f"x{np.exp(np.std(np.log(A))):.2f}, and within each run it is pinned to "
          "1-2%\n  by the bootstrap.  So the spread is real, not fit noise.")
    floor, groups = replicate_floor(good)
    for g, v, rat in groups:
        print(f"  replicates {g}: " + " ".join(f"{q:.2f}" for q in v)
              + f"   spread x{rat:.2f}")
    print(f"  pooled replicate floor x{np.exp(floor):.2f}.  No expression in L and "
          "theta_L can beat this.")
    print(f"  {OUTLIER} is the one bad actor; everything below is reported with and "
          "without it.")

    for tag, runs in (("all 10", good),
                      (f"without {OUTLIER}", [m for m in good if m["short"] != OUTLIER])):
        b, s, r = fit_ab(runs, fix_a=-2.0)
        f2, _ = replicate_floor(runs)
        print(f"\n  power-law fit A = K L^-2 theta_L^b  [{tag}]:  b = {b[1]:+.2f} "
              f"+/- {s[-1]:.2f},  rms {r:.3f}")
        print(f"    bending needs b=2 ({abs(b[1]-2)/s[-1]:.1f} sigma away), "
              f"friction needs b=3 ({abs(b[1]-3)/s[-1]:.1f} sigma away)")
        rank(runs, nom, tag)

    g2 = [m for m in good if m["short"] != OUTLIER]
    print(f"\n  -> With {OUTLIER} in, nothing is resolved: every candidate sits near "
          "the raw scatter.\n     With it out, the packing forms come closest "
          "to the replicate floor and the\n     two classical closed forms do not.")

    # ---- does the winner survive the parameter uncertainty? -------------
    print("\n  Monte Carlo over the stated ranges for E, nu, t, r0, r_nat, L "
          f"({NMC} draws), without {OUTLIER}:")
    rng = np.random.default_rng(0)
    A = np.array([m["A"] for m in g2])
    keep = {}
    for _ in range(NMC):
        t = rng.uniform(RANGES["t"][0], RANGES["t"][2])
        r0 = rng.uniform(RANGES["r0"][0], RANGES["r0"][2])
        E = rng.uniform(RANGES["E"][0], RANGES["E"][2])
        nu = rng.uniform(RANGES["nu"][0], RANGES["nu"][2])
        rn = rng.uniform(RANGES["rnat"][0], RANGES["rnat"][2])
        L = np.array([m["L"] for m in g2]) + rng.uniform(-0.01, 0.01)
        th = np.array([theta_L(m, l, t, r0) for m, l in zip(g2, L)])
        for nm, P, a, b in library(L, th, t, r0, E, nu, rn):
            P = np.asarray(P, float) * np.ones_like(L)
            if np.any(P <= 0):
                continue
            keep.setdefault(nm, []).append(np.exp(np.std(np.log(A / P))))
    print(f"    {'expression':<52}{'scatter median':>16}{'90% interval':>22}")
    for nm, v in sorted(keep.items(), key=lambda kv: np.median(kv[1])):
        v = np.array(v)
        clean = PLAIN.get(nm, nm)
        print(f"    {clean[:50]:<52}{'x%.3f' % np.median(v):>16}"
              f"{'[x%.3f, x%.3f]' % tuple(np.percentile(v, [5, 95])):>22}")
    floor2, _ = replicate_floor(g2)
    print(f"    replicate floor for these runs: x{np.exp(floor2):.3f}")
    print("\n    Reading.  1/(1-rho) is the tightest and by far the most stable "
          "under the\n    parameter ranges -- its 90% interval barely moves, because "
          "rho depends on t and\n    r0 the same way in every run.  It beats both "
          "closed forms and it costs no\n    fitted exponent.  Two honest limits: "
          f"x{np.exp(floor2):.2f} is the replicate floor here and the\n    packing "
          "form sits at x1.20, so it is not a complete account; and A0 ~ 0.5 N is a\n"
          "    bare constant -- nothing that could set it (E, t, W, nu, r_nat, mu) "
          "varied between\n    runs, so the data constrains the SHAPE 1/(1-rho) and "
          "says nothing about the prefactor.")


# ---------------------------------------------------------------------------
def figure(metas, good, nom):
    col = colors(metas)
    fig = plt.figure(figsize=(13.8, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.24)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- (a) the diagnostic ---------------------------------------------
    lab = []
    for m in metas:
        if m["lin"].sum() > 20:
            ax_a.semilogy(m["c"][m["lin"]], m["Y"][m["lin"]], ".", ms=1.5,
                          alpha=0.16, color="0.55")
        if m["cln"].sum() > 30:
            ax_a.semilogy(m["c"][m["cln"]], m["Y"][m["cln"]], ".", ms=2.4,
                          alpha=0.65, color=col[m["file"]])
        if m["A"] > 1e-3 and m["short"] not in UNUSABLE:
            ax_a.axhline(m["A"], color=col[m["file"]], lw=0.8, ls="--", alpha=0.85)
            lab.append([np.log10(m["A"]), m["short"], col[m["file"]]])
    lab.sort()
    for j in range(1, len(lab)):
        lab[j][0] = max(lab[j][0], lab[j - 1][0] + 0.036)
    for yv, txt, cc in lab:
        ax_a.text(0.268, 10 ** yv, txt, color=cc, fontsize=6.5, va="center")
    ax_a.set_xlim(0, 0.295); ax_a.set_ylim(0.35, 12)
    ax_a.set_xlabel(r"$\chi = x/L$")
    ax_a.set_ylabel(r"$Y=\left[F_s-B\!\left(e^{\mu\theta_L\chi}-1\right)\right]/\chi$  (N)")
    ax_a.set_title("(a)  capstan term removed, divided by $\\chi$.  A slow term "
                   "$A\\chi$\nis a horizontal line at height $A$ (dashed).  Grey: "
                   "floor- or capstan-\ncontaminated.", fontsize=10, loc="left")

    # ---- (b) flatness ----------------------------------------------------
    for m in metas:
        k = m["cln"]
        if k.sum() < 30 or m["A"] < 1e-3:
            continue
        ax_b.plot(m["c"][k], m["Y"][k] / m["A"], ".", ms=2.0, alpha=0.5,
                  color=col[m["file"]])
        ax_b.text(m["c"][k].max() + 0.002, np.median(m["Y"][k][-40:]) / m["A"],
                  m["short"], fontsize=6, color=col[m["file"]], va="center")
    ax_b.axhline(1.0, color="k", lw=2.0, zorder=8)
    ax_b.set_ylim(0.55, 1.55); ax_b.set_xlim(0, 0.25)
    ax_b.set_xlabel(r"$\chi = x/L$"); ax_b.set_ylabel(r"$Y(\chi)\,/\,A$")
    sl = np.array([m["slope"] for m in good if np.isfinite(m["slope"])])
    ax_b.set_title("(b)  the same, normalised, over the window where $A$ is\n"
                   rf"observable. Log-log slope median ${np.median(sl):+.2f}$ "
                   "(exactly linear is $0$).\nThe sawtooth is the stick--slip.",
                   fontsize=10, loc="left")

    # ---- (c) the height against theta_L ----------------------------------
    D = bending_D(nom["E"], nom["nu"], nom["t"])
    lab = []
    for m in good:
        Ah = m["A"] * m["L"]**2 / (D * W_EXACT)
        ax_c.plot([m["thetaL"]] * 2,
                  [m["Alo"] * m["L"]**2 / (D * W_EXACT),
                   m["Ahi"] * m["L"]**2 / (D * W_EXACT)], color=col[m["file"]], lw=2)
        ax_c.plot(m["thetaL"], Ah, "o", color=col[m["file"]], ms=6, mec="k", mew=0.5)
        lab.append([np.log10(Ah), m["thetaL"], m["short"], col[m["file"]]])
    lab.sort()
    for j in range(1, len(lab)):
        lab[j][0] = max(lab[j][0], lab[j - 1][0] + 0.052)
    for yv, xv, txt, cc in lab:
        ax_c.text(xv * 1.005, 10 ** yv, txt, fontsize=6, color=cc, va="center")
    g = np.linspace(245, 320, 50)
    nu = nom["nu"]
    ax_c.plot(g, 2 * (1 - nu) * g**2, "k--", lw=1.6, label=r"bending, $\theta_L^{2}$")
    ax_c.plot(g, MU_STATIC * (1 - nu) * g**3, color="crimson", ls="-.", lw=1.6,
              label=r"friction, $\mu\theta_L^{3}$")
    b2, s2, _ = fit_ab([m for m in good if m["short"] != OUTLIER], fix_a=-2.0)
    ax_c.plot(g, np.exp(b2[0]) * g**b2[1] / (D * W_EXACT), color="tab:green", lw=1.6,
              label=rf"best power, $\theta_L^{{{b2[1]:.1f}\pm{s2[-1]:.1f}}}$")
    ax_c.set_xscale("log"); ax_c.set_yscale("log"); ax_c.set_xlim(246, 322)
    ax_c.set_xticks([250, 260, 270, 280, 290, 300, 310, 320])
    ax_c.set_xticklabels([str(v) for v in [250, 260, 270, 280, 290, 300, 310, 320]])
    ax_c.minorticks_off()
    ax_c.set_xlabel(r"$\theta_L$  (rad)")
    ax_c.set_ylabel(r"$\hat A = A L^{2}/(DW)$")
    ax_c.set_title("(c)  the height, nondimensionalised.  The measured rise with\n"
                   r"$\theta_L$ is far steeper than either closed form allows.",
                   fontsize=10, loc="left")
    ax_c.legend(fontsize=8, loc="upper left")

    # ---- (d) the packing collapse ----------------------------------------
    A = np.array([m["A"] for m in good if m["short"] != OUTLIER])
    X = np.array([1 / (1 - m["rho"]) for m in good if m["short"] != OUTLIER])
    A0 = np.exp(np.mean(np.log(A / X)))
    sc = np.exp(np.std(np.log(A / X)))
    for m in good:
        xx = 1 / (1 - m["rho"])
        mk = "s" if m["short"] == OUTLIER else "o"
        ax_d.plot([xx] * 2, [m["Alo"], m["Ahi"]], color=col[m["file"]], lw=2)
        ax_d.plot(xx, m["A"], mk, color=col[m["file"]], ms=7, mec="k", mew=0.6,
                  zorder=5)
        ax_d.text(xx * 1.02, m["A"], m["short"], fontsize=6, color=col[m["file"]],
                  va="center")
    gg = np.linspace(1.4, 5.0, 50)
    ax_d.plot(gg, A0 * gg, "k-", lw=2,
              label=rf"$A = A_0/(1-\rho)$,  $A_0={A0:.2f}$ N")
    ax_d.fill_between(gg, A0 * gg / sc, A0 * gg * sc, color="k", alpha=0.10, lw=0)
    floor2, _ = replicate_floor([m for m in good if m["short"] != OUTLIER])
    ax_d.plot([], [], " ", label=(rf"scatter $\times{sc:.2f}$ (floor "
                                  rf"$\times{np.exp(floor2):.2f}$), $n=9$"))
    ax_d.plot([], [], "s", color="0.5", mec="k", label=f"{OUTLIER}, excluded")
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.set_ylim(0.6, 6.5)
    ax_d.set_xticks([1.5, 2, 2.5, 3, 4, 5]); ax_d.minorticks_off()
    ax_d.set_xticklabels(["1.5", "2", "2.5", "3", "4", "5"])
    ax_d.set_xlabel(r"$1/(1-\rho)$,   $\rho = t\,\theta_L/[4\pi(\bar r - r_0)]$"
                    "\n(packing fraction of the wound stack)")
    ax_d.set_ylabel(r"$A$  (N)")
    ax_d.set_title("(d)  what does work: not a power of $\\theta_L$ but the\n"
                   "approach to close packing. One constant, no fitted exponent.",
                   fontsize=10, loc="left")
    ax_d.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(r"The slow term is $A\chi$.  Its height is not bending and not "
                 r"sliding friction — it tracks how tightly the coil is packed.",
                 fontsize=13, y=0.975)
    p = os.path.join(OUT, "A_diagnostic.png")
    fig.savefig(p); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
