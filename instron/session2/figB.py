"""
WHAT IS B?     F_s(chi) = A chi + B ( e^{mu theta_L chi} - 1 )

Exponent frozen at mu theta_L throughout, as in figC.py and figA.py.  B is a
force, like A; the exponent's argument is what is dimensionless.

The premise worth testing first: because the exponential dominates the linear
term over most of the interesting range, on a LINEAR scale one could drop
A chi entirely and keep only B(e^{mu theta_L chi}-1).  That is checked in
panel (b) and it holds -- but B has to be refitted, and comes out about a
third larger, because it must absorb the linear term it replaced.

The answer to "what is B", though, is that B is the wrong thing to ask about.

  1. B is NOT a stable property of the curve.  Truncating the fit window moves
     the fitted A by x1.03 and the fitted B by up to x9.4 (median x1.4).

  2. The reason is an identity.  Let chi* be the crossover, where the two terms
     are equal, and xi* = mu theta_L chi*.  Then

         ln B  =  ln(A chi*)  -  ln(e^{xi*} - 1)  ~  ln(A chi*) - xi* .

     xi* measures 6.7 with a scatter of x1.34, i.e. +/- 2.0 absolute.  Exponentiate
     that and you get exactly the observed spread of B: std(ln B) = 1.97 against
     std(xi*) = 1.96.  B's three decades are a modest uncertainty in the
     crossover, amplified by the exponential.

  3. So the predictable object is xi*, not B.  xi* drifts only x1.09 under
     truncation.  Reparametrized,

         F_s(chi) = A [ chi + chi* (e^{mu theta_L chi} - 1)/(e^{xi*} - 1) ],

     with xi* ~ 6.7: the friction term overtakes the bending term once the
     capstan gain reaches e^{6.7} ~ 800.  No dependence of xi* on L, theta_L or
     the packing fraction is resolvable here.

  4. A side benefit: the algebraic prefactor the theory actually predicts,
     A chi [1 + k(e^{mu theta_L chi} - 1)], fits identically well (pooled rms
     0.2014 against 0.2018).  The mismatch flagged in the report between the
     theory's chi(e^{C chi}-1) and the fit's constant B is therefore not a
     contradiction: the data cannot tell the two apart.

Run:  python figB.py      ->  figs/B_diagnostic.png
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq

from processing import (parse, load, nominal, theta_L, usable, fit, colors,
                        MU_STATIC, QUANT, HERE, OUT)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

CUT = 2 * QUANT / 4
UNUSABLE = ("s2r1", "s2r2")      # jammed from the start, no usable branch
CUTS = [0.15, 0.20, 0.27, 0.33]  # truncation windows for the stability test


def _e(z):
    return np.exp(np.clip(z, -700, 700))


def m_frozen(p, chi, C):
    """A chi + B(e^{mu theta_L chi} - 1).  Free: A, B."""
    A, B = _e(p)
    return A * chi + B * (_e(C * chi) - 1.0)


def m_Bonly(p, chi, C):
    """B(e^{mu theta_L chi} - 1) alone -- the linear term chopped.  Free: B."""
    B, = _e(p)
    return B * (_e(C * chi) - 1.0)


def m_kchi(p, chi, C):
    """A chi [1 + k(e^{mu theta_L chi} - 1)] -- the prefactor the theory gives."""
    A, k = _e(p)
    return A * chi * (1.0 + k * (_e(C * chi) - 1.0))


def crossover(A, B, C):
    """chi* where A chi = B(e^{C chi} - 1)."""
    try:
        return brentq(lambda z: A * z - B * (_e(C * z) - 1.0), 1e-8, 2.0)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
def main():
    metas = [parse(p) for p in sorted(glob.glob(os.path.join(HERE, "set*.csv")))]
    metas.sort(key=lambda m: (m["set"], m["run"]))
    nom = nominal()

    print("=" * 100)
    print("B, with the exponent frozen at mu theta_L.  B is a force, like A.")
    print("=" * 100)
    print(f"{'run':<7}{'thetaL':>8}{'chimax':>8}{'A (N)':>8}{'B (N)':>10}"
          f"{'B 1-term':>10}{'infl':>6}{'chi*':>7}{'xi*':>7}"
          f"{'relerr>1N':>11}{'relerr all':>11}")

    good = []
    for m in metas:
        x, F, Fs, tare = load(m)
        m.update(chi=x / m["L"], Fs=Fs, short=m["label"].split()[0])
        m["thetaL"] = theta_L(m, m["L"], nom["t"], nom["r0"])
        m["C"] = MU_STATIC * m["thetaL"]
        k = usable(Fs)
        m["c"], m["y"] = m["chi"][k], Fs[k]
        if m["short"] in UNUSABLE:
            continue
        c, y, C = m["c"], m["y"], m["C"]
        p, r = fit(m_frozen, [np.log(1.0), np.log(1e-3)], c, y, args=(C,))
        A, B = np.exp(p)
        dom = B * (_e(C * c) - 1.0) > A * c
        p1, _ = fit(m_Bonly, [np.log(B)], c[dom], y[dom], args=(C,))
        B1 = np.exp(p1)[0]
        pred = B1 * (_e(C * c) - 1.0)
        rel = lambda kk: np.sqrt(np.mean(((pred[kk] - y[kk]) / y[kk])**2))
        cs = crossover(A, B, C)
        m.update(A=A, B=B, B1=B1, cstar=cs, xi=C * cs, rms=r, dom=dom,
                 rel_hi=rel(y > 1.0), rel_all=rel(np.ones_like(y, bool)))
        good.append(m)
        print(f"{m['short']:<7}{m['thetaL']:8.1f}{c.max():8.3f}{A:8.2f}{B:10.2e}"
              f"{B1:10.2e}{B1/B:6.2f}{cs:7.3f}{C*cs:7.2f}"
              f"{m['rel_hi']:11.1%}{m['rel_all']:11.1%}")

    gm = lambda k: np.exp(np.mean(np.log([m[k] for m in good])))
    gs = lambda k: np.exp(np.std(np.log([m[k] for m in good])))

    print(f"\n  (1) CHOPPING THE LINEAR TERM.  Fitting B alone, on a linear scale the"
          f"\n      one-term form is good to {np.median([m['rel_hi'] for m in good]):.0%} "
          f"(median) once F_s > 1 N -- range "
          f"{min(m['rel_hi'] for m in good):.0%}-{max(m['rel_hi'] for m in good):.0%} --"
          f"\n      but {np.median([m['rel_all'] for m in good]):.0%} over the whole "
          "record, where the linear term is everything.\n      B must be inflated by "
          f"x{np.median([m['B1']/m['B'] for m in good]):.2f} (median, range "
          f"x{min(m['B1']/m['B'] for m in good):.2f}-x{max(m['B1']/m['B'] for m in good):.2f})"
          " to absorb what it replaced.")

    stability(good)
    identity(good, gm, gs)
    prefactor(good)
    figure(metas, good, gm, gs)


# ---------------------------------------------------------------------------
def stability(good):
    print("\n  (2) IS B A PROPERTY OF THE CURVE?  refit on truncated windows chi <= cut")
    print(f"      {'run':<7}" + "".join(f"{'B@%.2f' % c:>10}" for c in CUTS)
          + "   " + "".join(f"{'xi*@%.2f' % c:>9}" for c in CUTS)
          + f"{'B drift':>10}{'A drift':>9}{'xi* drift':>11}")
    for m in good:
        bb, xx, aa = [], [], []
        for cut in CUTS:
            k = m["c"] <= cut
            if k.sum() < 40 or (m["y"][k] > 2 * CUT).sum() < 15:
                bb.append(np.nan); xx.append(np.nan); aa.append(np.nan); continue
            p, _ = fit(m_frozen, [np.log(1.0), np.log(1e-3)], m["c"][k], m["y"][k],
                       args=(m["C"],))
            A, B = np.exp(p)
            bb.append(B); aa.append(A); xx.append(m["C"] * crossover(A, B, m["C"]))
        d = lambda v: (np.nanmax(v) / np.nanmin(v)
                       if np.isfinite(v).sum() > 1 else np.nan)
        m["Btrace"], m["xitrace"], m["Atrace"] = (np.array(bb), np.array(xx),
                                                  np.array(aa))
        m["Bdrift"], m["Adrift"], m["xidrift"] = d(bb), d(aa), d(xx)
        print(f"      {m['short']:<7}" + "".join(f"{v:10.1e}" for v in bb)
              + "   " + "".join(f"{v:9.2f}" for v in xx)
              + f"{'x%.1f' % m['Bdrift']:>10}{'x%.2f' % m['Adrift']:>9}"
              + f"{'x%.2f' % m['xidrift']:>11}")
    print(f"      medians:  B x{np.nanmedian([m['Bdrift'] for m in good]):.1f},"
          f"  A x{np.nanmedian([m['Adrift'] for m in good]):.2f},"
          f"  xi* x{np.nanmedian([m['xidrift'] for m in good]):.2f}")
    print("      -> Within one run Z(chi) does plateau (panel a), so the SHAPE is"
          " right and B is\n         well defined given the whole record.  But it is"
          " badly conditioned: change\n         the window and it slides by up to a"
          f" factor of {max(m['Bdrift'] for m in good):.0f}, because it is an"
          " extrapolation\n         back through a factor e^{xi*} ~ 800.  A and xi*"
          " do not move.")


def identity(good, gm, gs):
    print("\n  (3) WHY.  chi* is the crossover, xi* = mu theta_L chi*, and")
    print("        ln B = ln(A chi*) - ln(e^{xi*} - 1)   [exact, by definition of chi*]")
    lnB = np.log([m["B"] for m in good]); xi = np.array([m["xi"] for m in good])
    print(f"      xi*  = {gm('xi'):.2f}, scatter x{gs('xi'):.2f}, "
          f"i.e. +/- {np.std(xi):.2f} absolute")
    print(f"      std(xi*)   = {np.std(xi):.2f}")
    print(f"      std(ln B)  = {np.std(lnB):.2f}   <- the same number")
    print(f"      B spans {np.exp(lnB).min():.1e} to {np.exp(lnB).max():.1e} N "
          f"({np.log10(np.exp(lnB).max()/np.exp(lnB).min()):.1f} decades), "
          f"scatter x{gs('B'):.1f}")
    print("      Every bit of B's spread is a +/-30% uncertainty in the crossover,"
          "\n      exponentiated.  B also inherits the full sensitivity of the "
          "exponent:\n      one turn of error in theta_L (dC ~ 1) moves ln B by "
          "-chi* ~ -0.15, and set 2's\n      +/-20% uncertainty on theta_L moves B "
          "by about a factor of 10.")
    print(f"\n      chi*   = {gm('cstar'):.3f}, scatter x{gs('cstar'):.2f}")
    cm = np.array([m["c"].max() for m in good])
    cs = np.array([m["cstar"] for m in good])
    print(f"      chi*/chi_max = {np.exp(np.mean(np.log(cs/cm))):.3f}, "
          f"scatter x{np.exp(np.std(np.log(cs/cm))):.2f}, "
          f"corr(log xi*, log chi_max) = {np.corrcoef(np.log(xi), np.log(cm))[0,1]:+.3f}")
    print("      That correlation is the stopping rule, not a confound: xi* barely"
          "\n      moves under truncation, so it belongs to the curve, and the "
          "operator\n      stopped each pull at a roughly fixed multiple of its own "
          "crossover.")
    print("\n      correlations of log xi* with the things that varied:")
    for nm, key in (("thetaL", "thetaL"), ("L", "L"), ("A", "A")):
        v = np.log([m[key] for m in good])
        print(f"        vs log {nm:<8} {np.corrcoef(np.log(xi), v)[0,1]:+.3f}")
    print("      -> nothing resolvable.  The simplest hypothesis consistent with"
          f" these data\n         is that xi* is a constant near {gm('xi'):.1f}, hence"
          "\n           B = A xi* / [ mu theta_L (e^{xi*} - 1) ]"
          f"  ~  {gm('xi')/(np.exp(gm('xi'))-1):.4f} A/(mu theta_L),"
          "\n         which reproduces B only to the factor of ~7 that the"
          " exponential imposes.")


def prefactor(good):
    print("\n  (4) THE PREFACTOR THE THEORY PREDICTS.  M1: A chi + B(e^{..}-1);"
          "\n      M2: A chi[1 + k(e^{..}-1)], i.e. the amplified term carries the"
          " chi the\n      derivation gives it.")
    r1 = [], []
    a, b = [], []
    for m in good:
        p2, r2 = fit(m_kchi, [np.log(1.0), np.log(1e-3)], m["c"], m["y"], args=(m["C"],))
        a.append(m["rms"]); b.append(r2)
        m["k"] = np.exp(p2[1])
    print(f"      pooled rms   M1 {np.exp(np.mean(np.log(a))):.4f}"
          f"   M2 {np.exp(np.mean(np.log(b))):.4f}")
    kk = np.array([m["k"] for m in good])
    print(f"      k = {np.exp(np.mean(np.log(kk))):.2e}, scatter "
          f"x{np.exp(np.std(np.log(kk))):.1f} -- no better conditioned than B.")
    print("      -> the two are indistinguishable in this data. The report's flagged"
          "\n         structural mismatch between theory and fit is therefore not a"
          "\n         contradiction; it is simply not resolved.")


# ---------------------------------------------------------------------------
def figure(metas, good, gm, gs):
    col = colors(metas)
    fig = plt.figure(figsize=(13.8, 9.6))
    gsp = GridSpec(2, 2, figure=fig, hspace=0.33, wspace=0.24)
    ax_a, ax_b = fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[0, 1])
    ax_c, ax_d = fig.add_subplot(gsp[1, 0]), fig.add_subplot(gsp[1, 1])

    # ---- (a) the diagnostic ---------------------------------------------
    lab = []
    for m in good:
        c, y, C = m["c"], m["y"], m["C"]
        Z = (y - m["A"] * c) / (_e(C * c) - 1.0)
        k = (Z > 0) & (c > 0.02)
        ax_a.semilogy(c[k], Z[k], ".", ms=1.6, alpha=0.18, color="0.55")
        k2 = k & m["dom"]
        ax_a.semilogy(c[k2], Z[k2], ".", ms=2.4, alpha=0.6, color=col[m["file"]])
        ax_a.axhline(m["B"], color=col[m["file"]], lw=0.8, ls="--", alpha=0.85)
        lab.append([np.log10(m["B"]), m["short"], col[m["file"]]])
    lab.sort()
    for j in range(1, len(lab)):
        lab[j][0] = max(lab[j][0], lab[j - 1][0] + 0.30)
    for yv, txt, cc in lab:
        ax_a.text(0.305, 10 ** yv, txt, color=cc, fontsize=6.5, va="center")
    ax_a.set_xlim(0, 0.34); ax_a.set_ylim(1e-6, 1e-1)
    ax_a.set_xlabel(r"$\chi = x/L$")
    ax_a.set_ylabel(r"$Z=\left(F_s-A\chi\right)/\left(e^{\mu\theta_L\chi}-1\right)$  (N)")
    ax_a.set_title("(a)  the analogue of the $A$ diagnostic: divide by the capstan\n"
                   "factor instead. A constant $B$ is a horizontal line (dashed).\n"
                   "Coloured: where the capstan term dominates -- flat plateaus,\n"
                   "but spread over three decades.", fontsize=10, loc="left")

    # ---- (b) chopping the linear term, on a linear scale ----------------
    for m in good:
        c, y = m["c"], m["y"]
        ax_b.plot(c, y, "-", lw=1.4, alpha=0.9, color=col[m["file"]])
        g = np.linspace(0, c.max(), 300)
        ax_b.plot(g, m["B1"] * (_e(m["C"] * g) - 1.0), "k--", lw=1.0, zorder=6)
    ax_b.plot([], [], "k--", lw=1.0,
              label=r"$B\!\left(e^{\mu\theta_L\chi}-1\right)$ alone, $B$ refitted")
    ax_b.set_xlim(0, 0.34); ax_b.set_ylim(0, 8)
    ax_b.set_xlabel(r"$\chi = x/L$"); ax_b.set_ylabel(r"$F_s$  (N)")
    med = np.median([m["rel_hi"] for m in good])
    infl = np.median([m["B1"] / m["B"] for m in good])
    ax_b.set_title("(b)  the premise, checked on a linear scale: drop $A\\chi$ and\n"
                   rf"refit. Good to {med:.0%} (median) once $F_s>1$ N, with $B$"
                   "\n" rf"inflated ${{\times}}{infl:.2f}$ to absorb the missing term.",
                   fontsize=10, loc="left")
    ax_b.legend(fontsize=8, loc="upper left")

    # ---- (c) stability under truncation ---------------------------------
    for m in good:
        for key, mk, lw in (("Btrace", "o-", 1.6), ("Atrace", "s:", 1.0),
                            ("xitrace", "^--", 1.0)):
            v = m[key]
            if not np.isfinite(v).any():
                continue
            ref = v[np.isfinite(v)][-1]
            if key == "Btrace":
                ax_c.plot(CUTS, v / ref, mk, color=col[m["file"]], lw=lw, ms=3.5)
    for m in good:
        ax_c.plot(CUTS, m["Atrace"] / m["Atrace"][np.isfinite(m["Atrace"])][-1],
                  "-", color="0.35", lw=0.8, alpha=0.6)
        ax_c.plot(CUTS, m["xitrace"] / m["xitrace"][np.isfinite(m["xitrace"])][-1],
                  "-", color="tab:green", lw=0.8, alpha=0.6)
    ax_c.axhline(1.0, color="k", lw=1.5)
    ax_c.plot([], [], "o-", color="tab:blue", ms=4, label=r"$B$  (coloured by run)")
    ax_c.plot([], [], "-", color="tab:green", lw=1.2, label=r"$\xi^\ast$")
    ax_c.plot([], [], "-", color="0.35", lw=1.2, label=r"$A$")
    ax_c.set_yscale("log"); ax_c.set_ylim(0.08, 12)
    ax_c.set_xlabel(r"fit window truncated to $\chi \leq$ cut")
    ax_c.set_ylabel("parameter / its full-window value")
    ax_c.set_title("(c)  but $B$ is badly conditioned. Refitting on shorter windows\n"
                   rf"moves it by up to ${{\times}}"
                   rf"{max(m['Bdrift'] for m in good):.0f}$, because it is an "
                   r"extrapolation back" "\n" r"through a factor $e^{\xi^\ast}$. "
                   r"$A$ and $\xi^\ast$ barely move.", fontsize=10, loc="left")
    ax_c.legend(fontsize=8, loc="upper right")

    # ---- (d) the crossover, and the amplification identity --------------
    xi = np.array([m["xi"] for m in good])
    for m in good:
        v = m["xitrace"][np.isfinite(m["xitrace"])]
        ax_d.plot([m["thetaL"]] * 2, [v.min(), v.max()], color=col[m["file"]], lw=2)
        ax_d.plot(m["thetaL"], m["xi"], "o", color=col[m["file"]], ms=7, mec="k",
                  mew=0.6, zorder=6)
        ax_d.text(m["thetaL"] * 1.004, m["xi"], m["short"], fontsize=6,
                  color=col[m["file"]], va="center")
    g = np.array([246, 315])
    mu_xi, s_xi = gm("xi"), gs("xi")
    ax_d.plot(g, [mu_xi] * 2, "k-", lw=2, label=rf"$\xi^\ast={mu_xi:.1f}$")
    ax_d.fill_between(g, mu_xi / s_xi, mu_xi * s_xi, color="k", alpha=0.10, lw=0,
                      label=rf"${{\times}}{s_xi:.2f}$, i.e. $\pm{np.std(xi):.1f}$")
    ax_d.set_xlim(246, 318); ax_d.set_ylim(0, 13)
    ax_d.set_xlabel(r"$\theta_L$  (rad)")
    ax_d.set_ylabel(r"$\xi^\ast=\mu\theta_L\chi^\ast$   (capstan gain at crossover)")
    ax_d.set_title("(d)  what is well determined instead: the crossover. Bars are\n"
                   "the truncation range. No dependence on $\\theta_L$ is resolvable.",
                   fontsize=10, loc="left")
    ax_d.legend(fontsize=8, loc="lower right")
    ax_d.text(0.03, 0.95,
              r"$\ln B=\ln(A\chi^\ast)-\ln\!\left(e^{\xi^\ast}-1\right)$" "\n"
              rf"$\mathrm{{std}}(\xi^\ast)={np.std(xi):.2f}$,   "
              rf"$\mathrm{{std}}(\ln B)={np.std(np.log([m['B'] for m in good])):.2f}$"
              "\n" r"$B$'s three decades $=$ this $\pm2$, exponentiated",
              transform=ax_d.transAxes, va="top", fontsize=8.5,
              bbox=dict(fc="w", ec="0.6", alpha=0.9))

    fig.suptitle(r"$B$ is the wrong parameter to hypothesise: it is "
                 r"$A\chi^\ast e^{-\xi^\ast}$, and the exponential turns a "
                 "±30% crossover into three decades.", fontsize=12.5, y=0.975)
    p = os.path.join(OUT, "B_diagnostic.png")
    fig.savefig(p); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
