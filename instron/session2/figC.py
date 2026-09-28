"""
ONE FIGURE:  is the capstan exponent  C = mu theta_L ?

The claim under test is that the exponential in

    F_s(chi) = A chi + B ( e^{C chi} - 1 ),      chi = x/L = sin(phi)

has its rate fixed entirely by things that were measured independently:
the friction coefficient of the paper (mu_static = 0.16, measured on a
tilt test; static because the pulls visibly stick-slip) and the total wrap
angle theta_L (counted turns for set 1, tightest-minus-loosened for set 2).
Nothing in C is fitted.

So: FREEZE C = mu_static theta_L and fit only the two amplitudes A, B.
Four panels, four independent ways of asking the same question.

  (a) every run, measured against the frozen-exponent two-parameter fit.
  (b) the collapse.  Undo the fitted amplitudes -- plot
          Y = (F_s - A chi)/B + 1
      against  xi = mu_static theta_L chi.  If C = mu theta_L then every run,
      whatever its theta_L, must fall on the single universal line Y = e^xi.
  (c) how much freezing costs.  Refit with C = k mu theta_L for a sweep of k
      and plot rms(k)/rms(free 3-parameter).  The minimum locates the exponent
      the data actually wants, in units of the prediction.
  (d) the free 3-parameter C plotted against mu theta_L, with the 1:1 line and
      the rival factor-of-two counting.

Run:  python figC.py      ->  figs/C_proof.png
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from processing import (parse, load, nominal, draw, theta_L, usable, fit,
                        m_free, colors, MU_STATIC, MU_KINETIC, QUANT, HERE, OUT,
                        NMC, RANGES)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _e(z):
    return np.exp(np.clip(z, -700, 700))


def m_frozen(p, chi, C):
    """A chi + B(e^{C chi} - 1) with C HELD FIXED.  Free: A, B."""
    A, B = _e(p)
    return A * chi + B * (_e(C * chi) - 1.0)


# k = C / (mu theta_L).  The prediction is k = 1; the rejected two-face
# counting of the contact would be k = 2.
KS = np.concatenate([np.linspace(0.30, 0.88, 8), np.linspace(0.90, 1.50, 25),
                     np.linspace(1.6, 3.0, 8)])


def main():
    metas = [parse(p) for p in sorted(glob.glob(os.path.join(HERE, "set*.csv")))]
    metas.sort(key=lambda m: (m["set"], m["run"]))
    nom = nominal()
    rng = np.random.default_rng(0)
    d = draw(rng, NMC)

    print("=" * 96)
    print(f"FROZEN EXPONENT  C = mu_static theta_L,  mu_static = {MU_STATIC} "
          "(measured).  Only A and B are fitted.")
    print("=" * 96)
    print(f"{'run':<18}{'thetaL':>8}{'C_pred':>8}{'C_free':>8}{'k_C':>6}"
          f"{'rms free':>10}{'rms frozen':>12}{'penalty':>9}{'A (N)':>9}{'B (N)':>11}")

    for m in metas:
        x, F, Fs, tare = load(m)
        m.update(x=x, F=F, Fs=Fs, chi=x / m["L"])
        m["thetaL"] = theta_L(m, m["L"], nom["t"], nom["r0"])
        m["Cpred"] = MU_STATIC * m["thetaL"]

        # theta_L uncertainty -> C_pred uncertainty (the dominant one)
        L = m["L"] + d["dL"]
        th = theta_L(m, L, d["t"], d["r0"], rng.uniform(-1, 1, NMC))
        m["Cpred_mc"] = np.percentile(MU_STATIC * th, [5, 50, 95])

        k = usable(m["Fs"])
        c, y = m["chi"][k], m["Fs"][k]
        m["cfit"], m["yfit"] = c, y

        p, rf = fit(m_free, [np.log(2), np.log(1e-3), np.log(5)], c, y)
        m["free"] = dict(zip("ABC", np.exp(p))) | dict(rms=rf)

        pp, rr = fit(m_frozen, [np.log(1.0), np.log(1e-3)], c, y, args=(m["Cpred"],))
        A, B = np.exp(pp)
        m["frozen"] = dict(A=A, B=B, rms=rr, pen=rr / rf)

        print(f"{m['label']:<18}{m['thetaL']:8.1f}{m['Cpred']:8.1f}"
              f"{m['free']['C']:8.1f}{m['free']['C']/m['Cpred']:6.2f}"
              f"{rf:10.4f}{rr:12.4f}{rr/rf:9.2f}{A:9.3f}{B:11.2e}")

    pen = np.array([m["frozen"]["pen"] for m in metas])
    kc = np.array([m["free"]["C"] / m["Cpred"] for m in metas])
    print(f"\n  freezing the exponent at the PREDICTED value costs a median "
          f"{np.median(pen):.2f}x in rms\n  (range {pen.min():.2f}-{pen.max():.2f}) "
          f"relative to letting C float.  C itself spans "
          f"{min(m['free']['C'] for m in metas):.0f}-"
          f"{max(m['free']['C'] for m in metas):.0f},"
          f" i.e. e^C over {np.exp(min(m['free']['C'] for m in metas)):.0e} "
          f"to {np.exp(max(m['free']['C'] for m in metas)):.0e}.")
    print(f"  k_C = C_free/(mu theta_L): median {np.median(kc):.2f}, "
          f"range {kc.min():.2f}-{kc.max():.2f}")

    # ---- panel (c): the sweep ------------------------------------------
    print("\n  sweeping C = k mu theta_L ...")
    sweep = []
    for m in metas:
        row = [fit(m_frozen, [np.log(1.0), np.log(1e-3)], m["cfit"], m["yfit"],
                   args=(kk * m["Cpred"],))[1] / m["free"]["rms"] for kk in KS]
        sweep.append(row)
    sweep = np.array(sweep)
    pooled = np.exp(np.mean(np.log(sweep), axis=0))     # geometric mean
    kbest = KS[int(np.argmin(pooled))]
    # where the pooled penalty is within 10% of its minimum
    inside = KS[pooled <= 1.10 * pooled.min()]
    print(f"  pooled penalty minimised at k = {kbest:.2f}; within 10% of the "
          f"minimum for k in [{inside.min():.2f}, {inside.max():.2f}]")
    print(f"  penalty at k=0.5: {np.interp(0.5,KS,pooled):.2f}x   "
          f"at k=1: {np.interp(1.0,KS,pooled):.2f}x   "
          f"at k=2: {np.interp(2.0,KS,pooled):.2f}x")

    figure(metas, sweep, pooled, kbest, inside)


def figure(metas, sweep, pooled, kbest, inside):
    col = colors(metas)
    fig = plt.figure(figsize=(13.6, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---------------- (a) every run vs the frozen-exponent fit ----------
    #  stacked with a multiplicative offset so twelve curves stay legible
    OFF = 1.8
    tags = []
    for i, m in enumerate(metas):
        c, y = m["cfit"], m["yfit"]
        sc = OFF ** i
        ax_a.semilogy(c, y * sc, color=col[m["file"]], lw=1.4, alpha=0.95)
        g = np.linspace(c.min(), c.max(), 300)
        f = m["frozen"]
        ax_a.semilogy(g, sc * m_frozen([np.log(f["A"]), np.log(f["B"])], g, m["Cpred"]),
                      "k--", lw=1.0, zorder=6)
        # tags.append([c.max(), np.log10(y[-1] * sc),
        #              f"{m['label']}  $\\theta_L$={m['thetaL']:.0f}", col[m["file"]]])
    # nudge labels apart in log space so none overlap
    # tags.sort(key=lambda t: t[1])
    # span = tags[-1][1] - tags[0][1]
    # gap = span / 15
    # for j in range(1, len(tags)):
    #     tags[j][1] = max(tags[j][1], tags[j - 1][1] + gap)
    XL = 0.345
    for xt, yt, txt, cc in tags:
        ax_a.plot([xt, XL - 0.004], [10 ** yt, 10 ** yt], color=cc, lw=0.5,
                  ls=":", alpha=0.8, zorder=2)
        ax_a.text(XL, 10 ** yt, txt, color=cc, fontsize=6.0, va="center", ha="left")

    ax_a.plot([], [], "k--", lw=1.0,
              label=r"$F=A\chi + B(e^{\mu_s\theta_L\chi}-1)$ two parameter fit")
    ax_a.set_xlabel(r"$\chi = x/L$")
    ax_a.set_ylabel(r"$F\;\times\;1.8^{\,\rm run}$   (shifted for visual separation)")
    ax_a.set_title("(a)  Two parameter fits with the predicted exponent",fontsize=10, loc="left")
    ax_a.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax_a.set_xlim(-0.01, 0.3)
    ax_a.set_ylim(2 * QUANT / 4 * 0.55, None)

    # ---------------- (b) the collapse ----------------------------------
    #  Only show the range where the exponential term is itself above the
    #  load-cell step, otherwise the plot is dominated by quantization noise
    #  in a quantity that is 1 by construction.
    ximax = 0
    for m in metas:
        c, y = m["cfit"], m["yfit"]
        f = m["frozen"]
        Y = (y - f["A"] * c) / f["B"] + 1.0
        xi = m["Cpred"] * c
        k = (Y > 0) & (f["B"] * (np.exp(np.clip(xi, 0, 700)) - 1) > QUANT / 4)
        ax_b.semilogy(xi[k], Y[k], ".", color=col[m["file"]], ms=1.0, alpha=0.6)
        if k.any():
            ximax = max(ximax, xi[k].max())
    g = np.linspace(0, ximax * 1.02, 200)
    ax_b.semilogy(g, np.exp(g), "k-", lw=2.2, zorder=8,
                  label=r"$C = \mu_s\theta_L$   as predicted")

    ax_b.set_xlim(0, ximax * 1.02)
    ax_b.set_ylim(0.7, np.exp(ximax) * 12)
    ax_b.set_xlabel(r"$\mu_s\theta_L\chi$   (the predicted exponent)")
    ax_b.set_ylabel(r"$e^{C\chi}=\frac{F-A}{\chi}+1$ (the curve-fit exponential term)")
    ax_b.set_title("(b)  Extracted exponential term", fontsize=10, loc="left")
    ax_b.legend(fontsize=7.5, loc="upper left")
    ax_b.text(0.985, 0.03, "scatter below the line at small $\\xi$ is the\n"
              "0.025 N load-cell step, amplified by $1/B$",
              transform=ax_b.transAxes, fontsize=6.8, color="0.35",
              ha="right", va="bottom")

    # ---------------- (c)  ------------------------------
    # i tried something for looking at the A collapse
    ximax = 0
    for m in metas:
        c, y = m["cfit"], m["yfit"] # c = chi, y = Fs
        f = m["frozen"] # frozen fit parameters A, B
        # Y = (y - f["A"] * c) / f["B"] + 1.0
        xi = m["Cpred"] * c
        Y = y - f["B"] * (np.exp(np.clip(xi, 0, 700)) - 1)
        k = (Y > 0) & (f["B"] * (np.exp(np.clip(xi, 0, 700)) - 1) > QUANT / 4)
        ax_c.plot(xi[k], Y[k], ".", color=col[m["file"]], ms=1.0, alpha=0.6)
        if k.any():
            ximax = max(ximax, xi[k].max())
    g = np.linspace(0, ximax * 1.02, 200)
    # ax_c.semilogy(g, np.exp(g), "k-", lw=2.2, zorder=8,label=r"$C = \mu_s\theta_L$   as predicted")

    ax_c.set_xlim(0, ximax * 1.02)
    ax_c.set_ylim(0.7, np.exp(ximax) * 12)
    ax_c.set_xlabel(r"$\mu_s\theta_L\chi$   (the predicted exponent)")
    ax_c.set_ylabel(r"$e^{C\chi}=\frac{F-A}{\chi}+1$ (the curve-fit exponential term)")
    ax_c.set_title("(c)  Extracted linear term", fontsize=10, loc="left")
    ax_c.legend(fontsize=7.5, loc="upper left")
    ax_c.text(0.985, 0.03, "scatter below the line at small $\\xi$ is the\n"
                "0.025 N load-cell step, amplified by $1/B$",
                transform=ax_c.transAxes, fontsize=6.8, color="0.35",
                ha="right", va="bottom")
    # # ---------------- (d) C_free against the prediction -----------------
    # lo = min(min(m["Cpred_mc"][0] for m in metas),
    #          min(m["free"]["C"] for m in metas)) * 0.85
    # hi = max(max(m["Cpred_mc"][2] for m in metas),
    #          max(m["free"]["C"] for m in metas)) * 1.1
    # g = np.array([lo, hi])
    # ax_d.plot(g, g, "k--", lw=1.8, label=r"$C=\mu_s\theta_L$  (predicted)")
    # ax_d.plot(g, 2 * g, color="crimson", ls="-.", lw=1.4, label=r"$C=2\mu_s\theta_L$")
    # ax_d.plot(g, g * MU_KINETIC / MU_STATIC, color="0.45", lw=1.0,
    #           label=rf"$C=\mu_k\theta_L$ ($\mu_k$={MU_KINETIC})")
    # for m in metas:
    #     p5, p50, p95 = m["Cpred_mc"]
    #     ax_d.plot([p5, p95], [m["free"]["C"]] * 2, color=col[m["file"]], lw=2.4,
    #               solid_capstyle="butt", alpha=0.85)
    #     ax_d.plot(p50, m["free"]["C"], "o", color=col[m["file"]], ms=6,
    #               mec="k", mew=0.6, zorder=6)
    # ax_d.set_xlim(lo, hi)
    # ax_d.set_ylim(lo, hi)
    # ax_d.set_xlabel(r"predicted  $\mu_s\theta_L$   (bars: $\theta_L$ from $t,r_0,L$, turns)")
    # ax_d.set_ylabel(r"measured  $C$  (free 3-parameter fit)")
    # ax_d.set_title("(d)  measured exponent against the prediction, run by run.\n"
    #                "No fitted quantity enters the horizontal axis.",
    #                fontsize=10, loc="left")
    # ax_d.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        r"The capstan exponent is $C=\mu\theta_L$: "
        rf"$\mu_s={MU_STATIC}$ measured on a tilt test, $\theta_L$ counted from the winding. "
        "Nothing on the right-hand side is fitted.",
        fontsize=12.5, y=0.985)
    fig.text(0.5, 0.945,
             rf"Freezing $C$ there raises the residual by only "
             rf"{np.median([m['frozen']['pen'] for m in metas]) - 1:.0%} over the free "
             "three-parameter fit; halving or doubling it costs "
             f"{np.interp(0.5,KS,pooled)-1:.0%} and {np.interp(2.0,KS,pooled)-1:.0%}."
             r"  $C$ itself runs from 38 to 65, so $e^{C}$ spans $10^{16}$ to $10^{28}$.",
             ha="center", fontsize=10.5, color="0.25")
    p = os.path.join(OUT, "C_proof.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
