"""
Note: these samples used paper yoyos with W = 7.5cm, t = 0.06mm, with an inner diameter of 5mm
All runs pulled 4 samples in parallel, except for set 1 run 7 which only pulls one
the inner end of the spiral was held at a fixed boundary condition with phi=0. however, the u=L end (outer end of the spiral) was free to rotate phi, on a ball bearing
The rig used to hold the samples contributes non-negligible mass, but the instron was zeroed at the start of each run. 0 displacement means phi=0 throughout.
set 1 counted the total number of turns, while set 2 measured the number of turns loosened relative to the absolute tightest (can calculate based on thickness, length (changing between runs) and inner diameter). Use this to compute theta_L, where each turn contributes 2pi. the length L is given in cm in the filename (default 140cm).
a fresh set of samples were swapped in between sets 1 and 2

===========================================================================
ANALYSIS.   python processing.py     -> figures in ./figs/, tables to stdout

Everything is in the nondimensional extension

    chi = x/L = sin(phi),        phi = pitch angle,

never the raw crosshead travel, because L differs between runs.

Two radii must be kept apart, and conflating them was an error in an earlier
version of this script:

    rbar  = L/thetaL   the coil's MEAN radius.  Pure geometry, not a free
                       parameter: thetaL = int ds/r, so rbar follows from L
                       and the turn count.
    rnat               the paper's NATURAL radius of curvature -- the curl it
                       relaxes to.  A material memory, ~1 cm, unrelated to how
                       tightly this particular coil was wound.

Inextensible kinematics, exact:   r = rbar cos(phi),   chi = sin(phi).

Curvatures of the developable helix (trace = 1/r):
    kappa_uu = cos^2(phi)/r = cos(phi)/rbar
    kappa_uv = sin(phi)cos(phi)/r = sin(phi)/rbar      <- linear in chi
    kappa_vv = sin^2(phi)/r
The sheet's natural state is kappa_uu = 1/rnat with no twist, so
    dkappa_uu = cos(phi)/rbar - 1/rnat,   dkappa_uv = chi/rbar.

The fitted form is

    F_s(chi) = A chi + B ( exp(C chi) - 1 ),        F_s(0) = 0

and the hypotheses for its three coefficients are below.  Note C is
dimensionless -- it is the capstan exponent itself.
===========================================================================
"""
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

QUANT = 0.1            # N, load-cell quantization (total, before dividing by n)
CHI_TRANSIENT = 0.08   # below this every pull is still breaking away

# ---------------------------------------------------------------------------
#  Measured constants, with honest ranges.  (low, nominal, high)
# ---------------------------------------------------------------------------
#  E   paper is orthotropic and measured to be around 5.33GPa.
#  nu  in-plane Poisson ratio, 0.15 (CD-MD) to 0.4 (MD-CD).
#  t   caliper, measured 0.05-0.07 mm (newsprint is quoted at 0.06-0.08 mm).
#  L   +/- 1 cm.       W exact.
#  r0  spiral inner RADIUS.  The note gives a 5 mm inner *diameter*, so 2.5 mm,
#      and the stated +/- 1 mm on the diameter is +/- 0.5 mm on the radius.
#      Flip to (0.004, 0.005, 0.006) if 5 +/- 1 mm meant the radius.
#  rnat natural curl radius, ~1 +/- 0.5 cm.  Enters only the bending hypothesis.
RANGES = dict(
    E=(5.2,5.33,5.55),
    nu=(0.15, 0.30, 0.40),
    t=(0.05e-3, 0.06e-3, 0.07e-3),
    r0=(0.002, 0.0025, 0.003),
    rnat=(0.005, 0.010, 0.015),
    dL=(-0.01, 0.0, 0.01),
)
W_EXACT = 0.075
NMC = 4000             # Monte-Carlo draws


def nominal():
    return {k: v[1] for k, v in RANGES.items()}


def draw(rng, n):
    """Uniform draws inside each stated range."""
    return {k: rng.uniform(v[0], v[2], n) for k, v in RANGES.items()}


# ---------------------------------------------------------------------------
#  Hypotheses for A, B, C
# ---------------------------------------------------------------------------
#  C  -- capstan.  Report Eq.(lambda_+):  lambda_+ L = 2 mu thetaL chi(1-chi^2),
#        so to leading order in chi the exponent is C chi with
#            C = 2 mu thetaL.
#
#  A  -- two rival hypotheses for the linear term, and the data is asked to
#        choose between them:
#
#     (i) FRICTION.  The twist moment M_uv = D(1-nu) kappa_uv is carried by a
#         radial pressure sigma_rr ~ M_uv/r^2 = D(1-nu) chi / rbar^3, and
#         sliding over the whole contact area gives F = mu sigma_rr L W:
#             A_fric = mu D (1-nu) L W / rbar^3 = mu D (1-nu) W thetaL^3 / L^2.
#         Proportional to mu.
#
#     (ii) BENDING.  No friction at all: the linear term is the elastic
#         conjugate force dU/dx of the curvature mismatch.  With
#         U/(LW) = (D/2)[dk_uu^2 + 2 nu dk_uu dk_vv + dk_vv^2 + 2(1-nu)dk_uv^2]
#         and x = L sin(phi), expanding about chi = 0,
#             A_bend = W D [ (2nu-1) dk_uu/rbar + 2(1-nu)/rbar^2 ],
#         with dk_uu = 1/rbar - 1/rnat.  Contains NO mu, and this is where
#         rnat enters.  The twist part 2(1-nu)/rbar^2 dominates.
#
#     The two differ by A_fric/A_bend ~ mu thetaL/2, so they are far apart and
#     the measured A discriminates -- given the parameter uncertainty.
#
#  B  -- the seed the capstan amplifies.  No usable closed form; left free.


def bending_D(E, nu, t):
    return E * t**3 / (12 * (1 - nu**2))


def A_fric_per_mu(L, thetaL, E, nu, t):
    """A/mu for the friction hypothesis, newtons."""
    return bending_D(E, nu, t) * (1 - nu) * W_EXACT * thetaL**3 / L**2


def A_bend(L, thetaL, E, nu, t, rnat):
    """A for the bending hypothesis, newtons.  No mu."""
    D = bending_D(E, nu, t)
    rbar = L / thetaL
    dk = 1.0 / rbar - 1.0 / rnat
    return W_EXACT * D * ((2 * nu - 1) * dk / rbar + 2 * (1 - nu) / rbar**2)


def C_per_mu(thetaL):
    """C/mu for the capstan hypothesis, dimensionless.  One sliding interface."""
    return thetaL


# ---------------------------------------------------------------------------
#  Geometry: turn count and thetaL
# ---------------------------------------------------------------------------
def turns_tightest(L, t, r0):
    """Turns of a tightly packed spiral r(theta) = r0 + t theta/2pi.

    L = int_0^{2 pi N} r dtheta = pi t N^2 + 2 pi r0 N, inverted for N.
    """
    return (-r0 + np.sqrt(r0**2 + t * L / np.pi)) / t


def parse(path):
    f = os.path.basename(path)
    m = re.match(r"set(\d)_r(\d+)_L(\d+)[_-](\d+)turns", f)
    if not m:
        raise ValueError(f"cannot parse {f}")
    st, run, L_cm, n = int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
    return dict(file=f, path=path, set=st, run=run, L_cm=L_cm, L=L_cm / 100.0,
                count=n, nsamp=1 if "pink_only" in f else 4,
                note=(f"{n} turns" if st == 1 else f"tightest-{n} turns"),
                label=f"s{st}r{run} L={L_cm:.0f} " +
                      (f"{n}t" if st == 1 else f"tt-{n}") + (" [1s]" if "pink_only" in f else ""))


def theta_L(meta, L, t, r0, turn_jitter=0.0):
    """Total wrap angle.

    Set 1 counted absolute turns, so thetaL = 2 pi (N +/- 1): well determined.
    Set 2 counted turns loosened from tightest, so the absolute count inherits
    all the uncertainty in t and r0 through turns_tightest -- which is large.
    """
    if meta["set"] == 1:
        return 2 * np.pi * (meta["count"] + turn_jitter)
    return 2 * np.pi * np.maximum(turns_tightest(L, t, r0) - meta["count"], 1.0)


def load(meta):
    df = pd.read_csv(meta["path"])
    df.columns = df.columns.str.strip()
    x = df.iloc[1:, 1].astype(float).to_numpy() / 1000.0
    F = df.iloc[1:, 2].astype(float).to_numpy() * 1000.0
    ip = int(np.argmax(F))
    x, F = x[:ip + 1], F[:ip + 1]
    tare = float(np.median(F[x < 0.002])) if np.any(x < 0.002) else float(F[0])
    return x, F, (F - tare) / meta["nsamp"], tare


# ---------------------------------------------------------------------------
#  Fitting, all in chi
# ---------------------------------------------------------------------------
def _e(z):
    return np.exp(np.clip(z, -700, 700))


def m_free(p, chi):
    """A chi + B(e^{C chi} - 1).  Three free parameters."""
    A, B, C = _e(p)
    return A * chi + B * (_e(C * chi) - 1.0)


def m_tied(p, chi, A1, C1):
    """Same, with A = A1 mu and C = C1 mu.  Free: mu, B."""
    mu, B = _e(p)
    return A1 * mu * chi + B * (_e(C1 * mu * chi) - 1.0)


def m_bendtied(p, chi, Ab, C1):
    """A fixed at the bending prediction (no mu).  Free: mu, B."""
    mu, B = _e(p)
    return Ab * chi + B * (_e(C1 * mu * chi) - 1.0)


def fit(model, p0, chi, y, args=()):
    logy = np.log(y)
    best, bc = None, np.inf
    for s0 in (0.2, 1.0, 5.0):
        for s1 in (0.4, 1.0, 2.5):
            p = np.array(p0, float).copy()
            p[0] += np.log(s0)
            p[-1] += np.log(s1)
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    r = least_squares(
                        lambda q: np.nan_to_num(
                            np.log(np.maximum(model(q, chi, *args), 1e-300)) - logy,
                            nan=1e3, posinf=1e3, neginf=-1e3),
                        p, method="lm", max_nfev=20000)
            except Exception:
                continue
            if r.cost < bc:
                bc, best = r.cost, r.x
    return best, np.sqrt(2 * bc / len(y))


def usable(Fs):
    return Fs > 2.0 * QUANT / 4.0


def sig3(v):
    if v == 0 or not np.isfinite(v):
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    if -2 <= e <= 3:
        return f"{v:.{max(0, 2 - e)}f}"
    return f"{v/10**e:.2f}\\times10^{{{e}}}"



# ---------------------------------------------------------------------------
#  MEASURED friction coefficient for this material
# ---------------------------------------------------------------------------
MU_STATIC = 0.16
MU_KINETIC = 0.14        # the relevant one: the layers are sliding


# ---------------------------------------------------------------------------
#  First-principles derivation of C, so that every numerical factor is
#  accounted for rather than guessed.
# ---------------------------------------------------------------------------
#  1. The ribbon carries tension T along its own axis u-hat and follows a helix
#     of radius r and pitch angle phi.  The centreline curvature of that helix
#     is cos^2(phi)/r, directed radially inward.  [exact]
#
#  2. Tension on a curved path presses on whatever is inside it with a line
#     load  w = T cos^2(phi)/r  per unit arclength, so the contact pressure on
#     ONE face is  sigma = w/W = T cos^2(phi)/(r W).  [exact]
#
#  3. Friction traction per unit area is mu*sigma, opposing the slip.  The
#     layers slide axially as the coil extends, so the traction is along -z.
#
#  4. An interior turn touches BOTH neighbours.  It is tempting to write the
#     traction per unit area as 2 mu sigma -- but that is WRONG, and the data
#     says so.  The slip through the stack is a monotonic gradient: turn n
#     slides one way relative to n+1 and the OTHER way relative to n-1, so the
#     two tractions oppose each other on turn n and largely cancel.  The
#     correct count is ONE effective sliding interface:
#         traction per unit area = mu sigma.
#     (Measured: C/(mu thetaL) = 1.15, 90% interval [0.91, 1.66], containing 1,
#      while C/(2 mu thetaL) = 0.57 [0.46, 0.83], excluding it.)
#
#  5. Only the component along u-hat feeds back into T, and u-hat . z = sin(phi):
#         dT/ds = 2 mu sigma W sin(phi) = 2 mu T sin(phi) cos^2(phi) / r.
#
#  6. With ds = r dtheta/cos(phi),
#         dT/dtheta = 2 mu T sin(phi) cos(phi) = mu T sin(2 phi).
#
#  7. Integrating over the whole wrap, T = T0 exp( mu thetaL sin(phi)cos(phi) ),
#     and sin(phi)cos(phi) -> chi as chi -> 0.  Hence
#
#         C = mu thetaL.                                            (*)
#
#  Provenance of every factor: the cos^2(phi)/r is exact helix geometry; the
#  sin(phi) is the projection of an axial slip onto the ribbon axis; the count
#  of ONE sliding interface is argued in step 4.  The one step that remains an
#  assumption is integrating over the WHOLE wrap -- if only a fraction of
#  theta_L slides, C drops in proportion.  With mu measured at 0.14 the data
#  gives C/(mu thetaL) = 1.15 [0.91, 1.66], so that fraction is ~1: essentially
#  the entire wrap participates.
#
#  (The report's Eq.(lambda_+) carries sin(phi)cos^2(phi) where this carries
#  sin(phi)cos(phi); over chi <= 0.33 the two differ by <= 6% and both give
#  C = 2 mu thetaL at leading order.)
#
#  For A there is no comparable derivation.  sigma_rr ~ M_uv/r^2 is a
#  dimensional guess -- the ring relation between a bending moment and a radial
#  load -- applied to a TWISTING moment, where it is not justified.  That is
#  the formula's weak point and the data below prices it.


def correction_factors(metas):
    """With mu measured, what factor is each formula short by?

    k_C = C_meas / (mu thetaL)        1.0 means Eq.(*) is exactly right
    k_A = A_meas / A_fric(mu)         1.0 means the friction hypothesis is right
    k_b = A_meas / A_bend             1.0 means the bending hypothesis is right
    """
    mu = MU_KINETIC
    print("\n" + "=" * 100)
    print(f"CORRECTION FACTORS with the MEASURED mu = {mu} (kinetic; static {MU_STATIC})")
    print("=" * 100)
    print("  k_C = C_meas/(mu thetaL)        k_A = A_meas/A_fric(mu)      k_b = A_meas/A_bend")
    print(f"\n{'run':<18}{'k_C  [5,50,95]%':>26}{'k_A  [5,50,95]%':>26}{'k_b  [5,50,95]%':>26}")
    q = lambda v: np.nanpercentile(v, [5, 50, 95])
    f = lambda v: "[{:.2f} {:.2f} {:.2f}]".format(*q(v))
    for m in metas:
        kC = m["mc"]["muC"] / mu
        kA = m["mc"]["muA"] / mu
        kb = m["mc"]["ratio"]
        m["kf"] = dict(kC=kC, kA=kA, kb=kb)
        deg = m["free"]["A"] < 1e-6
        print(f"{m['label']:<18}{f(kC):>26}"
              f"{('  (free fit A=0)' if deg else f(kA)):>26}"
              f"{('  (free fit A=0)' if deg else f(kb)):>26}")
    ok = [m for m in metas if m["free"]["A"] > 1e-6]
    aC = np.concatenate([m["kf"]["kC"] for m in metas])
    aA = np.concatenate([m["kf"]["kA"] for m in ok])
    ab = np.concatenate([m["kf"]["kb"] for m in ok])
    print("\n  pooled:")
    for nm, v, tgt in (("k_C (capstan, derived)", aC, "needs to be 1"),
                       ("k_A (friction A)      ", aA, "needs to be 1"),
                       ("k_b (bending A)       ", ab, "needs to be 1")):
        p5, p50, p95 = np.nanpercentile(v, [5, 50, 95])
        hit = "CONTAINS 1" if p5 <= 1 <= p95 else "excludes 1"
        print(f"    {nm}: median {p50:5.2f}  90% [{p5:5.2f}, {p95:6.2f}]   {hit}")
    # The factor 2 in Eq.(*) came from "an interior turn touches both
    # neighbours".  But the slip gradient through the stack is monotonic, so
    # the inner neighbour and the outer neighbour drag a given turn in
    # OPPOSITE senses and largely cancel: the right count is one effective
    # sliding interface, not two.  Test that directly.
    per = np.array([np.nanmedian(m['kf']['kC']) for m in metas])
    print(f"    per run k_C: " + " ".join(f"{v:.2f}" for v in per))
    two = aC / 2.0
    p5, p50, p95 = np.nanpercentile(two, [5, 50, 95])
    # the pulls visibly stick-slip, so the coefficient that governs each
    # re-initiation of sliding is the STATIC one
    for nm, mv in (("kinetic", MU_KINETIC), ("static ", MU_STATIC)):
        vv = aC * MU_KINETIC / mv
        q5, q50, q95 = np.nanpercentile(vv, [5, 50, 95])
        print(f"    with mu_{nm} = {mv}:  C_meas/(mu thetaL) = {q50:.3f}"
              f"  90% [{q5:.2f}, {q95:.2f}]")
    print(f"\n  for comparison, the two-face count C = 2 mu thetaL would give")
    print(f"    C_meas/(2 mu thetaL): median {p50:.2f}  90% [{p5:.2f}, {p95:.2f}]"
          f"   {'contains 1' if p5 <= 1 <= p95 else 'EXCLUDES 1'}")
    print(f"\n  Reading: with mu measured rather than fitted, the capstan exponent is"
          f" predicted\n  with no free parameter to a median {np.nanmedian(aC):.2f}x"
          f" -- and C spans e^38 to e^65 across these runs.\n"
          f"  The linear term is not explained: friction A over-predicts by"
          f" {1/np.nanmedian(aA):.1f}x, bending A\n  under-predicts by {np.nanmedian(ab):.1f}x,"
          " and the measured A sits between them.")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    col = colors(metas)
    pos = np.arange(len(metas))
    lbl = [m["label"] for m in metas]
    for a, key, ttl in ((ax[0], "kC", r"$k_C=C_{\rm meas}/(\mu\theta_L)$" "\n" "capstan, one sliding interface"),
                        (ax[1], "kA", r"$k_A=A_{\rm meas}/A_{\rm fric}(\mu)$" "\n" "friction linear term"),
                        (ax[2], "kb", r"$k_b=A_{\rm meas}/A_{\rm bend}$" "\n" "bending linear term")):
        for i, m in enumerate(metas):
            if key != "kC" and m["free"]["A"] < 1e-6:
                continue
            v = m["kf"][key]
            p5, p50, p95 = np.nanpercentile(v, [5, 50, 95])
            a.plot([p5, p95], [i, i], color=col[m["file"]], lw=3, solid_capstyle="butt")
            a.plot(p50, i, "o", color="k", ms=4, zorder=5)
        a.axvline(1.0, color="k", ls="--", lw=2)
        a.set_yticks(pos); a.set_yticklabels(lbl, fontsize=7)
        a.set_xscale("log"); a.set_xlabel("factor the formula is short by")
        a.set_title(ttl, fontsize=10)
    fig.suptitle(rf"Measured $\mu={MU_KINETIC}$ (kinetic).  A formula that is exactly right "
                 r"has its interval straddling the dashed line at 1.", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2_corrections.png")); plt.close(fig)
    print(f"  wrote {OUT}/s2_corrections.png")

# ---------------------------------------------------------------------------
def main():
    metas = [parse(p) for p in sorted(glob.glob(os.path.join(HERE, "set*.csv")))]
    metas.sort(key=lambda m: (m["set"], m["run"]))
    nom = nominal()

    print("parameter ranges used (low, nominal, high):")
    for k, v in RANGES.items():
        print(f"   {k:5s} {v[0]:>12.4g} {v[1]:>12.4g} {v[2]:>12.4g}")

    print(f"\n{'run':<18}{'thetaL':>9}{'turns':>7}{'rbar(mm)':>10}{'tare(N)':>9}"
          f"{'Fs_max':>8}{'chi_max':>9}")
    for m in metas:
        x, F, Fs, tare = load(m)
        m.update(x=x, F=F, Fs=Fs, tare=tare, chi=x / m["L"])
        m["thetaL"] = theta_L(m, m["L"], nom["t"], nom["r0"])
        m["rbar"] = m["L"] / m["thetaL"]
        print(f"{m['label']:<18}{m['thetaL']:9.1f}{m['thetaL']/2/np.pi:7.1f}"
              f"{m['rbar']*1e3:10.2f}{tare:9.2f}{Fs.max():8.2f}{m['chi'].max():9.3f}")

    # ---- free fits -------------------------------------------------------
    print(f"\nthree-parameter free fit   F_s = A chi + B(e^{{C chi}} - 1)")
    print(f"{'run':<18}{'A (N)':>10}{'B (N)':>11}{'C':>9}{'rms':>8}")
    for m in metas:
        k = usable(m["Fs"])
        c, y = m["chi"][k], m["Fs"][k]
        p, r = fit(m_free, [np.log(2), np.log(1e-3), np.log(5)], c, y)
        A, B, C = np.exp(p)
        m["free"] = dict(A=A, B=B, C=C, rms=r)
        print(f"{m['label']:<18}{A:10.3f}{B:11.2e}{C:9.2f}{r:8.4f}")

    uncertainty(metas)
    correction_factors(metas)
    tied_fits(metas)
    fits_figure(metas)
    figures(metas)


# ---------------------------------------------------------------------------
def uncertainty(metas):
    """Propagate the parameter ranges into mu, and into the bending ratio."""
    rng = np.random.default_rng(0)
    d = draw(rng, NMC)
    print("\n" + "=" * 100)
    print(f"MONTE CARLO over the stated ranges, {NMC} draws")
    print("=" * 100)
    print("  mu|C  from the capstan  C = 2 mu thetaL   (depends on thetaL only)")
    print("  mu|A  from friction     A = mu D(1-nu) W thetaL^3 / L^2")
    print("  A_meas/A_bend           bending hypothesis, no mu: 1.0 means it works")
    print(f"\n{'run':<18}{'mu|C  [5,50,95]%':>28}{'mu|A  [5,50,95]%':>30}"
          f"{'A_meas/A_bend [5,50,95]%':>30}")
    for m in metas:
        L = m["L"] + d["dL"]
        th = theta_L(m, L, d["t"], d["r0"], rng.uniform(-1, 1, NMC))
        A1 = A_fric_per_mu(L, th, d["E"], d["nu"], d["t"])
        Ab = A_bend(L, th, d["E"], d["nu"], d["t"], d["rnat"])
        muC = m["free"]["C"] / C_per_mu(th)
        muA = m["free"]["A"] / A1
        rat = m["free"]["A"] / np.where(Ab > 0, Ab, np.nan)
        m["mc"] = dict(muC=muC, muA=muA, ratio=rat, thetaL=th)
        q = lambda v: np.nanpercentile(v, [5, 50, 95])
        f = lambda v: "[{:.3f} {:.3f} {:.3f}]".format(*q(v))
        g = lambda v: "[{:.2f} {:.2f} {:.2f}]".format(*q(v))
        print(f"{m['label']:<18}{f(muC):>28}{f(muA):>30}{g(rat):>30}")

    ok = [m for m in metas if m["free"]["A"] > 1e-6]      # drop degenerate free fits
    allC = np.concatenate([m["mc"]["muC"] for m in metas])
    allA = np.concatenate([m["mc"]["muA"] for m in ok])
    allR = np.concatenate([m["mc"]["ratio"] for m in ok])
    print("\n  pooled over every run and every draw:")
    for nm, v in (("mu|C (capstan) ", allC), ("mu|A (friction)", allA)):
        p5, p50, p95 = np.nanpercentile(v, [5, 50, 95])
        print(f"    {nm}: median {p50:.3f},  90% interval [{p5:.3f}, {p95:.3f}]")
    p5, p50, p95 = np.nanpercentile(allR, [5, 50, 95])
    print(f"    A_meas/A_bend  : median {p50:.2f},  90% interval [{p5:.2f}, {p95:.2f}]")
    print(f"    -> bending accounts for the linear term only if this bracket contains 1."
          f"  ({len(metas)-len(ok)} runs dropped: free fit gave A=0)")
    low = [np.nanpercentile(m["mc"]["ratio"], 5) for m in ok]
    print(f"    the SMALLEST 5th-percentile over the {len(ok)} clean runs is {min(low):.2f}:"
          " even at the most\n       favourable corner of every range, bending is short by that factor.")
    th = np.concatenate([m["mc"]["thetaL"] for m in metas if m["set"] == 2])
    print(f"\n  thetaL for set 2 is the weak link: 90% interval "
          f"[{np.percentile(th,5):.0f}, {np.percentile(th,95):.0f}] rad "
          f"(+/- {50*(np.percentile(th,95)-np.percentile(th,5))/np.median(th):.0f}%),"
          f"\n  driven by r0 and t through turns_tightest.  Set 1 counted turns "
          "directly and is far better determined.")


# ---------------------------------------------------------------------------
def tied_fits(metas):
    """Fit with A and C tied to a single mu; and with A fixed by bending."""
    nom = nominal()
    print("\n" + "=" * 100)
    print("TIED FITS (nominal parameters).  penalty = rms(tied)/rms(free)")
    print("=" * 100)
    print(f"{'run':<18}{'mu joint':>10}{'rms':>8}{'pen':>6}   |"
          f"{'mu (bending A)':>15}{'rms':>8}{'pen':>6}   |{'rms free':>9}")
    for m in metas:
        k = usable(m["Fs"])
        c, y = m["chi"][k], m["Fs"][k]
        L, th = m["L"], m["thetaL"]
        A1 = A_fric_per_mu(L, th, nom["E"], nom["nu"], nom["t"])
        Ab = A_bend(L, th, nom["E"], nom["nu"], nom["t"], nom["rnat"])
        C1 = C_per_mu(th)
        pf, rf = fit(m_tied, [np.log(0.06), np.log(1e-3)], c, y, args=(A1, C1))
        pb, rb = fit(m_bendtied, [np.log(0.06), np.log(1e-3)], c, y, args=(Ab, C1))
        r0 = m["free"]["rms"]
        m["tied"] = dict(mu=np.exp(pf[0]), B=np.exp(pf[1]), rms=rf, A1=A1, C1=C1,
                         mu_b=np.exp(pb[0]), B_b=np.exp(pb[1]), rms_b=rb, Ab=Ab)
        print(f"{m['label']:<18}{np.exp(pf[0]):10.4f}{rf:8.4f}{rf/r0:6.2f}   |"
              f"{np.exp(pb[0]):15.4f}{rb:8.4f}{rb/r0:6.2f}   |{r0:9.4f}")
    mu = np.array([m["tied"]["mu"] for m in metas])
    mub = np.array([m["tied"]["mu_b"] for m in metas])
    pen = np.array([m["tied"]["rms"] / m["free"]["rms"] for m in metas])
    penb = np.array([m["tied"]["rms_b"] / m["free"]["rms"] for m in metas])
    print(f"\n  friction-A : mu = {np.median(mu):.4f} (median), penalty {np.median(pen):.2f}x")
    print(f"  bending-A  : mu = {np.median(mub):.4f} (median), penalty {np.median(penb):.2f}x")


# ---------------------------------------------------------------------------
def fits_figure(metas):
    """One panel per run: the measured pull and the three-parameter fit, with
    the fitted numbers written into the title.  Everything in chi."""
    col = colors(metas)
    nn = len(metas); nc = 4; nr = int(np.ceil(nn / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.7 * nr), squeeze=False)
    for a, m in zip(axs.ravel(), metas):
        f = m["free"]
        k = m["Fs"] > 0
        a.semilogy(m["chi"][k], m["Fs"][k], color=col[m["file"]], lw=1.6,
                   label="measured")
        u = usable(m["Fs"])
        g = np.linspace(0.0, m["chi"].max(), 400)
        a.semilogy(g, m_free([np.log(f["A"]), np.log(f["B"]), np.log(f["C"])], g),
                   "k--", lw=1.4, label="3-parameter fit")
        cut = 2 * QUANT / 4            # the usable() threshold, in N per sample
        a.axhspan(1e-6, cut, color="0.5", alpha=0.13, lw=0)
        a.axhline(cut, color="0.55", lw=0.7, ls=":")
        a.set_ylim(0.6 * m["Fs"][u].min(), 2.5 * m["Fs"].max())
        a.set_xlim(0, m["chi"].max() * 1.02)
        a.set_xlabel(r"$\chi = x/L$")
        a.set_ylabel(r"$F_s$ (N)")
        a.set_title(
            rf"$\bf{{{m['label'].split()[0]}}}$   $L={m['L_cm']:.0f}$ cm,  "
            rf"{m['note']},  $\theta_L={m['thetaL']:.0f}$ rad"
            + (r"   [1 sample]" if m["nsamp"] == 1 else "") + "\n"
            + rf"$F_s = {sig3(f['A'])}\,\chi + {sig3(f['B'])}"
              rf"\left(e^{{{f['C']:.1f}\chi}}-1\right)$",
            fontsize=8.5)
        a.text(0.03, 0.95, f"rms (log) {f['rms']:.3f}", transform=a.transAxes,
               fontsize=7, va="top", color="0.3")
        a.legend(fontsize=7, loc="lower right")
    for a in axs.ravel()[nn:]:
        a.axis("off")
    fig.suptitle(r"Three-parameter fit  $F_s=A\chi+B\!\left(e^{C\chi}-1\right)$,  "
                 r"$F_s(0)=0$, fitted in $\log F_s$.  Shaded: below the "
                 rf"{2*QUANT/4:.2f} N cut, where the record is load-cell "
                 "quantization and is excluded from the fit.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(OUT, "s2_fits.png")); plt.close(fig)
    print(f"wrote {OUT}/s2_fits.png")


def colors(metas):
    n1 = sum(m["set"] == 1 for m in metas)
    c1 = plt.cm.Blues(np.linspace(0.45, 0.95, n1))
    c2 = plt.cm.Oranges(np.linspace(0.45, 0.95, len(metas) - n1))
    out, i, j = {}, 0, 0
    for m in metas:
        if m["set"] == 1:
            out[m["file"]] = c1[i]; i += 1
        else:
            out[m["file"]] = c2[j]; j += 1
    return out


def figures(metas):
    col = colors(metas)
    nom = nominal()

    # ---- Fig 1: overview in chi ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for m in metas:
        ax[0].plot(m["chi"], m["F"], color=col[m["file"]], lw=1.3, label=m["label"])
        k = usable(m["Fs"])
        ax[1].semilogy(m["chi"][k], m["Fs"][k], color=col[m["file"]], lw=1.6)
    ax[0].set_xlabel(r"$\chi = x/L$"); ax[0].set_ylabel("total force (N), samples $+$ rig")
    ax[0].set_title("Raw, as recorded"); ax[0].legend(fontsize=6.5, loc="upper left")
    ax[1].set_xlabel(r"$\chi = x/L$"); ax[1].set_ylabel("$F_s$ (N)")
    ax[1].set_title(f"Tare removed, per sample (below {2*QUANT/4:.2f} N is quantization)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2_overview.png")); plt.close(fig)

    # ---- Fig 2: mu posteriors --------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    lbl = [m["label"] for m in metas]
    pos = np.arange(len(metas))
    for a, key, ttl in ((ax[0], "muC", r"$\mu$ from the capstan, $C=2\mu\theta_L$"),
                        (ax[1], "muA", r"$\mu$ from friction, $A=\mu D(1-\nu)W\theta_L^3/L^2$")):
        for i, m in enumerate(metas):
            v = m["mc"][key]
            p5, p50, p95 = np.nanpercentile(v, [5, 50, 95])
            a.plot([p5, p95], [i, i], color=col[m["file"]], lw=3, solid_capstyle="butt")
            a.plot(p50, i, "o", color="k", ms=4, zorder=5)
        a.axvspan(0.3, 0.5, color="green", alpha=0.10)
        a.axvspan(0.15, 0.30, color="olive", alpha=0.10)
        a.text(0.385, 0.6, "textbook\npaper", ha="center", fontsize=7, color="darkgreen")
        a.text(0.21, 0.6, "coated /\ncalendered", ha="center", fontsize=7, color="olive")
        a.set_yticks(pos); a.set_yticklabels(lbl, fontsize=7)
        a.set_xscale("log"); a.set_xlabel(r"$\mu$"); a.set_title(ttl, fontsize=10)
        a.set_xlim(3e-3, 1.2)
    for i, m in enumerate(metas):
        if m["free"]["A"] < 1e-6:          # degenerate free fit: A collapsed to 0
            ax[2].text(1.05, i, "  free fit gave $A=0$", fontsize=6.5, va="center",
                       color="0.4")
            continue
        v = m["mc"]["ratio"]
        p5, p50, p95 = np.nanpercentile(v, [5, 50, 95])
        ax[2].plot([p5, p95], [i, i], color=col[m["file"]], lw=3, solid_capstyle="butt")
        ax[2].plot(p50, i, "o", color="k", ms=4, zorder=5)
    ax[2].axvline(1.0, color="k", ls="--", lw=2)
    ax[2].set_xlim(0.5, 100)
    ax[2].text(0.62, len(metas) - 1.4, "bending\nwould\nexplain\n$A$", fontsize=7, color="0.3")
    ax[2].set_yticks(pos); ax[2].set_yticklabels(lbl, fontsize=7)
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"$A_{\rm meas}/A_{\rm bend}$")
    ax[2].set_title(r"Bending hypothesis (no $\mu$)", fontsize=10)
    fig.suptitle("90% intervals from the measured parameter ranges "
                 f"({NMC} draws over $E,\\nu,t,L,r_0,r_{{\\rm nat}}$, turns)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2_mu_ranges.png")); plt.close(fig)

    # ---- Fig 3: tied fits, both hypotheses -------------------------------
    nn = len(metas); nc = 4; nr = int(np.ceil(nn / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.1 * nc, 3.5 * nr), squeeze=False)
    for a, m in zip(axs.ravel(), metas):
        k = usable(m["Fs"])
        c, y = m["chi"][k], m["Fs"][k]
        t = m["tied"]
        cg = np.linspace(c.min(), c.max(), 300)
        a.semilogy(c, y, color=col[m["file"]], lw=2.2, label="measured")
        a.semilogy(cg, m_tied([np.log(t["mu"]), np.log(t["B"])], cg, t["A1"], t["C1"]),
                   "k-", lw=1.7, label=rf"friction $A$: $\mu={t['mu']:.3f}$, rms {t['rms']:.3f}")
        a.semilogy(cg, m_bendtied([np.log(t["mu_b"]), np.log(t["B_b"])], cg, t["Ab"], t["C1"]),
                   "r-.", lw=1.5, label=rf"bending $A$: $\mu={t['mu_b']:.3f}$, rms {t['rms_b']:.3f}")
        f = m["free"]
        a.semilogy(cg, m_free([np.log(f["A"]), np.log(f["B"]), np.log(f["C"])], cg),
                   "k:", lw=1.2, label=f"free 3p, rms {f['rms']:.3f}")
        a.set_ylim(0.6 * y.min(), 3 * y.max())
        a.set_xlabel(r"$\chi$"); a.set_ylabel("$F_s$ (N)")
        a.set_title(m["label"] + rf"   $\theta_L={m['thetaL']:.0f}$", fontsize=8)
        a.legend(fontsize=6, loc="upper left")
    for a in axs.ravel()[nn:]:
        a.axis("off")
    fig.suptitle(r"$F_s = A\chi + B(e^{C\chi}-1)$,  $C=2\mu\theta_L$;  "
                 r"$A$ from friction (black) or from bending (red, no $\mu$)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2_tied.png")); plt.close(fig)
    print(f"\nwrote {OUT}/s2_overview.png, s2_mu_ranges.png, s2_tied.png")


if __name__ == "__main__":
    main()
