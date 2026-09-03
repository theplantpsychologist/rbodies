"""
energy_coupled_solver.py
========================

Successor to `numerical_solver.py` (see SOLVER_ARCHITECTURE.md).  The physics
that is new here is the bending/membrane **energy balance**, which supplies the
energy-minimizing coil radius r(x) and -- crucially -- a genuine inhomogeneous
normalization for the friction subsystem.  The old solver's central difficulty
was that its whole system was homogeneous, so an arbitrary anchor
(`sigma_uu(L,0) := 1`) had to set the scale; that anchor is gone.

WHAT CHANGED, AND WHY
---------------------

1. Energy minimization over rho := r / r_nat at fixed x.
       U(rho; x) = U_bend(rho, phi) + U_mem(eps),   eps = lambda - 1,
       lambda = sqrt(rho^2 + chi^2),                chi = x / L
   The extensible kinematics come from the helix arc length: a material line
   along u is a helix of radius r, turn theta_L, rise x, so its arc length is
   L*lambda.  Minimizing gives rho(x), hence the mean axial strain eps(x).
   Floored below by geometric jamming, rho >= rho_jam = t*theta_L^2/(4 pi L).

2. The mean strain becomes a DIRICHLET BOUNDARY CONDITION, not an anchor.
   The ribbon's total elongation is fixed by geometry: a(L,v) - a(0,v) =
   L*(lambda-1), where a is the u-displacement.  This is the physical
   replacement for the arbitrary scale anchor, and it is inhomogeneous, so the
   trivial all-zero field is no longer a solution.

3. DISPLACEMENT formulation instead of stress formulation.  Unknowns are the
   in-plane displacements a(u,v), b(u,v) rather than sigma_uu/uv/vv.  Three
   consequences, all of them fixes for known problems in the old solver:
     * Compatibility (eq:beltrami_michell, old Eq. 3) is satisfied
       IDENTICALLY and is dropped.  It was the source of the old solver's
       over-determination and of the inconsistent least-squares residual
       documented in SOLVER_ARCHITECTURE.md section 9.5.
     * The friction body force enters the Navier equations UNDIFFERENTIATED.
       Section 9.5's leading hypothesis was that differentiating bilinearly
       interpolated contact lookups produced the grid-scale noise; in this form
       nothing interpolated is ever differentiated.
     * The system is square and solved directly (splu), never by least squares.
   Verified independently: the old Eq. 3 is exactly the standard plane-stress
   Beltrami-Michell equation nabla^2(sig_kk) = -(1+nu) b_k,k with body force
   b = -f, so solving in displacements is the same problem, not a different one.

4. The layer-contact relation is applied as the EXACT RECURSION
   sigma_rr^+(u,v) = sigma_rr^-(u^+,v^+), marched inward from the outermost
   turn, rather than as the Taylor-expanded transport PDE (old Eq. 4).  This
   removes the narrow-band instability of section 9.2 (no finite differencing
   of a first-order PDE across a few grid points) and enforces
   sigma_rr^+ = 0 at the last turn geometrically.  sigma_rr^+ is carried as an
   extra unknown field with the recursion as its equation, so the whole coupled
   problem is ONE square sparse solve -- no fixed-point iteration, which
   matters because the friction feedback here turns out to be strong enough
   that simple Picard iteration would diverge.

5. F(x) is computed TWO independent ways and compared, as a physics check:
       (a) end reaction:   F = W t [<sig_uu(L)> sin(phi) + <sig_uv(L)> cos(phi)]
       (b) energy balance: F = dU_elastic/dx + D(x),
           D(x) = mu * (slip per unit x) * integral |sigma_rr^-| dA
   Route (b) is the virtual-work statement F dx = dU + dW_dissipated.  If the
   two disagree badly, the quasi-static stress state is not consistent, and
   that is reported rather than hidden.

   NOTE a dimensional bug in the write-up: eq:end_load reads
   F = W[sig_uu sin phi + sig_uv cos phi], which is a force per unit length --
   it is missing the thickness t.  Corrected to W*t*[...] here.  It never
   mattered before because the old solver's overall scale was arbitrary; with
   the energy closure the scale is physical, so it does.

BLOCKER: THE FRICTION AMPLIFICATION EXPONENT
--------------------------------------------
The friction feedback loop (sigma_uu -> Delta_sigma_rr -> friction traction ->
d sigma_uu/du) amplifies stress along u at rate

    lambda_+ = mu sin(2 phi) kappa_uu  =>  lambda_+ L ~ 0.77 mu theta_L,

so the exponent is set by mu times the NUMBER OF TURNS, and is independent of
E, t and W.  For the report's own parameters (theta_L = 200 rad, ~32 turns,
mu = 0.3) this peaks at lambda_+ L = 46.2 near x/L = 0.58, i.e. a growing-to-
decaying mode ratio of e^46 ~ 1.1e20.  Double precision carries ~1e16, so the
two-point boundary value problem in u is genuinely unsolvable there: measured
directly, F(x) at x/L = 0.6 came out 14.5, 41.1, -80.2, -26.3 N on successively
refined grids -- sign-flipping noise, not a converging sequence.  This is very
likely the same underlying pathology behind the old solver's "astronomical
range" and wildly non-monotonic F(x) (SOLVER_ARCHITECTURE.md sections 9.3, 9.5);
it is a conditioning wall, not a discretization bug, and no amount of grid
refinement or solver iteration touches it.

Two honest consequences:
  * The default configuration here uses theta_L = 80 rad (~12.7 turns), giving
    max lambda_+ L ~ 18, which IS solvable and converges.  `lambda_plus_L()`
    reports the exponent and `solve_fields` warns past a threshold.
  * Physically, lambda_+ L >> 1 means the stress lives in boundary layers of
    thickness 1/lambda_+ at each end with a numerically dead interior.  The
    correct treatment in that regime is a one-sided boundary-layer problem near
    u=L, not a two-point BVP across the whole ribbon.  That reformulation is
    NOT implemented here.

ALSO NOT CLOSED: the two F(x) routes do not agree (typically 5-15x apart in the
mid range).  The reason is structural rather than a bug: route (b) needs
dU_elastic/dx, but the friction-amplified membrane energy is held in place by a
dissipative, history-dependent contact law, so it is not a recoverable
potential, and a rate-independent quasi-static model gives no way to split
stored-but-recoverable from stored-but-friction-locked strain energy.  The same
issue killed the Xi feedback (see run_sweep).  Route (a), the end reaction, is
the one reported as F(x).

UNITS: cm, N, N/cm^2.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Physical parameters (paper ribbon; matches numerical_solver.py where shared)
# ---------------------------------------------------------------------------
L       = 200.0     # ribbon length                        [cm]
W       = 8.0       # ribbon width                         [cm]
t       = 0.01      # ribbon thickness                     [cm]
r_nat   = 2.0       # natural / x=0 coil radius            [cm]
                    # r_nat, W and mu are pinned between TWO competing
                    # constraints (see BLOCKER below):
                    #   self-contact:  shift_v_max = pi*r_nat  must be well
                    #     under W, else neighbouring turns separate axially
                    #     near phi=45 deg and the friction vanishes outright
                    #     (measured: at W/(pi r_nat)=1.27 the contact fraction
                    #     collapses to 0.22 mid-sweep and F(x) develops a real
                    #     ~4% non-monotonic dip there; at 2.04 it only dips to
                    #     0.52 and F stays monotonic).  Want W/(pi r_nat) >~ 2.
                    #   conditioning:  0.77*mu*L/r_nat <~ 30.
                    # r_nat=2, W=8, mu=0.3, L=200 => 15.9 turns,
                    # W/(pi r_nat)=1.27, max lambda_+L ~ 23.
                    # A finer winding (r_nat=1.25, mu=0.2, 25.5 turns) gives a
                    # better contact margin but was found to trip a spurious,
                    # GRID-DEPENDENT near-null mode of the coupled operator
                    # (|a|max/elongation spiking to ~5800 at isolated x, with
                    # F(x) then sign-flipping); at ppt=4 the same sweep is
                    # smooth, so it is a discretization artifact of the
                    # recursion block, not physics.  HEALTH_MAX below flags it.
theta_L = L / r_nat # total wind angle (fixed by topology) [rad]
E       = 3.0e5     # Young's modulus (paper, ~3 GPa)      [N/cm^2]
nu      = 0.3       # Poisson ratio
mu      = 0.3       # friction coefficient (paper-on-paper, 0.2-0.5)

D_plate = E * t**3 / (12.0 * (1.0 - nu**2))       # bending stiffness [N cm]
GAMMA   = E * t * r_nat**2 / D_plate              # = 12(1-nu^2)(r_nat/t)^2
RHO_JAM = t * theta_L**2 / (4.0 * np.pi * L)      # geometric jamming floor

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
Nu, Nv = 101, 25
u_grid = np.linspace(0.0, L, Nu)
v_grid = np.linspace(-W / 2.0, W / 2.0, Nv)
du = u_grid[1] - u_grid[0]
dv = v_grid[1] - v_grid[0]
UU, VV = np.meshgrid(u_grid, v_grid, indexing="ij")
NN = Nu * Nv

# unknown blocks:  a (u-displacement) | b (v-displacement) | sp (sigma_rr^+)
def IA(i, j): return i * Nv + j
def IB(i, j): return NN + i * Nv + j
def IP(i, j): return 2 * NN + i * Nv + j
NTOT = 3 * NN

c1 = E / (1.0 - nu**2)
c2 = E / (2.0 * (1.0 + nu))


# ===========================================================================
# 1.  Energy minimization  ->  rho(x)
# ===========================================================================
def u_bend_density(rho, chi):
    """Bending energy per unit mid-surface area, eq:bending_energy_density,
    evaluated at independent rho and phi (sin phi = chi/lambda)."""
    lam = np.sqrt(rho**2 + chi**2)
    sin2phi = (chi / lam) ** 2
    return 0.5 * D_plate * (
        (1.0 / (rho * r_nat) - 1.0 / r_nat) ** 2
        + 2.0 * (1.0 - nu) * sin2phi / (rho * r_nat * r_nat)
    )


def U_total_nondim(rho, chi, Xi):
    """U / (L W D / 2 r_nat^2).  Xi >= 1 is the membrane non-uniformity factor
    <eps^2>/<eps>^2, fed back from the solved field (1 for uniform strain)."""
    lam = np.sqrt(rho**2 + chi**2)
    return ((1.0 / rho - 1.0) ** 2
            + 2.0 * (1.0 - nu) * chi**2 / (rho * lam**2)
            + GAMMA * Xi * (lam - 1.0) ** 2)


def minimize_rho(chi, Xi=1.0):
    """Golden-section / dense scan minimization of U over rho, floored at the
    jamming radius.  Returns (rho, lambda, eps_mean, jammed_flag)."""
    lo, hi = max(RHO_JAM, 1e-4), 1.2
    grid = np.linspace(lo, hi, 4001)
    vals = U_total_nondim(grid, chi, Xi)
    k = int(np.argmin(vals))
    # local refinement
    a_, b_ = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    for _ in range(80):
        m1 = a_ + 0.382 * (b_ - a_)
        m2 = a_ + 0.618 * (b_ - a_)
        if U_total_nondim(m1, chi, Xi) < U_total_nondim(m2, chi, Xi):
            b_ = m2
        else:
            a_ = m1
    rho = 0.5 * (a_ + b_)
    jammed = rho <= RHO_JAM * (1.0 + 1e-6)
    rho = max(rho, RHO_JAM)
    lam = np.sqrt(rho**2 + chi**2)
    return rho, lam, lam - 1.0, jammed


# ===========================================================================
# 2.  Grid management and vectorized sparse operators
# ===========================================================================
# The binding numerical requirement is RESOLUTION PER COIL TURN.  The layer
# contact partner sits a distance shift_u = 2 pi r cos(phi) away in u, and the
# recursion is meaningless unless several grid points fit inside that distance.
# Measured convergence behaviour (see notes at the bottom of this file):
# F(x) is converged to a few percent at >~10 points/turn and is not converged
# at all below ~3.  Since shift_u shrinks as the coil tightens, the demand grows
# with x -- realistic windings (theta_L = 200 rad, ~32 turns) need Nu of several
# thousand near full extension.  Assembly is therefore fully vectorized as
# sparse matrix algebra rather than per-node Python loops.

PTS_PER_TURN = 12          # target grid points per coil turn
NU_MIN, NU_MAX = 201, 7000
NV = 13                    # width resolution (cheap; L/W ~ 40 so u dominates)


class Grid:
    def __init__(self, Nu, Nv):
        self.Nu, self.Nv = Nu, Nv
        self.u = np.linspace(0.0, L, Nu)
        self.v = np.linspace(-W / 2.0, W / 2.0, Nv)
        self.du = self.u[1] - self.u[0]
        self.dv = self.v[1] - self.v[0]
        self.n = Nu * Nv
        self.N = 3 * self.n
        ii, jj = np.meshgrid(np.arange(Nu), np.arange(Nv), indexing="ij")
        self.ii, self.jj = ii.ravel(), jj.ravel()
        self.UU, self.VV = np.meshgrid(self.u, self.v, indexing="ij")

    def nid(self, i, j):
        return i * self.Nv + j


def _first_deriv(g, axis):
    """Central inside, one-sided at the two ends, as a sparse n x n operator."""
    Nu, Nv, n = g.Nu, g.Nv, g.n
    h = g.du if axis == 0 else g.dv
    N = Nu if axis == 0 else Nv
    idx = g.ii if axis == 0 else g.jj
    rows, cols, vals = [], [], []
    nod = np.arange(n)

    def shift(k):
        return nod + k * (Nv if axis == 0 else 1)

    inner = (idx > 0) & (idx < N - 1)
    rows += [nod[inner], nod[inner]]
    cols += [shift(+1)[inner], shift(-1)[inner]]
    vals += [np.full(inner.sum(), 0.5 / h), np.full(inner.sum(), -0.5 / h)]

    lo = idx == 0
    rows += [nod[lo], nod[lo]]
    cols += [shift(+1)[lo], nod[lo]]
    vals += [np.full(lo.sum(), 1.0 / h), np.full(lo.sum(), -1.0 / h)]

    hi = idx == N - 1
    rows += [nod[hi], nod[hi]]
    cols += [nod[hi], shift(-1)[hi]]
    vals += [np.full(hi.sum(), 1.0 / h), np.full(hi.sum(), -1.0 / h)]

    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def _second_deriv(g, axis):
    Nu, Nv, n = g.Nu, g.Nv, g.n
    h = g.du if axis == 0 else g.dv
    N = Nu if axis == 0 else Nv
    idx = g.ii if axis == 0 else g.jj
    st = Nv if axis == 0 else 1
    nod = np.arange(n)
    rows, cols, vals = [], [], []

    inner = (idx > 0) & (idx < N - 1)
    for k, c in ((+1, 1.0), (0, -2.0), (-1, 1.0)):
        rows.append(nod[inner]); cols.append(nod[inner] + k * st)
        vals.append(np.full(inner.sum(), c / h**2))
    # one-sided 3-point at the ends
    lo = idx == 0
    for k, c in ((0, 1.0), (1, -2.0), (2, 1.0)):
        rows.append(nod[lo]); cols.append(nod[lo] + k * st)
        vals.append(np.full(lo.sum(), c / h**2))
    hi = idx == N - 1
    for k, c in ((0, 1.0), (-1, -2.0), (-2, 1.0)):
        rows.append(nod[hi]); cols.append(nod[hi] + k * st)
        vals.append(np.full(hi.sum(), c / h**2))

    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def _interp_matrix(g, u_t, v_t, active):
    """Sparse n x n bilinear-interpolation lookup matrix; inactive rows empty."""
    Nu, Nv, n = g.Nu, g.Nv, g.n
    fi = np.clip((u_t - g.u[0]) / g.du, 0.0, Nu - 1.0)
    fj = np.clip((v_t - g.v[0]) / g.dv, 0.0, Nv - 1.0)
    i0 = np.minimum(np.floor(fi).astype(int), Nu - 2)
    j0 = np.minimum(np.floor(fj).astype(int), Nv - 2)
    wi, wj = fi - i0, fj - j0
    nod = np.arange(n)
    rows, cols, vals = [], [], []
    for di, dj, wt in ((0, 0, (1 - wi) * (1 - wj)), (1, 0, wi * (1 - wj)),
                       (0, 1, (1 - wi) * wj), (1, 1, wi * wj)):
        rows.append(nod[active])
        cols.append(((i0 + di) * Nv + (j0 + dj))[active])
        vals.append(wt[active])
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def _hstack3(g, Ba, Bb, Bp):
    z = sp.csr_matrix((g.n, g.n))
    return sp.hstack([Ba if Ba is not None else z,
                      Bb if Bb is not None else z,
                      Bp if Bp is not None else z], format="csr")


# ===========================================================================
# 3.  One field solve at frozen (x, rho)
# ===========================================================================
def lambda_plus_L(chi, rho, lam):
    """Friction amplification exponent; > ~37 is beyond double precision."""
    phi = np.arctan2(chi, rho)
    return mu * np.sin(2 * phi) * (rho / lam) ** 2 / (rho * r_nat) * L


LAMBDA_WARN = 30.0
HEALTH_MAX = 500.0    # |a|max/elongation above this => near-null mode, reject
_warned = set()


def choose_Nu(rho, lam):
    shift_u = 2 * np.pi * (rho * r_nat) * (rho / lam)
    Nu = int(np.clip(round(PTS_PER_TURN * L / shift_u), NU_MIN, NU_MAX))
    return Nu + (Nu % 2 == 0), shift_u


def solve_fields(chi, rho, lam, Nu=None, Nv=NV, verbose=False):
    lpL = lambda_plus_L(chi, rho, lam)
    if lpL > LAMBDA_WARN and round(chi, 3) not in _warned:
        _warned.add(round(chi, 3))
        print(f"    !! WARNING x/L={chi:.3f}: lambda_+ L = {lpL:.1f} "
              f"(mode ratio e^{lpL:.0f}); past ~37 this is beyond double "
              f"precision and the result is not trustworthy")
    if Nu is None:
        Nu, _ = choose_Nu(rho, lam)
    g = Grid(Nu, Nv)
    n = g.n

    x = chi * L
    r = rho * r_nat
    sin_phi, cos_phi = chi / lam, rho / lam
    phi = np.arctan2(chi, rho)
    sin2phi, cos2phi = np.sin(2 * phi), np.cos(2 * phi)
    kap_uu = cos_phi**2 / r
    kap_uv = sin_phi * cos_phi / r
    kap_vv = sin_phi**2 / r
    shift_u = 2 * np.pi * r * cos_phi
    shift_v = 2 * np.pi * r * sin_phi

    Du, Dv = _first_deriv(g, 0), _first_deriv(g, 1)
    Duu, Dvv = _second_deriv(g, 0), _second_deriv(g, 1)
    Duv = (Du @ Dv).tocsr()

    # in-plane stresses as operators on z = [a ; b ; sp]
    S_uu = _hstack3(g, c1 * Du, c1 * nu * Dv, None)
    S_vv = _hstack3(g, c1 * nu * Du, c1 * Dv, None)
    S_uv = _hstack3(g, c2 * Dv, c2 * Du, None)
    P_sel = _hstack3(g, None, None, sp.identity(n, format="csr"))

    # Delta_sigma_rr  (eq:outofplane_exact, leading order)
    M_D = t * (kap_uu * S_uu - 2 * kap_uv * S_uv + kap_vv * S_vv)

    # ---- contact gates -------------------------------------------------
    Uf, Vf = g.UU.ravel(), g.VV.ravel()
    has_outer = (Vf >= -W / 2 + shift_v - 1e-12) & (Uf + shift_u <= L + 1e-12)
    has_inner = (Vf <= W / 2 - shift_v + 1e-12) & (Uf - shift_u >= -1e-12)
    Hi = sp.diags(has_inner.astype(float))

    # sigma_rr^- = has_inner * (sp - Delta);  S_rr = sp + sigma_rr^-
    M_S = (sp.diags(1.0 + has_inner.astype(float)) @ P_sel) - Hi @ M_D

    F_u = mu * sin2phi * (M_D / t + M_S / (4.0 * r))
    F_v = mu * cos2phi * (M_D / t - M_S / (4.0 * r))

    # ---- Navier equilibrium --------------------------------------------
    cross = c1 * nu + c2
    EQ_u = _hstack3(g, c1 * Duu + c2 * Dvv, cross * Duv, None) - F_u
    EQ_v = _hstack3(g, cross * Duv, c2 * Duu + c1 * Dvv, None) - F_v

    # ---- row selection masks -------------------------------------------
    is_end = (g.ii == 0) | (g.ii == Nu - 1)
    is_vedge = (~is_end) & ((g.jj == 0) | (g.jj == Nv - 1))
    is_int = (~is_end) & (~is_vedge)
    Mint, Mv, Me = (sp.diags(m.astype(float)) for m in (is_int, is_vedge, is_end))

    Pa = _hstack3(g, sp.identity(n, format="csr"), None, None)
    Pb = _hstack3(g, None, sp.identity(n, format="csr"), None)

    # row block 1: interior -> eq_u ; v-edge -> sigma_vv = 0 ; ends -> a given
    R1 = Mint @ EQ_u + Mv @ S_vv + Me @ Pa
    b1 = np.zeros(n)
    elong = L * (lam - 1.0)
    b1[g.ii == Nu - 1] = elong

    # row block 2: interior -> eq_v ; v-edge -> sigma_uv = 0 ;
    #   u=0 -> b = 0 (fully clamped fixture) ; u=L -> sigma_uv = 0 (pulling grip
    #   transmits axial load only, a roller).  Clamping BOTH ends in v puts a
    #   clamped-meets-free corner at u=L, whose stress singularity diverges
    #   linearly with 1/du and contaminates F (read at u=L) even though every
    #   interior-of-width value is converged; leaving u=L shear-free moves the
    #   only singular corners to u=0, away from where F is measured.  Imposing
    #   sigma_uv = 0 at BOTH ends instead over-constrains the global z-balance
    #   (it forces W t sin(phi) * int sigma_uu to be equal at the two ends).
    is_u0 = g.ii == 0
    is_uL = g.ii == Nu - 1
    M0, ML = sp.diags(is_u0.astype(float)), sp.diags(is_uL.astype(float))
    R2 = Mint @ EQ_v + Mv @ S_uv + M0 @ Pb + ML @ S_uv
    b2 = np.zeros(n)

    # row block 3: exact layer-contact recursion for sigma_rr^+
    B = _interp_matrix(g, Uf + shift_u, Vf - shift_v, has_outer)
    R3 = P_sel - B @ (P_sel - Hi @ M_D)
    b3 = np.zeros(n)

    A = sp.vstack([R1, R2, R3], format="csc")
    rhs = np.concatenate([b1, b2, b3])
    z = spla.spsolve(A, rhs)
    if not np.all(np.isfinite(z)):
        raise RuntimeError("singular system")

    a = z[:n].reshape(Nu, Nv)
    b = z[n:2 * n].reshape(Nu, Nv)
    spl = z[2 * n:].reshape(Nu, Nv)

    sig_uu = (S_uu @ z).reshape(Nu, Nv)
    sig_vv = (S_vv @ z).reshape(Nu, Nv)
    sig_uv = (S_uv @ z).reshape(Nu, Nv)
    Delta = (M_D @ z).reshape(Nu, Nv)
    eps_uu = (Du @ z[:n]).reshape(Nu, Nv)
    eps_vv = (Dv @ z[n:2 * n]).reshape(Nu, Nv)
    eps_uv = 0.5 * ((Dv @ z[:n]) + (Du @ z[n:2 * n])).reshape(Nu, Nv)
    smin = np.where(has_inner.reshape(Nu, Nv), spl - Delta, 0.0)
    S_rr = spl + smin

    out = dict(a=a, b=b, sig_uu=sig_uu, sig_uv=sig_uv, sig_vv=sig_vv,
               eps_uu=eps_uu, eps_uv=eps_uv, eps_vv=eps_vv,
               sp=spl, sm=smin, S_rr=S_rr, Delta=Delta,
               has_inner=has_inner.reshape(Nu, Nv),
               has_outer=has_outer.reshape(Nu, Nv),
               phi=phi, r=r, x=x, chi=chi, rho=rho, lam=lam,
               sin_phi=sin_phi, cos_phi=cos_phi,
               Nu=Nu, Nv=Nv, pts_per_turn=shift_u / g.du, lam_plus_L=lpL,
               u=g.u, v=g.v)

    # health check: for a linear system whose only inhomogeneity is the
    # prescribed elongation, |a|max/elongation is a pure conditioning number
    # and should be O(1-100).  Spikes of 1e3+ mean the solve has been captured
    # by a near-null mode of the coupled operator and the result is garbage.
    out["health"] = float(np.abs(a).max() / max(abs(elong), 1e-300))
    out["healthy"] = out["health"] < HEALTH_MAX

    out["F_end"] = W * t * (sig_uu[-1, :].mean() * sin_phi
                            + sig_uv[-1, :].mean() * cos_phi)
    out["F_end_0"] = W * t * (sig_uu[0, :].mean() * sin_phi
                              + sig_uv[0, :].mean() * cos_phi)

    dens = 0.5 * (sig_uu * eps_uu + 2 * sig_uv * eps_uv + sig_vv * eps_vv)
    out["U_mem"] = t * np.trapezoid(np.trapezoid(dens, g.v, axis=1), g.u)
    out["U_bend"] = L * W * u_bend_density(rho, chi)
    out["U_mem_uniform"] = 0.5 * E * t * L * W * (lam - 1.0) ** 2
    eb = lam - 1.0
    out["Xi"] = out["U_mem"] / out["U_mem_uniform"] if eb > 1e-14 else 1.0
    out["contact_frac"] = float(out["has_inner"].mean())
    # Unilateral-contact violation: two surfaces can only push, so sigma_rr
    # must be <= 0.  The model has no complementarity condition, so it can and
    # does return tensile patches.  Reported, not silently clipped -- the same
    # limitation the write-up flags in the stick-shear-seed section.
    act = out["has_inner"]
    out["tensile_frac"] = (float((smin[act] > 0).mean()) if act.any() else 0.0)
    out["contact_integral"] = np.trapezoid(
        np.trapezoid(np.abs(smin) * out["has_inner"], g.v, axis=1), g.u)
    if verbose:
        print(f"    Nu={Nu} ppt={out['pts_per_turn']:.1f} rho={rho:.4f} "
              f"eps={eb:.3e} F(L)={out['F_end']:.5g}")
    return out


# ===========================================================================
# 4.  Sweep over x with the rho <-> field self-consistency loop
# ===========================================================================
def run_sweep(chis, feedback_Xi=False):
    """Sweep x.  Xi feedback is OFF by default, deliberately.

    Feeding the solved field's <eps^2>/<eps>^2 back into the rho-minimization
    as an effective membrane stiffening was tried and is WRONG: it comes out
    ~4e5 here, which drives the minimizer to the inextensible radius, collapses
    eps to ~1e-12, and makes F ~ 1e-6 N.  The reason it is wrong is physical,
    not numerical -- the friction-amplified internal stress stores energy, but
    that energy is pinned in place by a dissipative, history-dependent contact
    law, not by a potential.  It is therefore not available to a reversible
    energy minimization over the coil radius.  Only the bare (uniform-strain)
    membrane stiffness belongs in the rho balance; Xi is reported as a
    diagnostic instead of being used.
    """
    res = []
    for chi in chis:
        Xi = 1.0
        rho, lam, eps_bar, jammed = minimize_rho(chi, Xi)
        sol = solve_fields(chi, rho, lam)
        if feedback_Xi:
            for _ in range(3):
                Xi = 0.5 * Xi + 0.5 * max(1.0, min(sol["Xi"], 1e6))
                rho, lam, eps_bar, jammed = minimize_rho(chi, Xi)
                sol = solve_fields(chi, rho, lam)
        sol["jammed"] = jammed
        sol["Xi_used"] = Xi
        # global z-force balance diagnostic (see notes: the in-plane subsystem
        # alone does not conserve it -- the radial equation carries the rest)
        sol["F_end_0"] = W * t * (sol["sig_uu"][0, :].mean() * sol["sin_phi"]
                                  + sol["sig_uv"][0, :].mean() * sol["cos_phi"])
        res.append(sol)
        print(f"x/L={chi:.3f} rho={sol['rho']:.4f}{' [JAM]' if jammed else ''} "
              f"eps={sol['lam']-1:.2e} lam+L={sol['lam_plus_L']:5.1f} "
              f"Nu={sol['Nu']:5d} ppt={sol['pts_per_turn']:.0f} "
              f"cont={sol['contact_frac']:.2f} health={sol['health']:.0f}"
              f"{'' if sol['healthy'] else ' <<REJECT>>'} "
              f"tens={sol['tensile_frac']:.2f} F(L)={sol['F_end']:.5g} N")
    return res


def post_process(res):
    chis = np.array([s["chi"] for s in res])
    xs = chis * L
    rho = np.array([s["rho"] for s in res])
    lam = np.array([s["lam"] for s in res])
    phi = np.array([s["phi"] for s in res])
    r_of_x = rho * r_nat
    U_el = np.array([s["U_mem"] + s["U_bend"] for s in res])
    F_end = np.array([s["F_end"] for s in res])
    cint = np.array([s["contact_integral"] for s in res])

    # slip length per unit x:  2 pi sqrt((dr/dx)^2 + (dphi/dx)^2 r^2)
    drdx = np.gradient(r_of_x, xs)
    dphidx = np.gradient(phi, xs)
    slip_rate = 2 * np.pi * np.sqrt(drdx**2 + (dphidx * r_of_x) ** 2)

    F_diss = mu * slip_rate * cint
    F_elastic = np.gradient(U_el, xs)
    F_energy = F_elastic + F_diss
    return dict(xs=xs, chis=chis, rho=rho, lam=lam, phi=phi, r=r_of_x,
                F_end=F_end, F_diss=F_diss, F_elastic=F_elastic,
                F_energy=F_energy, U_el=U_el)


# ===========================================================================
# 5.  Plots and animations
# ===========================================================================
def plot_force(pp, fname="F_of_x_energy_coupled.png"):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(pp["xs"], pp["F_end"], "o-", label="end reaction $Wt[\\ldots]$")
    ax[0].plot(pp["xs"], pp["F_energy"], "s--", label="$dU/dx + \\mathcal{D}$")
    ax[0].set_xlabel("x [cm]"); ax[0].set_ylabel("F [N]"); ax[0].legend()
    ax[0].set_title("F(x), two routes")

    ax[1].semilogy(pp["xs"], np.abs(pp["F_end"]), "o-", label="|end reaction|")
    ax[1].semilogy(pp["xs"], np.abs(pp["F_energy"]), "s--", label="|energy|")
    ax[1].semilogy(pp["xs"], np.abs(pp["F_elastic"]), ":", label="|elastic|")
    ax[1].semilogy(pp["xs"], np.abs(pp["F_diss"]), ":", label="|dissipation|")
    ax[1].set_xlabel("x [cm]"); ax[1].set_ylabel("F [N]"); ax[1].legend(fontsize=8)
    ax[1].set_title("semilog: straight line = exponential")

    ax[2].plot(pp["xs"], pp["rho"], "o-", label=r"$\rho = r/r_{nat}$")
    ax[2].axhline(RHO_JAM, color="r", ls="--", label=r"$\rho_{jam}$")
    ax[2].plot(pp["xs"], np.sqrt(np.maximum(1 - pp["chis"] ** 2, 0)), "k:",
               label="inextensible $\\cos\\phi$")
    ax[2].set_xlabel("x [cm]"); ax[2].legend(); ax[2].set_title("energy-minimizing radius")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    print(f"wrote {fname}")


NU_PLOT = 240


def _resample(s, key):
    """Adaptive grids differ per x, so put every frame on one plotting grid."""
    f = s[key].astype(float)
    up = np.linspace(0.0, L, NU_PLOT)
    out = np.empty((NU_PLOT, s["Nv"]))
    for j in range(s["Nv"]):
        out[:, j] = np.interp(up, s["u"], f[:, j])
    return out


def animate(res, key, fname, title):
    frames = np.array([_resample(s, key) for s in res])
    vmax = np.percentile(np.abs(frames), 99.5)
    vmax = vmax if vmax > 0 else 1.0
    norm = matplotlib.colors.SymLogNorm(linthresh=max(vmax * 1e-4, 1e-30),
                                        vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(frames[0].T, origin="lower", aspect="auto", cmap="RdBu_r",
                   norm=norm, extent=[0, L, -W / 2, W / 2])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("u [cm]"); ax.set_ylabel("v [cm]")
    ttl = ax.set_title("")

    def upd(k):
        im.set_data(frames[k].T)
        ttl.set_text(f"{title}   x/L={res[k]['chi']:.3f}")
        return im, ttl

    anim = animation.FuncAnimation(fig, upd, frames=len(frames), blit=False)
    anim.save(fname, writer=animation.PillowWriter(fps=6))
    plt.close(fig)
    print(f"wrote {fname}")


def animate_gate(res, fname="gate_energy_coupled.gif"):
    frames = []
    for s in res:
        g = np.where(s["has_inner"] & s["has_outer"], 0,
             np.where(s["has_outer"], 1, np.where(s["has_inner"], 2, 3)))
        frames.append(_resample({**s, "g": g}, "g"))
    frames = np.array(frames)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(frames[0].T, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=3, extent=[0, L, -W / 2, W / 2])
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3],
                 label="0 both / 1 outer / 2 inner / 3 none")
    ax.set_xlabel("u [cm]"); ax.set_ylabel("v [cm]")
    ttl = ax.set_title("")

    def upd(k):
        im.set_data(frames[k].T); ttl.set_text(f"contact gate  x/L={res[k]['chi']:.3f}")
        return im, ttl

    anim = animation.FuncAnimation(fig, upd, frames=len(frames), blit=False)
    anim.save(fname, writer=animation.PillowWriter(fps=6)); plt.close(fig)
    print(f"wrote {fname}")


# ===========================================================================
if __name__ == "__main__":
    print(f"theta_L = {theta_L:.0f} rad ({theta_L/2/np.pi:.1f} turns)   "
          f"Gamma = {GAMMA:.4g}   rho_jam = {RHO_JAM:.4g}   "
          f"rho_* ~ {(2/GAMMA)**(1/6):.4g}")
    print(f"D = {D_plate:.4g} N cm   EtW = {E*t*W:.4g} N   "
          f"target {PTS_PER_TURN} pts/turn, Nv={NV}")
    lpmax = max(lambda_plus_L(c, *minimize_rho(c)[:2])
                for c in np.linspace(0.05, 0.99, 60))
    print(f"max lambda_+ L over sweep = {lpmax:.1f}  "
          f"({'OK' if lpmax < LAMBDA_WARN else 'BEYOND DOUBLE PRECISION'})\n")

    chis = np.linspace(0.10, 0.985, 28)
    res = run_sweep(chis)
    pp = post_process(res)

    print("\n--- F(x) summary ---")
    for k in range(len(pp["xs"])):
        print(f"x/L={pp['chis'][k]:.3f}  F_end={pp['F_end'][k]:.5g}  "
              f"F_energy={pp['F_energy'][k]:.5g}  "
              f"(elastic {pp['F_elastic'][k]:.4g} + diss {pp['F_diss'][k]:.4g})")
    ok = np.array([s["healthy"] for s in res])
    print(f"\nhealthy points: {ok.sum()}/{len(ok)}")
    Fok = pp["F_end"][ok]
    d = np.diff(Fok)
    print(f"F_end monotonic increasing (healthy pts only): {np.all(d > 0)}  "
          f"(min step {d.min():.4g})")
    lg = np.log(np.abs(Fok)); xo = pp["xs"][ok]
    rate = np.gradient(lg, xo)
    print(f"local exponential rate d(lnF)/dx: {rate.min():.4g} .. {rate.max():.4g} /cm"
          f"  (increasing => faster than exponential)")
    print(f"F_end range: {pp['F_end'][0]:.4g} -> {pp['F_end'][-1]:.4g} N "
          f"({pp['F_end'][-1]/max(pp['F_end'][0],1e-30):.4g}x)")

    plot_force(pp)
    animate(res, "sig_uu", "sig_uu_energy_coupled.gif", r"$\bar\sigma_{uu}$ [N/cm$^2$]")
    animate(res, "sm", "sigma_rr_minus_energy_coupled.gif", r"$\sigma_{rr}^-$ [N/cm$^2$]")
    animate(res, "S_rr", "S_rr_energy_coupled.gif", r"$S_{rr}$ [N/cm$^2$]")
    animate_gate(res)
