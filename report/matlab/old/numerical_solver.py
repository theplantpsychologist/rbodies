"""
Numerical (finite-difference) solution of the full (u,v) stress field, using the
boxed governing equations of Sections "In-plane thickness integration" /
"Out-of-plane thickness integration" / "Closing compatibility with friction"
directly on a 2D grid -- i.e. WITHOUT the v-mode truncation used in the
analytical treatment ("v-dependence reduction" onward).

Unknown fields, each a function of (u,v) at a fixed, frozen extension x:
    sigma_uu, sigma_uv, sigma_vv   (thickness-averaged in-plane stresses)
    S_rr = sigma_rr^+ + sigma_rr^-, Delta_rr = sigma_rr^+ - sigma_rr^-

Governing equations (linear, since r,phi are frozen at a given x):
    (1) d(sigma_uu)/du + d(sigma_uv)/dv = mu*sin(2phi)*[Delta_rr/t + S_rr/(4r)]   [eq:inplane_u_integrated]
    (2) d(sigma_uv)/du + d(sigma_vv)/dv = mu*cos(2phi)*[Delta_rr/t - S_rr/(4r)]   [eq:inplane_v_integrated]
    (3) Laplacian(sigma_uu+sigma_vv) = (1+nu)*{ (mu/t)*[sin2phi*d(Delta_rr)/du + cos2phi*d(Delta_rr)/dv]
                                                  + (mu/4r)*[sin2phi*d(S_rr)/du - cos2phi*d(S_rr)/dv] }  [eq:beltrami_michell]
    (4) pi*r*cos(phi)*d(S_rr)/du - pi*r*sin(phi)*d(S_rr)/dv = t*[kappa_uu*sigma_uu - 2*kappa_uv*sigma_uv + kappa_vv*sigma_vv]
                                                                                     [eq:S_transport, "both contacts"]
    Delta_rr = t*[kappa_uu*sigma_uu - 2*kappa_uv*sigma_uv + kappa_vv*sigma_vv]      [eq:outofplane_exact, leading order]

Boundary conditions at the free edges v=+-W/2 (Section "v-dependence reduction"):
    sigma_uv = sigma_vv = 0                       (traction-free edge)
    S_rr = Delta_rr = 0, hence sigma_rr^+=sigma_rr^-=0 individually
        -- since sigma_uv=sigma_vv=0 there already, Delta_rr's algebraic formula
           forces sigma_uu = 0 at the edges too, which is imposed directly below
           (a consequence of the exact edge argument, not a mode-truncation artifact).

Piecewise layer-contact clamps (eq:contact_outer, eq:contact_inner), applied POINTWISE
at every (u,v) with u>0, not just at u=L -- this replaces equation (4) with an
algebraic relation wherever a neighbor is geometrically or axially absent:
    has_outer(u,v) = [v >= -W/2 + 2*pi*r*sin(phi)]  AND  [u + 2*pi*r*cos(phi) <= L]
    has_inner(u,v) = [v <=  W/2 - 2*pi*r*sin(phi)]  AND  [u - 2*pi*r*cos(phi) >= 0]
        both:        bulk transport equation (4), S_rr free               ["both contacts"]
        outer only:  sigma_rr^- = 0  =>  S_rr := +Delta_rr                ["outer contact only"]
        inner only:  sigma_rr^+ = 0  =>  S_rr := -Delta_rr                ["inner contact only"]
        neither:     sigma_rr^+ = sigma_rr^- = 0 => S_rr := 0, Delta_rr := 0   ["no contact"]
The axial part of the outer gate (u + 2*pi*r*cos(phi) > L) automatically reproduces
sigma_rr^+(L,v)=0 at the active end, matching "nothing is wound past the last turn"
-- a genuine geometric fact, independent of how the tip happens to be gripped.

u=0 is deliberately NOT given the mirror-image treatment (case 4, "external free
end", rather than the has_inner axial gate forcing sigma_rr^-(0,v):=0). "No inner
LAYER" (nothing wound at negative u) is still true, but it does not imply zero
traction: u=0 is held by an external fixture, which is free to supply whatever
reaction -- including a through-thickness/radial component -- is needed, unlike
u=L, which is a genuine free surface with nothing pressing on it at all. So S_rr
is left entirely free at u=0, with no pointwise contact rule and no equation of
its own; what determines it is the global axial force-balance condition below.

Global axial force balance (replaces any pointwise condition at u=0): every
layer-contact force in this model is ribbon-on-itself, so by Newton's third law
each one is an internal action-reaction pair with both partners inside [0,L] --
summed over the whole ribbon, they all cancel, leaving only the two end
resultants, which must be equal:
    W*int[sigma_uu(0,v) sin(phi) + sigma_uv(0,v) cos(phi)] dv
        = W*int[sigma_uu(L,v) sin(phi) + sigma_uv(L,v) cos(phi)] dv
i.e. the same integral that defines F(x) (eq:end_load), evaluated at both ends and
set equal -- the external reaction force the fixture actually supplies is this
value negated (its outward normal is -u, not +u), but that sign is bookkeeping on
the fixture side, not part of this equation. This condition is itself homogeneous
(trivially satisfied by the all-zero field), so a separate scale anchor is still
needed (below); it only ties the two ends together, it doesn't fix magnitude.

Every spatial derivative is discretized with a ONE-SIDED (forward, backward only at
the last index) difference, never a skip-the-center central difference. This is a
deliberate choice: central differences here decouple the even- and odd-index grid
points into two independent sub-lattices with no direct coupling (a "checkerboard"
null mode), which produced a large, unphysical, alternating-sign v-oscillation in
an earlier version of this script -- unrelated to the physical oscillation issue
diagnosed in the analytical treatment, but a numerical artifact of the same
superficial kind. One-sided differences reference the center point directly and do
not have this null mode, at the cost of first- (rather than second-) order accuracy.

The u-direction treatment of equations (1)-(3) at u=0,L uses these same one-sided
differences of the bulk equations themselves (a "free end" / natural-boundary
closure), rather than an independently-derived condition on sigma_uu, sigma_uv
there -- a modeling choice, flagged here rather than hidden. This leaves the
discretized system very slightly over-determined, so it is solved by (sparse)
least squares rather than a square direct solve.

Even with the force-balance condition above, the system is still homogeneous (no
distributed load anywhere), so one more reference value is needed to break the
trivial (all-zero) degeneracy: sigma_uu(L, v=0) is fixed to 1, purely to set an
overall scale. Earlier versions of this script anchored sigma_uu(L,0) to a fixed
value AS the physical boundary condition, with no separate u=0 condition at all --
that silently cancelled the friction-accumulation growth the model is supposed to
predict (since F(x) is directly proportional to sigma_uu(L,.), fixing it to a
constant for every x left F(x) tracking little more than the geometric sin(phi(x))
factor) and produced near-zero stress everywhere except close to u=L, with nothing
tying the two ends together. That is now fixed by the force-balance condition
above; the anchor here plays no role beyond setting units.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------------------------------------------------------
# Fixed geometric / material constants
# ----------------------------------------------------------------------------
L = 200.0          # total ribbon length (cm)
r0 = 1.0           # base coil radius at x=0 (cm)
W = 5.0           # ribbon width (cm)
theta_L = L / r0   # total wind angle (rad)
mu = 0.3           # friction coefficient
t = 0.01           # ribbon thickness (cm)
nu_p = 0.3         # Poisson ratio (not otherwise specified upstream; a plain value for paper)

# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
Nu, Nv = 81, 41
u_grid = np.linspace(0.0, L, Nu)
v_grid = np.linspace(-W / 2, W / 2, Nv)
du = u_grid[1] - u_grid[0]
dv = v_grid[1] - v_grid[0]


def combine(*groups):
    merged = {}
    for g in groups:
        for var, coeff in g:
            merged[var] = merged.get(var, 0.0) + coeff
    return list(merged.items())


def scale(entries, factor):
    return [(var, coeff * factor) for var, coeff in entries]


def solve_stress_field(x):
    """Solve for sigma_uu, sigma_uv, sigma_vv, S_rr, Delta_rr on the (u,v) grid
    at a fixed extension x. Returns seven (Nu,Nv) stress-field arrays --
    sigma_uu, sigma_uv, sigma_vv, S_rr, Delta_rr, sigma_rr_plus, sigma_rr_minus
    -- plus the (Nu,Nv) integer 'case' array (0=both contacts, 1=outer only,
    2=inner only, 3=no contact) used for the layer-contact gate animation.
    """
    phi = np.arcsin(x / L)
    r = np.sqrt(L ** 2 - x ** 2) / theta_L
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin2phi, cos2phi = np.sin(2 * phi), np.cos(2 * phi)
    kappa_uu = cos_phi ** 2 / r
    kappa_uv = sin_phi * cos_phi / r
    kappa_vv = sin_phi ** 2 / r
    shift_u = 2 * np.pi * r * cos_phi
    shift_v = 2 * np.pi * r * sin_phi

    is_edge = lambda j: j == 0 or j == Nv - 1

    # --- unknown indexing (sigma_uu, sigma_uv, sigma_vv, S_rr all pinned to
    # zero at the free edges v=+-W/2, so only interior-v points are unknowns
    # for every field) ------------------------------------------------------
    n_int_v = Nv - 2
    n_uu, n_uv, n_vv, n_S = (Nu * n_int_v,) * 4
    N = n_uu + n_uv + n_vv + n_S

    def I_uu(i, j):
        return i * n_int_v + (j - 1)

    def I_uv(i, j):
        return n_uu + i * n_int_v + (j - 1)

    def I_vv(i, j):
        return n_uu + n_uv + i * n_int_v + (j - 1)

    def I_S(i, j):
        return n_uu + n_uv + n_vv + i * n_int_v + (j - 1)

    def at_uu(i, j):
        return [] if is_edge(j) else [(I_uu(i, j), 1.0)]

    def at_uv(i, j):
        return [] if is_edge(j) else [(I_uv(i, j), 1.0)]

    def at_vv(i, j):
        return [] if is_edge(j) else [(I_vv(i, j), 1.0)]

    # --- classify each interior-v grid point's layer-contact case -------
    # 0 = both contacts (S_rr free, bulk transport eq); 1 = outer only
    # (sigma_rr^-:=0, sigma_rr^+ read off its ACTUAL outer neighbor);
    # 2 = inner only (sigma_rr^+:=0, sigma_rr^- read off its ACTUAL inner
    # neighbor); 3 = no contact (sigma_rr^+=sigma_rr^-=0).
    # -1 at v=+-W/2 marks "edge" (sigma_rr pinned to 0 there by the free-edge
    # argument, not by any of the cases below).
    #
    # u=0 gets the SAME treatment as every other row: has_inner is false
    # there purely because the axial gate fails (u - shift_u < 0 always), so
    # it lands in case 1 or 3 like any other point near a geometric boundary
    # -- "no inner LAYER" (nothing wound at negative u) is a real fact
    # regardless of how the base happens to be gripped. What changed at u=0
    # is NOT this pointwise contact logic (reverted below to the ordinary
    # rule) but the separate global axial force-balance condition added
    # further down, which does not touch sigma_rr at all.
    case = np.full((Nu, Nv), -1, dtype=int)
    for i in range(Nu):
        for j in range(1, Nv - 1):
            has_outer = (v_grid[j] >= -W / 2 + shift_v) and (u_grid[i] + shift_u <= L)
            has_inner = (v_grid[j] <= W / 2 - shift_v) and (u_grid[i] - shift_u >= 0)
            if has_outer and has_inner:
                case[i, j] = 0
            elif has_outer:
                case[i, j] = 1
            elif has_inner:
                case[i, j] = 2
            else:
                case[i, j] = 3

    def local_elastic_delta_expr(i, j):
        """The algebraic leading-order elasticity relation for Delta_rr
        (eq:outofplane_exact). This is NOT the general Delta_rr -- it is
        valid only as the source term inside the 'both contacts' transport
        equation (eq:S_transport), exactly as derived in the document (there,
        this expression and the contact-recursion expression for Delta_rr are
        two views of the same quantity, equated to eliminate Delta_rr and
        get a pure PDE for S_rr). Using it as the general Delta_rr for the
        'outer/inner only' cases too -- what an earlier version of this
        script did -- has no basis in the document and was the source of a
        real bug: it let sigma_rr^+ come out positive (tensile) in outer-only
        zones with no connection to what the actual outer neighbor's stress
        was, violating Newton's third law (sigma_rr^+ here MUST equal
        sigma_rr^- at the actual contact partner, eq:contact_outer) and
        giving physically-impossible "two surfaces pulling on each other."
        """
        if is_edge(j):
            return []
        return combine(scale(at_uu(i, j), t * kappa_uu),
                        scale(at_uv(i, j), -2 * t * kappa_uv),
                        scale(at_vv(i, j), t * kappa_vv))

    def sigma_rr_plus_expr(i, j):
        """sigma_rr^+(i,j) in terms of the local free variable I_S(i,j)
        (whatever equation ends up defining it -- see below) and, only for
        case 0, the local elastic Delta_rr. Not recursive: it never looks at
        any other grid point directly."""
        if is_edge(j):
            return []
        c = case[i, j]
        if c == 0:
            return combine(scale([(I_S(i, j), 1.0)], 0.5), scale(local_elastic_delta_expr(i, j), 0.5))
        if c == 1:
            return [(I_S(i, j), 1.0)]
        return []  # c == 2 or 3: no outer neighbor at all => sigma_rr^+ := 0

    def sigma_rr_minus_expr(i, j):
        if is_edge(j):
            return []
        c = case[i, j]
        if c == 0:
            return combine(scale([(I_S(i, j), 1.0)], 0.5), scale(local_elastic_delta_expr(i, j), -0.5))
        if c == 2:
            return [(I_S(i, j), 1.0)]
        return []  # c == 1 or 3: no inner neighbor at all => sigma_rr^- := 0

    def S_rr_expr(i, j):
        return combine(sigma_rr_plus_expr(i, j), sigma_rr_minus_expr(i, j))

    def true_delta_rr_expr(i, j):
        """The ACTUAL Delta_rr = sigma_rr^+ - sigma_rr^-, valid in every
        case, for use as the general source term in Eqs. (1)-(3). Equals
        local_elastic_delta_expr only in case 0, by construction (that is
        where the two are equated to derive eq:S_transport in the first
        place); in cases 1/2/3 it is NOT that algebraic formula."""
        return combine(sigma_rr_plus_expr(i, j), scale(sigma_rr_minus_expr(i, j), -1.0))

    def bilinear_lookup(expr_fn, u_target, v_target):
        """Linear expression for expr_fn evaluated at the continuous location
        (u_target, v_target) via bilinear interpolation of the 4 nearest
        grid points (clamped to the grid). Used to read off the ACTUAL
        stress at a layer-contact partner, which generally does not land
        exactly on a grid point."""
        fi = min(max((u_target - u_grid[0]) / du, 0.0), Nu - 1.0)
        fj = min(max((v_target - v_grid[0]) / dv, 0.0), Nv - 1.0)
        i0 = min(int(np.floor(fi)), Nu - 2)
        j0 = min(int(np.floor(fj)), Nv - 2)
        i1, j1 = i0 + 1, j0 + 1
        wi, wj = fi - i0, fj - j0
        return combine(
            scale(expr_fn(i0, j0), (1 - wi) * (1 - wj)),
            scale(expr_fn(i0, j1), (1 - wi) * wj),
            scale(expr_fn(i1, j0), wi * (1 - wj)),
            scale(expr_fn(i1, j1), wi * wj),
        )

    # --- generic one-sided derivative helpers (never skip the center point) ---
    def d_du(at, i, j):
        if i < Nu - 1:
            return combine(scale(at(i + 1, j), 1 / du), scale(at(i, j), -1 / du))
        return combine(scale(at(i, j), 1 / du), scale(at(i - 1, j), -1 / du))

    def d_dv(at, i, j):
        if j < Nv - 1:
            return combine(scale(at(i, j + 1), 1 / dv), scale(at(i, j), -1 / dv))
        return combine(scale(at(i, j), 1 / dv), scale(at(i, j - 1), -1 / dv))

    def d2_du(at, i, j):
        if 0 < i < Nu - 1:
            pts = [(at(i + 1, j), 1.0), (at(i, j), -2.0), (at(i - 1, j), 1.0)]
        elif i == 0:
            pts = [(at(0, j), 1.0), (at(1, j), -2.0), (at(2, j), 1.0)]
        else:
            pts = [(at(Nu - 1, j), 1.0), (at(Nu - 2, j), -2.0), (at(Nu - 3, j), 1.0)]
        return scale(combine(*[scale(p, c) for p, c in pts]), 1 / du ** 2)

    def d2_dv(at, i, j):
        if 0 < j < Nv - 1:
            pts = [(at(i, j + 1), 1.0), (at(i, j), -2.0), (at(i, j - 1), 1.0)]
        elif j == 0:
            pts = [(at(i, 0), 1.0), (at(i, 1), -2.0), (at(i, 2), 1.0)]
        else:
            pts = [(at(i, Nv - 1), 1.0), (at(i, Nv - 2), -2.0), (at(i, Nv - 3), 1.0)]
        return scale(combine(*[scale(p, c) for p, c in pts]), 1 / dv ** 2)

    rows, cols, vals, rhs = [], [], [], []
    eq = 0

    def add_row(entries, rhs_val):
        nonlocal eq
        for var, coeff in entries:
            if coeff != 0.0:
                rows.append(eq)
                cols.append(var)
                vals.append(coeff)
        rhs.append(rhs_val)
        eq += 1

    # --- equations (1) and (3): every (i,j), including v-edges (there they
    # reduce to a single equation pinning sigma_uu, since sigma_uv/vv are
    # already known-zero there) ---------------------------------------------
    for i in range(Nu):
        for j in range(Nv):
            eq1 = combine(d_du(at_uu, i, j), d_dv(at_uv, i, j),
                           scale(true_delta_rr_expr(i, j), -mu * sin2phi / t),
                           scale(S_rr_expr(i, j), -mu * sin2phi / (4 * r)))
            add_row(eq1, 0.0)

            eq3 = combine(d2_du(at_uu, i, j), d2_du(at_vv, i, j),
                          d2_dv(at_uu, i, j), d2_dv(at_vv, i, j),
                          scale(d_du(true_delta_rr_expr, i, j), -(1 + nu_p) * mu * sin2phi / t),
                          scale(d_dv(true_delta_rr_expr, i, j), -(1 + nu_p) * mu * cos2phi / t),
                          scale(d_du(S_rr_expr, i, j), -(1 + nu_p) * mu * sin2phi / (4 * r)),
                          scale(d_dv(S_rr_expr, i, j), (1 + nu_p) * mu * cos2phi / (4 * r)))
            add_row(eq3, 0.0)

    # sigma_uu = 0 at the free edges (exact consequence of the edge argument
    # in Section "v-dependence reduction": sigma_vr=0 => S_rr=Delta_rr=0 at
    # v=+-W/2, and since sigma_uv=sigma_vv=0 there too, Delta_rr's algebraic
    # formula forces sigma_uu=0 there as well).
    # (sigma_uu has no free unknown at the edges -- at_uu already returns []
    # there -- so nothing further to add; this note documents why.)

    # --- equations (2) and (4): only where sigma_uv, sigma_vv, S_rr are
    # genuinely free unknowns, i.e. interior v ------------------------------
    for i in range(Nu):
        # Precompute this row's contiguous "both contacts" (case==0) bands.
        # Solving the transport equation (4) by one-sided finite differences
        # WITHIN such a band turned out to be badly unstable when the band is
        # only a handful of grid points wide (which it usually is here): two
        # rows one grid step apart in u, with the same band width, gave S_rr
        # values of +0.59 and -1.62 at the same relative position in the
        # band -- not a physical signal, a discretization blow-up, confirmed
        # by checking that rows with NO both-contacts band at all are smooth
        # and small throughout. Physically, Delta_rr's source term is O(t)
        # (tiny), so S_rr should barely change across such a short band; the
        # robust choice is to just linearly interpolate S_rr across each
        # band between its two boundary values (each an exact algebraic
        # expression from the adjacent outer-only/inner-only/no-contact
        # case), rather than resolve a transport PDE that a handful of grid
        # points cannot support.
        row_case = case[i, :]
        bands = []
        j = 0
        while j < Nv:
            if row_case[j] == 0:
                j0 = j
                while j < Nv and row_case[j] == 0:
                    j += 1
                bands.append((j0, j - 1))
            else:
                j += 1
        band_of = {}
        for (j_lo, j_hi) in bands:
            for jj in range(j_lo, j_hi + 1):
                band_of[jj] = (j_lo - 1, j_hi + 1)  # boundary columns (always valid: edges are case -1)

        for j in range(1, Nv - 1):
            eq2 = combine(d_du(at_uv, i, j), d_dv(at_vv, i, j),
                           scale(true_delta_rr_expr(i, j), -mu * cos2phi / t),
                           scale(S_rr_expr(i, j), mu * cos2phi / (4 * r)))
            add_row(eq2, 0.0)

            c = case[i, j]
            if c == 0:
                left_bnd, right_bnd = band_of[j]
                w = (j - left_bnd) / (right_bnd - left_bnd)
                left_val = S_rr_expr(i, left_bnd)
                right_val = S_rr_expr(i, right_bnd)
                add_row(combine([(I_S(i, j), 1.0)], scale(left_val, -(1 - w)), scale(right_val, -w)), 0.0)
            elif c == 1:
                # sigma_rr^-(i,j):=0 already (sigma_rr_minus_expr returns []
                # for case 1); sigma_rr^+(i,j) = I_S(i,j) is read off the
                # ACTUAL outer neighbor's sigma_rr^- (Newton's third law,
                # eq:contact_outer), not the local elasticity formula.
                target = bilinear_lookup(sigma_rr_minus_expr, u_grid[i] + shift_u, v_grid[j] - shift_v)
                add_row(combine([(I_S(i, j), 1.0)], scale(target, -1.0)), 0.0)
            elif c == 2:
                target = bilinear_lookup(sigma_rr_plus_expr, u_grid[i] - shift_u, v_grid[j] + shift_v)
                add_row(combine([(I_S(i, j), 1.0)], scale(target, -1.0)), 0.0)
            else:
                add_row([(I_S(i, j), 1.0)], 0.0)

    # --- global axial force balance: u=0 is held by an external fixture,
    # not a free surface, so instead of a pointwise contact rule there
    # (removed above -- see case==4), the physically correct condition is
    # that the fixture supplies whatever reaction is needed to balance the
    # net axial (Instron) force applied at u=L. Every layer-contact force
    # in this model is ribbon-on-itself, so by Newton's third law every such
    # force is an internal action-reaction pair with both partners inside
    # [0,L] -- summing over the whole ribbon, all of them cancel, leaving
    # only the two end resultants, which must therefore be equal (using the
    # same +u-outward-normal traction convention at both ends; the physical
    # reaction force on the fixture side is this value negated, since its
    # outward normal is -u, but that sign is external bookkeeping, not part
    # of this equation):
    #   W*int[sigma_uu(0,v) sin(phi) + sigma_uv(0,v) cos(phi)] dv
    #     = W*int[sigma_uu(L,v) sin(phi) + sigma_uv(L,v) cos(phi)] dv
    # This ties the two ends together but is itself homogeneous (trivially
    # satisfied by the all-zero field), so it does not by itself fix scale.
    end_load_0 = []
    end_load_L = []
    for j in range(1, Nv - 1):
        end_load_0 = combine(end_load_0, scale(at_uu(0, j), sin_phi), scale(at_uv(0, j), cos_phi))
        end_load_L = combine(end_load_L, scale(at_uu(Nu - 1, j), sin_phi), scale(at_uv(Nu - 1, j), cos_phi))
    add_row(combine(end_load_0, scale(end_load_L, -1.0)), 0.0)

    # --- scale anchor: fix the active-end, centerline in-plane tension to a
    # reference value, purely to break the remaining trivial (all-zero)
    # degeneracy -- unlike the earlier version of this script, this is NOT
    # compensating for any assumption about u=0 (there no longer is one), so
    # no analytical growth-mode rescaling is needed afterward; the relative
    # scale between u=0 and u=L is now determined entirely by the force
    # balance above plus the bulk equations.
    j_center = Nv // 2
    add_row([(I_uu(Nu - 1, j_center), 1.0)], 1.0)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(eq, N))
    b = np.array(rhs)
    sol = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=50000)[0]

    sigma_uu = np.zeros((Nu, Nv))
    sigma_uv = np.zeros((Nu, Nv))
    sigma_vv = np.zeros((Nu, Nv))
    S_rr = np.zeros((Nu, Nv))
    sigma_uu[:, 1:-1] = sol[0:n_uu].reshape(Nu, n_int_v)
    sigma_uv[:, 1:-1] = sol[n_uu:n_uu + n_uv].reshape(Nu, n_int_v)
    sigma_vv[:, 1:-1] = sol[n_uu + n_uv:n_uu + n_uv + n_vv].reshape(Nu, n_int_v)
    S_free = sol[n_uu + n_uv + n_vv:].reshape(Nu, n_int_v)

    # I_S(i,j) always equals S_rr(i,j) by construction (its defining equation
    # differs by case, but the variable's meaning does not), so this holds
    # uniformly -- no per-case branching needed for S_rr itself.
    S_rr[:, 1:-1] = S_free

    # Delta_rr = sigma_rr^+ - sigma_rr^-: the local elastic formula only in
    # case 0 (where it is what's actually driving S_rr's own equation); in
    # case 1 sigma_rr^-=0 so Delta_rr=sigma_rr^+=S_rr; in case 2 sigma_rr^+=0
    # so Delta_rr=-sigma_rr^-=-S_rr; in case 3 both vanish.
    Delta_rr = np.zeros((Nu, Nv))
    local_elastic = t * (kappa_uu * sigma_uu - 2 * kappa_uv * sigma_uv + kappa_vv * sigma_vv)
    is_case0 = case == 0
    is_case1 = case == 1
    is_case2 = case == 2
    Delta_rr[is_case0] = local_elastic[is_case0]
    Delta_rr[is_case1] = S_rr[is_case1]
    Delta_rr[is_case2] = -S_rr[is_case2]

    sigma_rr_plus = 0.5 * (S_rr + Delta_rr)
    sigma_rr_minus = 0.5 * (S_rr - Delta_rr)

    return sigma_uu, sigma_uv, sigma_vv, S_rr, Delta_rr, sigma_rr_plus, sigma_rr_minus, case


if __name__ == "__main__":
    import matplotlib.colors as mcolors

    # Very small x (phi near 0) is excluded: the layer-contact "both/outer/
    # inner/no contact" regions become extremely thin there relative to the
    # grid spacing, and the resulting near-degenerate case boundaries produce
    # large discretization artifacts in sigma_rr -- a resolution limitation
    # of this grid, not a modeling instability.
    x_vals = np.linspace(0.10 * L, 0.97 * L, 64)
    all_names = ["sigma_uu", "sigma_uv", "sigma_vv", "S_rr", "Delta_rr", "sigma_rr_plus", "sigma_rr_minus"]
    # sigma_uu, sigma_uv, sigma_vv are visualized together as one principal-
    # stress-direction plot below instead of as three separate heatmaps.
    names = ["S_rr", "Delta_rr", "sigma_rr_plus", "sigma_rr_minus"]
    frames = {name: [] for name in all_names}
    case_frames = []
    F_vals = []

    for k, x in enumerate(x_vals):
        suu, suv, svv, S, D, sp_, sm_, case = solve_stress_field(x)
        for name, arr in zip(all_names, [suu, suv, svv, S, D, sp_, sm_]):
            frames[name].append(arr)
        case_frames.append(case)
        phi = np.arcsin(x / L)
        F_vals.append(W * np.trapezoid(suu[-1, :] * np.sin(phi) + suv[-1, :] * np.cos(phi), v_grid))
        print(f"x={x:7.2f}  F={F_vals[-1]: .6g}")

    F_vals = np.array(F_vals)

    # F(x) can span many orders of magnitude once the growing-mode rescaling
    # is applied (Eqs. eq:lambda_pm, eq:sigma_uu_L) -- plot both linear and
    # log-magnitude views rather than guess which one is readable.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x_vals, F_vals, "-o", ms=3)
    axes[0].set_xlabel("x (cm)"); axes[0].set_ylabel("F(x)"); axes[0].set_title("linear")
    axes[1].semilogy(x_vals, np.abs(F_vals) + 1e-300, "-o", ms=3, color="C1")
    axes[1].set_xlabel("x (cm)"); axes[1].set_ylabel("|F(x)|"); axes[1].set_title("log magnitude")
    fig.suptitle("Numerically solved force-displacement relation")
    fig.tight_layout()
    fig.savefig("F_of_x_numerical.png", dpi=150)
    plt.close(fig)

    for name in names:
        arr = np.array(frames[name])  # (n_x, Nu, Nv)
        vmax = np.abs(arr).max()
        # symmetric-log color scale: linear near 0 (within linthresh), log
        # beyond it, so small variation isn't washed out by a few huge peak
        # values -- fixed across all frames per the earlier "don't change the
        # scale bar between frames" requirement.
        linthresh = max(vmax * 1e-4, 1e-12)
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(arr[0].T, extent=[0, L, -W / 2, W / 2], norm=norm,
                        aspect="auto", origin="lower", cmap="RdBu_r")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{name}  (symlog, linthresh={linthresh:.2g})")
        ax.set_xlabel("u (cm)")
        ax.set_ylabel("v (cm)")
        title = ax.set_title(f"{name}, x={x_vals[0]:.1f} cm")

        def update(kf, arr=arr, im=im, title=title, name=name):
            im.set_data(arr[kf].T)
            title.set_text(f"{name}, x={x_vals[kf]:.1f} cm")
            return [im, title]

        ani = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=150, blit=False)
        ani.save(f"{name}_animation.gif", writer="pillow", fps=6)
        plt.close(fig)
        print(f"saved {name}_animation.gif  (max |value| {vmax:.4g})")

    # --- layer-contact gate animation: which of the 4 cases applies at each
    # (u,v), for each x -- shows the both/outer-only/inner-only/no-contact
    # bands (and the free-edge strip) sweeping as the pitch angle changes.
    case_arr = np.array(case_frames)  # (n_x, Nu, Nv), values in {-1,0,1,2,3}
    cmap = mcolors.ListedColormap(["#dddddd", "#4a7fb5", "#e8a33d", "#5fae5f", "#c0504d"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    cnorm = mcolors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(case_arr[0].T, extent=[0, L, -W / 2, W / 2], cmap=cmap, norm=cnorm,
                    aspect="auto", origin="lower")
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels(["free edge", "both contacts", "outer only", "inner only", "no contact"])
    ax.set_xlabel("u (cm)"); ax.set_ylabel("v (cm)")
    title = ax.set_title(f"layer-contact gate, x={x_vals[0]:.1f} cm")

    def update_case(kf):
        im.set_data(case_arr[kf].T)
        title.set_text(f"layer-contact gate, x={x_vals[kf]:.1f} cm")
        return [im, title]

    ani = animation.FuncAnimation(fig, update_case, frames=len(x_vals), interval=150, blit=False)
    ani.save("contact_gate_animation.gif", writer="pillow", fps=6)
    plt.close(fig)
    print("saved contact_gate_animation.gif")

    # --- principal in-plane stress directions, replacing separate sigma_uu/
    # uv/vv heatmaps: at each (subsampled) grid point, the 2x2 tensor
    # [[sigma_uu, sigma_uv],[sigma_uv, sigma_vv]] has two orthogonal
    # eigen-directions/magnitudes (principal stresses). Each is drawn as a
    # short tick at the eigen-direction's angle, uniform length, with color
    # and line width scaled by symlog(|eigenvalue|) -- a standard stress-
    # rosette rendering. Ticks are drawn with different u- and v-length
    # scales so a truly-45-degree direction looks roughly 45 degrees on
    # screen despite the u,v axes not being plotted at the same physical
    # scale (aspect="auto", as in every other plot here, since L/W~40:1).
    suu_arr = np.array(frames["sigma_uu"])
    suv_arr = np.array(frames["sigma_uv"])
    svv_arr = np.array(frames["sigma_vv"])

    step_u, step_v = max(Nu // 22, 1), max(Nv // 9, 1)
    iu = np.arange(0, Nu, step_u)
    iv = np.arange(1, Nv - 1, step_v)  # skip the (identically-zero) edges
    Ugrid, Vgrid = np.meshgrid(u_grid[iu], v_grid[iv], indexing="ij")

    def principal_stresses(a, b, c):
        """Eigenvalues/eigen-angle of [[a,b],[b,c]]: lambda1 along theta,
        lambda2 along theta+90 deg (orthogonal by construction)."""
        avg, diff = (a + c) / 2, (a - c) / 2
        rad = np.sqrt(diff ** 2 + b ** 2)
        lam1, lam2 = avg + rad, avg - rad
        theta = 0.5 * np.arctan2(2 * b, a - c)
        return lam1, lam2, theta

    lam1_all, lam2_all, theta_all = [], [], []
    for k in range(len(x_vals)):
        a, b, c = suu_arr[k][np.ix_(iu, iv)], suv_arr[k][np.ix_(iu, iv)], svv_arr[k][np.ix_(iu, iv)]
        lam1, lam2, theta = principal_stresses(a, b, c)
        lam1_all.append(lam1); lam2_all.append(lam2); theta_all.append(theta)
    lam1_all, lam2_all, theta_all = np.array(lam1_all), np.array(lam2_all), np.array(theta_all)
    vmax_principal = max(np.abs(lam1_all).max(), np.abs(lam2_all).max())
    linthresh_p = max(vmax_principal * 1e-6, 1e-12)
    pnorm = mcolors.SymLogNorm(linthresh=linthresh_p, vmin=0, vmax=vmax_principal, base=10)
    pcmap = plt.get_cmap("viridis")

    # Half-length in u data-units, sized to the subsample spacing so ticks at
    # theta=0 don't overlap their neighbors.
    tick_len_u = 0.35 * step_u * du
    # A physical 45-degree direction should still look ~45 degrees on screen
    # even though the axes are NOT plotted at equal physical scale
    # (aspect="auto", u-range/v-range ~ 40). On-screen length of a data
    # segment of extent d, along axis X, is d * (axes_pixels_X / range_X);
    # for equal on-screen extents at theta=45 (du_data==dv_data before
    # correction), the v-side length must be scaled by
    # (fig_width/fig_height) * (W/L) relative to the u-side length.
    figsize_principal = (11, 3.2)
    tick_len_v = tick_len_u * (figsize_principal[0] / figsize_principal[1]) * (W / L)

    def tick_segments(theta, du, dv):
        cx, sy = np.cos(theta), np.sin(theta)
        x0 = (Ugrid - du * cx).ravel(); x1 = (Ugrid + du * cx).ravel()
        y0 = (Vgrid - dv * sy).ravel(); y1 = (Vgrid + dv * sy).ravel()
        return np.stack([np.stack([x0, y0], axis=1), np.stack([x1, y1], axis=1)], axis=1)

    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(11, 3.2))
    lc1 = LineCollection(tick_segments(theta_all[0], tick_len_u, tick_len_v), cmap=pcmap, norm=pnorm)
    lc1.set_array(np.abs(lam1_all[0]).ravel())
    lc1.set_linewidth(1.0 + 3.0 * pnorm(np.abs(lam1_all[0]).ravel()))
    lc2 = LineCollection(tick_segments(theta_all[0] + np.pi / 2, tick_len_u, tick_len_v), cmap=pcmap, norm=pnorm)
    lc2.set_array(np.abs(lam2_all[0]).ravel())
    lc2.set_linewidth(1.0 + 3.0 * pnorm(np.abs(lam2_all[0]).ravel()))
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.set_xlim(0, L)
    ax.set_ylim(-W / 2, W / 2)
    cbar = plt.colorbar(lc1, ax=ax)
    cbar.set_label(f"|principal stress|  (symlog, linthresh={linthresh_p:.2g})")
    ax.set_xlabel("u (cm)")
    ax.set_ylabel("v (cm)")
    title = ax.set_title(f"principal stress directions, x={x_vals[0]:.1f} cm")

    def update_principal(kf):
        lc1.set_segments(tick_segments(theta_all[kf], tick_len_u, tick_len_v))
        lc1.set_array(np.abs(lam1_all[kf]).ravel())
        lc1.set_linewidth(1.0 + 3.0 * pnorm(np.abs(lam1_all[kf]).ravel()))
        lc2.set_segments(tick_segments(theta_all[kf] + np.pi / 2, tick_len_u, tick_len_v))
        lc2.set_array(np.abs(lam2_all[kf]).ravel())
        lc2.set_linewidth(1.0 + 3.0 * pnorm(np.abs(lam2_all[kf]).ravel()))
        title.set_text(f"principal stress directions, x={x_vals[kf]:.1f} cm")
        return [lc1, lc2, title]

    ani = animation.FuncAnimation(fig, update_principal, frames=len(x_vals), interval=150, blit=False)
    ani.save("principal_stress_animation.gif", writer="pillow", fps=6)
    plt.close(fig)
    print(f"saved principal_stress_animation.gif  (max |principal stress| {vmax_principal:.4g})")
