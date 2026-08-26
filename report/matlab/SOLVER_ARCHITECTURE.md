# Numerical stress-field solver — architecture and status

This document describes `numerical_solver.py`: a finite-difference solver for
the full $(u,v)$ stress field of the paper-yoyo model, at a fixed extension
$x$, solving the boxed governing equations directly on a 2D grid rather than
the analytical treatment's reduced ($v$-mode-truncated) system. The main body
below describes the current architecture as implemented. The final section is
an honest account of problems that remain, approaches tried, and the
trade-offs each one introduced.

## 1. Unknown fields and governing equations

At a fixed $x$ (hence fixed $\phi,r$), the unknowns are four scalar fields of
$(u,v)$:

$$\bar\sigma_{uu}(u,v),\quad \bar\sigma_{uv}(u,v),\quad \bar\sigma_{vv}(u,v),\quad S_{rr}(u,v) := \sigma_{rr}^+ + \sigma_{rr}^-,$$

with $\Delta\sigma_{rr} := \sigma_{rr}^+ - \sigma_{rr}^-$ recovered algebraically
rather than tracked as an independent unknown. The governing equations are the
boxed relations from the analytical write-up, used **before** any $v$-mode
truncation:

$$\frac{\partial\bar\sigma_{uu}}{\partial u} + \frac{\partial\bar\sigma_{uv}}{\partial v} = \mu\sin(2\phi)\left[\frac{\Delta\sigma_{rr}}{t} + \frac{S_{rr}}{4r}\right] \qquad\text{(1, eq:inplane\_u\_integrated)}$$

$$\frac{\partial\bar\sigma_{uv}}{\partial u} + \frac{\partial\bar\sigma_{vv}}{\partial v} = \mu\cos(2\phi)\left[\frac{\Delta\sigma_{rr}}{t} - \frac{S_{rr}}{4r}\right] \qquad\text{(2, eq:inplane\_v\_integrated)}$$

$$\nabla^2(\bar\sigma_{uu}+\bar\sigma_{vv}) = (1+\nu)\left\{\frac{\mu}{t}\left[\sin(2\phi)\frac{\partial \Delta\sigma_{rr}}{\partial u} + \cos(2\phi)\frac{\partial \Delta\sigma_{rr}}{\partial v}\right] + \frac{\mu}{4r}\left[\sin(2\phi)\frac{\partial S_{rr}}{\partial u} - \cos(2\phi)\frac{\partial S_{rr}}{\partial v}\right]\right\} \quad\text{(3, eq:beltrami\_michell)}$$

$$\pi r\cos\phi\,\frac{\partial S_{rr}}{\partial u} - \pi r\sin\phi\,\frac{\partial S_{rr}}{\partial v} = t\left[\kappa_{uu}\bar\sigma_{uu} - 2\kappa_{uv}\bar\sigma_{uv} + \kappa_{vv}\bar\sigma_{vv}\right] \qquad\text{(4, eq:S\_transport, "both contacts")}$$

$$\Delta\sigma_{rr} = t\left[\kappa_{uu}\bar\sigma_{uu} - 2\kappa_{uv}\bar\sigma_{uv} + \kappa_{vv}\bar\sigma_{vv}\right] \qquad\text{(eq:outofplane\_exact, leading order)}$$

All four equations are **linear** at fixed $x$, since $\phi,r,\kappa_{uu},\kappa_{uv},\kappa_{vv}$
are constants once $x$ is frozen. The whole solve is therefore one large
sparse linear system per $x$; there is no time-stepping or nonlinear
iteration.

## 2. Grid and unknown layout

```python
Nu, Nv = 81, 41
u_grid = np.linspace(0.0, L, Nu)
v_grid = np.linspace(-W/2, W/2, Nv)
```

Traction-free long edges ($v=\pm W/2$) force $\bar\sigma_{uv}=\bar\sigma_{vv}=0$
there exactly, and — since $\sigma_{vr}=0$ at a free edge ties directly to
$S_{rr}$ — also $S_{rr}=\Delta\sigma_{rr}=0$, which in turn forces
$\bar\sigma_{uu}=0$ there too (its only source, $\Delta\sigma_{rr}$, vanishes).
So **all four fields** are pinned to zero at the two edge columns, and only
interior-$v$ grid points ($j=1,\dots,N_v-2$) carry a free unknown for any
field:

```python
n_int_v = Nv - 2
n_uu, n_uv, n_vv, n_S = (Nu * n_int_v,) * 4
N = n_uu + n_uv + n_vv + n_S   # total unknowns in the sparse system
```

Every equation is assembled as a **symbolic linear expression** — a list of
`(variable_index, coefficient)` pairs — built up with two helpers:

```python
def combine(*groups):        # sum several such expressions
    ...
def scale(entries, factor):  # multiply one by a constant
    ...
```

This lets every governing equation be written once, symbolically, in terms of
whatever combination of unknowns (or algebraic sub-expressions) actually
belongs there, then dropped into a sparse matrix row.

## 3. Layer-contact cases

At each interior grid point, whether the point has a genuine outer and/or
inner wound neighbor is a purely geometric question, from the contact
recursion (eq:contact\_outer, eq:contact\_inner):

```python
shift_u = 2*np.pi*r*np.cos(phi)
shift_v = 2*np.pi*r*np.sin(phi)
has_outer = (v >= -W/2 + shift_v) and (u + shift_u <= L)
has_inner = (v <=  W/2 - shift_v) and (u - shift_u >= 0)
```

giving four cases:

| case | condition | rule |
|---|---|---|
| 0 | both | Eq. (4), $S_{rr}$ solved as part of the linear system |
| 1 | outer only | $\sigma_{rr}^-:=0$; $\sigma_{rr}^+$ read off the **actual outer neighbor's** $\sigma_{rr}^-$ |
| 2 | inner only | $\sigma_{rr}^+:=0$; $\sigma_{rr}^-$ read off the **actual inner neighbor's** $\sigma_{rr}^+$ |
| 3 | neither | $\sigma_{rr}^+=\sigma_{rr}^-=0$ |

The axial part of the outer gate ($u+\text{shift}_u>L$) automatically forces
case 3/2 for the whole width at $u=L$, reproducing $\sigma_{rr}^+(L,v)=0$ (no
material wound past the active end) with no separate boundary condition
needed. $u=0$ is classified by the **same rule** as every interior row (see
§6 for why).

Because the true contact partner $(u\pm\text{shift}_u, v\mp\text{shift}_v)$
generally does not land on a grid point, cases 1 and 2 use **bilinear
interpolation** of the neighbor's stress rather than a local formula:

```python
def bilinear_lookup(expr_fn, u_target, v_target):
    """Linear expression for expr_fn at the continuous location
    (u_target, v_target), via bilinear interpolation of the 4 nearest
    grid points."""
    ...

# case 1 (outer only):
target = bilinear_lookup(sigma_rr_minus_expr, u_grid[i] + shift_u, v_grid[j] - shift_v)
# case 2 (inner only):
target = bilinear_lookup(sigma_rr_plus_expr,  u_grid[i] - shift_u, v_grid[j] + shift_v)
```

This is what makes the contact relation satisfy Newton's third law by
construction: $\sigma_{rr}^+$ at a case-1 point is defined to literally equal
(an interpolate of) $\sigma_{rr}^-$ at its real contact partner, not an
unrelated local quantity.

$\Delta\sigma_{rr}$ used **generally** (in Eqs. 1–3) is *not* the algebraic
elasticity formula everywhere — that formula is only valid, by the document's
own derivation, as the source term *inside* the case-0 transport equation
(4), where it is equated against the contact-recursion expression for
$\Delta\sigma_{rr}$ to eliminate it and obtain a pure PDE for $S_{rr}$. The
general $\Delta\sigma_{rr}$ used in Eqs. (1)–(3) is instead the *true*
$\sigma_{rr}^+-\sigma_{rr}^-$, built from whichever case-appropriate
expressions for $\sigma_{rr}^\pm$ apply at that point:

```python
def true_delta_rr_expr(i, j):
    return combine(sigma_rr_plus_expr(i, j), scale(sigma_rr_minus_expr(i, j), -1.0))
```

### The "both contacts" band is solved by interpolation, not the transport PDE

Eq. (4) is a first-order (hyperbolic/transport-like) PDE. Case-0 regions are
typically only a handful of grid points wide, and directly finite-differencing
Eq. (4) inside such a narrow band was found to be numerically unstable (see
§7). Instead, $S_{rr}$ across a case-0 band is **linearly interpolated**
between its two exact boundary values (the case-1/2/3 expressions on either
side):

```python
w = (j - left_bnd) / (right_bnd - left_bnd)
S_rr(i,j) := (1-w) * S_rr_expr(i, left_bnd) + w * S_rr_expr(i, right_bnd)
```

justified physically because $\Delta\sigma_{rr}$ (Eq. 4's source) is $O(t)$,
i.e. tiny, so $S_{rr}$ should barely change across such a short span anyway.

## 4. Spatial derivatives: one-sided, never central

Every $\partial/\partial u$, $\partial/\partial v$ is a **one-sided**
(forward, or backward only at the very last index) finite difference:

```python
def d_du(at, i, j):
    if i < Nu - 1:
        return combine(scale(at(i+1, j), 1/du), scale(at(i, j), -1/du))
    return combine(scale(at(i, j), 1/du), scale(at(i-1, j), -1/du))
```

Second derivatives (needed for the Laplacian in Eq. 3) use the standard
3-point one-sided stencil at boundaries and centered 3-point stencil in the
interior — these *do* reference the center point directly, unlike a
skip-the-center first-derivative central difference.

**Why not central differences for the first derivatives:** a plain central
difference $(f_{i+1}-f_{i-1})/2h$ never references $f_i$ itself, which
decouples the even- and odd-indexed grid points into two independent
sub-lattices with no coupling between them (a "checkerboard" null mode). An
early version of this solver used central differences and produced a large,
unphysical, alternating-sign oscillation in $v$ as a direct result. One-sided
differences avoid this by construction.

## 5. Boundary treatment at $u=0,L$

Equations (1)–(3) are applied at $u=0$ and $u=L$ using the **same one-sided
differencing of the bulk PDE itself** — a "natural free end" closure, not an
independently derived condition on $\bar\sigma_{uu},\bar\sigma_{uv}$ there.
This is a modeling simplification, not a physical derivation, and is the
main reason the assembled system ends up very slightly over-determined (see
§6), so it is solved by least squares rather than an exact square solve.

## 6. Global axial force balance (the $u=0$ condition)

$u=0$ is held by an external fixture, not a free surface. "No inner
*layer*" (nothing wound at negative $u$) is still a true geometric fact and
is *not* relaxed — $u=0$ is classified into cases 1/2/3 by the ordinary rule
in §3, exactly like every other boundary-adjacent row. What is different at
$u=0$ is that **no pointwise condition on $\bar\sigma_{uu},\bar\sigma_{uv}$ is
assumed there at all.** Instead, since every layer-contact force in this
model is ribbon-on-itself, each one is a Newton's-third-law action-reaction
pair with both partners inside $[0,L]$; summed over the whole ribbon they all
cancel, leaving only the two end resultants, which must be equal:

$$W\!\int_{-W/2}^{W/2}\!\left[\bar\sigma_{uu}(0,v)\sin\phi+\bar\sigma_{uv}(0,v)\cos\phi\right]dv \;=\; W\!\int_{-W/2}^{W/2}\!\left[\bar\sigma_{uu}(L,v)\sin\phi+\bar\sigma_{uv}(L,v)\cos\phi\right]dv$$

— i.e. the same integral that defines the Instron force $F(x)$
(eq:end\_load), evaluated at $u=0$ and $u=L$ and set equal. This single
equation is added directly to the sparse system:

```python
end_load_0 = sum(sin_phi*at_uu(0,j) + cos_phi*at_uv(0,j) for j in interior_v)
end_load_L = sum(sin_phi*at_uu(Nu-1,j) + cos_phi*at_uv(Nu-1,j) for j in interior_v)
add_row(end_load_0 - end_load_L, rhs=0.0)
```

It is itself homogeneous (trivially satisfied by the all-zero field), so it
ties the two ends together but does not fix the overall scale.

## 7. Scale anchor and solve

The whole system is otherwise homogeneous (no distributed load anywhere), so
one more equation is needed purely to break the trivial all-zero degeneracy:

```python
add_row([(I_uu(Nu-1, Nv//2), 1.0)], rhs=1.0)   # sigma_uu(L, v=0) := 1
```

This plays no physical role beyond setting units — with the force-balance
condition above in place, the *relative* scale between $u=0$ and $u=L$ comes
entirely from the bulk equations, not from this anchor.

The assembled system is solved by sparse least squares:

```python
A = sp.csr_matrix((vals, (rows, cols)), shape=(eq, N))
sol = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=50000)[0]
```

## 8. Post-processing and outputs

After solving, $S_{rr}$ is read directly off the solution vector (its
defining equation differs by case, but the variable's meaning does not).
$\Delta\sigma_{rr}$ is reconstructed per-case (elasticity formula in case 0;
$\pm S_{rr}$ in cases 1/2; zero in case 3), and $\sigma_{rr}^\pm =
\tfrac12(S_{rr}\pm\Delta\sigma_{rr})$.

Three kinds of animation (swept over $x$) are produced:

- **Per-component heatmaps** ($S_{rr}$, $\Delta\sigma_{rr}$,
  $\sigma_{rr}^+$, $\sigma_{rr}^-$): `imshow` with a `SymLogNorm` (linear
  near zero, log beyond a small threshold) and a color scale fixed across
  all frames of that component.
- **Layer-contact gate**: a categorical heatmap of the case array (0–3, plus
  the free-edge strip), showing the contact bands sweep as $x$ changes.
- **Principal stress directions**: $\bar\sigma_{uu},\bar\sigma_{uv},\bar\sigma_{vv}$
  are combined into the $2\times2$ tensor $\begin{bmatrix}a&b\\b&c\end{bmatrix}$
  at each (subsampled) grid point, eigen-decomposed
  ($\lambda_{1,2}=\tfrac{a+c}2\pm\sqrt{(\tfrac{a-c}2)^2+b^2}$,
  $\theta=\tfrac12\operatorname{atan2}(2b,a-c)$), and drawn as two
  perpendicular ticks (uniform length, color/linewidth by
  $\operatorname{symlog}|\lambda|$) — replacing three separate heatmaps with
  one stress-rosette-style rendering. Tick half-lengths in $u$ and $v$ are
  scaled differently so a true $45°$ direction still looks $\sim45°$ on
  screen despite the axes not sharing a physical scale ($L/W\sim40$).
- `F(x)`: computed from the $u=L$ end-load integral at each $x$, plotted both
  linearly and in log-magnitude.

---

## 9. Current problems, what was tried, and what traded off against what

This section is a chronological, honest account — not a polished changelog.

### 9.1 Checkerboard oscillation (fixed)

**Symptom:** using plain central differences for first derivatives produced
large, alternating-sign noise in $v$.
**Cause:** central differences skip the center point, decoupling even/odd
grid points into independent sub-lattices.
**Fix:** switched to one-sided differences everywhere (§4). **Trade-off:**
lower (first-order) formal accuracy for first derivatives, in exchange for
removing the null mode.

### 9.2 Narrow "both contacts" band instability (fixed)

**Symptom:** solving Eq. (4) by finite differences inside case-0 bands
(often only ~10 grid points wide) gave wildly different $S_{rr}$ values
between adjacent $u$-rows with the same band width (e.g. $+0.59$ vs. $-1.62$
at the same relative position) — a discretization blow-up, not signal.
**Fix:** replaced the finite-difference PDE solve inside these bands with
linear interpolation between the band's two boundary values (§3).
**Trade-off:** loses whatever genuine curvature Eq. (4) would produce across
the band, in exchange for stability; justified by $\Delta\sigma_{rr}$ being
$O(t)$ (small) so the band's true profile should be nearly flat anyway.

### 9.3 The anchor/boundary-condition saga at $u=0$

Three different choices were tried for what "boundary condition" $u=0$
should carry, each trading a different failure mode for another:

1. **Anchor $\bar\sigma_{uu}(0,0)=1$, plus $\sigma_{rr}^-(0,v)=0$ pointwise
   (the original approach).** Numerically fragile: matching
   $\sigma_{rr}^+(L,v)=0$ forced the system's growing eigenmode to have a
   tiny coefficient relative to a unit value at $u=0$, so the solution was
   dominated by the decaying mode — $F(x)$ came out minuscule and even
   sign-changing across $x$.
2. **Anchor $\bar\sigma_{uu}(L,0)=1$ instead, keep $\sigma_{rr}^-(0,v)=0$.**
   Numerically robust, but wrong in a subtler way: since $F(x)\propto
   \bar\sigma_{uu}(L,\cdot)$, fixing it to a constant for every $x$ silently
   cancelled the friction-accumulation growth the model is supposed to
   predict — $F(x)$ tracked little more than $\sin\phi(x)$ (near-linear,
   saturating), and stress was near-zero everywhere except close to $u=L$.
   This was the direct cause of "why is there no stress except near $u=L$."
3. **Rescale the whole field after solving, by the analytical treatment's
   closed-form growth factor $B(x)(1-\mu_+/\mu_-)e^{\lambda_+(x)L}$.** Made
   $F(x)$ show real (if extreme — many orders of magnitude) growth, but this
   was later abandoned: it was compensating for approach (2)'s own artifact,
   and it imported the *reduced, $v$-independent* analytical model's
   $\lambda_+(x)$ trend, including that model's own non-monotonic
   peak-then-decline behavior as $x\to L$ (traced to $r(x)\to0$ making the
   reduced model degenerate near full extension) — not necessarily a
   property of the full model.
4. **Current approach: drop the pointwise $\sigma_{rr}^-(0,v)=0$ condition
   in favor of a global axial force-balance condition (§6), and drop the
   growth-mode rescaling entirely.** This directly fixed the "no stress away
   from $u=L$" problem (confirmed: $F(0)=F(L)$ to $10^{-4}$–$10^{-8}$ across
   the swept $x$ range) and gave a mild, believable $F(x)$ range instead of
   an astronomical one. **New problem it exposed:** the very first version
   of this (before §9.4's fix) left $S_{rr}(0,v)$ completely free with *no*
   local equation at all, which turned out to be under-constrained in a
   direction the force-balance condition can't see (an odd-in-$v$ mode
   integrates to zero, so nothing pins its amplitude) — observed directly as
   a large, smooth, odd-symmetric swing in $\bar\sigma_{uu}(0,v)$,
   $\bar\sigma_{vv}(0,v)$ unrelated to any applied load.

### 9.4 Newton's-third-law bug in the single-contact cases (fixed, but see 9.5)

**Symptom (reported by inspection, not just aggregate statistics):**
$\sigma_{rr}^+$ was positive (tensile) in a substantial fraction of the
domain, concentrated in "outer only" zones, which is not physically possible
for a pure contact pressure — two surfaces can only push on each other, not
pull — and is inconsistent with $\sigma_{rr}^+$ at one layer having to equal
$\sigma_{rr}^-$ at its actual contact partner.
**Cause:** cases 1/2 were computing $\sigma_{rr}^\pm$ from the *local
elasticity formula* for $\Delta\sigma_{rr}$ (valid only inside case 0's own
transport equation, per the document's derivation), with no reference
whatsoever to the actual neighboring point's stress.
**Fix:** cases 1/2 now read off the true contact partner's stress via
bilinear interpolation (§3), and the general $\Delta\sigma_{rr}$ used in
Eqs. (1)–(3) was correspondingly split into `local_elastic_delta_expr`
(case-0-only, as documented) vs. `true_delta_rr_expr` (general, all cases).
This also required reverting the §9.3-item-4 "$S_{rr}$ fully free at $u=0$"
change, since with the corrected case-1/2 rule $u=0$ no longer needs special
treatment at all — it is classified by the ordinary geometric rule like any
other row.

### 9.5 Regression: severe checkerboard noise in $\sigma_{rr}$, wild $F(x)$ (open)

**Symptom:** after 9.4's fix, $\sigma_{rr}^+$ heatmaps show dense,
grid-scale, alternating-sign noise with no smooth structure (confirmed
visually — not a subtle effect), and $F(x)$, previously smoothly monotonic
under approach 9.3-item-4, became wildly non-monotonic (e.g. $4.4\to
2.0\to7.9\to0.66\to8.8\to0.31$ across successive $x$ steps).

**Diagnosis so far:** this is *not* an iterative-solver convergence
failure. Checked directly via `scipy.sparse.linalg.lsmr`'s diagnostics:
`normar` (the least-squares optimality residual, $\|A^Tr\|$) is tiny
($\sim10^{-4}$), meaning the solver has genuinely reached the best-fit
point; but `normr` (the residual itself) stays substantial ($\sim0.11$) even
at 400,000 iterations. That combination means **the assembled system is
genuinely over-determined and inconsistent** — there is no field that
satisfies all the equations simultaneously, and the leftover conflict is
being smeared across the unknowns as noise, not a matter of running the
solver longer.

**Leading hypothesis (untested):** the bilinear-interpolation lookups for
cases 1/2 make $\Delta\sigma_{rr}$ (and hence $S_{rr}$) a function of
*fractional* grid positions with weights that can shift discontinuously
between neighboring $(i,j)$ as the interpolation cell changes. Equation (3)
then differentiates this (via the same one-sided stencils used everywhere
else) — differentiating an interpolated lookup with a different, essentially
independent set of weights at each grid point is a plausible source of the
observed grid-scale noise, compounded by the already-approximate §9.2
interpolation inside case-0 bands.

**Not yet tried:** nearest-neighbor lookup instead of bilinear (avoids
fractional-position derivatives, at the cost of a discontinuous, less
"smooth" contact value); restructuring so the case-1/2 correction feeds only
Eqs. (1)/(2) and not the differentiated source in Eq. (3); or reconsidering
whether Eq. (3)'s source term needs the case-1/2 correction at all, given its
own derivation (eq:beltrami\_michell) assumes the same "both contacts"
regime that motivated the local elastic formula in the first place.

This is the current open problem: the physics of the contact fix (9.4) is
believed correct, but it has made the discretization measurably less
well-behaved than before, and that trade has not yet been resolved.
