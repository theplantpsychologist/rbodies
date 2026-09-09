%% energy_closed_form.m
%  Closed-form total elastic energy of the coil, its minimising radius, the
%  stress/strain fields of the minimising configuration, and F(x).
%
%  Structure follows the report: build u_el = (1/2) sigma_ij eps_ij, eliminate
%  unknowns, integrate, minimise over r.  Sections marked  % FIX  are corrections
%  to the original draft of this script; Section S1 shows that the system as
%  posed has NO solution, and S2 states the single relaxation used to proceed.
clear

% ===================== known constants =====================
syms x L theta_L t W E nu real positive      % FIX: L was declared twice

% ===================== geometric descriptor =====================
syms r real positive
% FIX: atan(x, r*theta_L) is not valid syntax (atan is one-argument; atan2 would
% be needed).  Carrying Lambda and the two direction cosines instead keeps every
% expression rational in r and x, which is what makes the algebra below collapse.
Lambda  = sqrt((r*theta_L)^2 + x^2);         % arc length of the mid-surface line
sin_phi = x/Lambda;
cos_phi = r*theta_L/Lambda;
r_nat   = L/theta_L;

% ===================== curvature =====================
% FIX: the ACTUAL curvature kappa_ij and the MISMATCH Delta kappa_ij are
% different tensors and are not interchangeable.  Equilibrium and the moment
% balance are written in the deformed configuration and use kappa_ij; only the
% bending part of the strain uses the mismatch.
kappa_uu = cos_phi^2/r;
kappa_uv = sin_phi*cos_phi/r;
kappa_vv = sin_phi^2/r;

% FIX: the original assigned k_vv from kappa_uv and k_uv from kappa_vv (swapped).
k_uu = kappa_uu - 1/r_nat;                   % natural state: [1/r_nat, 0, 0]
k_uv = kappa_uv - 0;
k_vv = kappa_vv - 0;

% Gaussian curvature vanishes identically -- used repeatedly below.
assert(isAlways(simplify(kappa_uu*kappa_vv - kappa_uv^2) == 0))

% ===================== core unknown strain components =====================
syms zeta u v real
syms e_uu(u,v) e_uv(u,v) e_vv(u,v)
epsilon_uu = e_uu + zeta*k_uu;               % Kirchhoff normality: exact, and the
epsilon_uv = e_uv + zeta*k_uv;               % only place the mismatch belongs
epsilon_vv = e_vv + zeta*k_vv;

% ===================== self-contact recursion =====================
% FIX: the original wrote the recursion twice, forward and backward, as two
% sequential assignments.  They are the same statement (Newtons third law
% Taylor-expanded over one turn), so only one is independent, and writing both as
% assignments makes sigma_minus a function of itself.  In terms of the mean and
% the jump the single relation is
%
%     Delta sigma_rr = 2*pi*r*( cos_phi d/du - sin_phi d/dv ) sbar_rr
%                    = 2*pi * d(sbar_rr)/d(theta),
%
% the second form because cos_phi d/du - sin_phi d/dv = (1/r) d/dtheta exactly
% (change of variables from Eq. position).  The jump across one thickness is the
% change over one full turn, which is what the recursion says.
syms s_rr(u,v) ds_rr(u,v)
recursion = ds_rr == 2*pi*r*( cos_phi*diff(s_rr,u) - sin_phi*diff(s_rr,v) );

% FIX: sigma_rr is NOT linear in zeta.  Radial equilibrium gives
% d(sigma_rr)/dzeta = sigma_theta(zeta)/r, and sigma_theta is affine in zeta
% because the in-plane stresses are, so sigma_rr is quadratic.  The quadratic
% part is proportional to (zeta^2 - t^2/4), which vanishes at BOTH faces, so it
% changes no surface value; P is fixed by the first moment of r_eq (Section S0).
syms P(u,v)
sigma_rr = s_rr + zeta/t*ds_rr + P/2*(zeta^2 - t^2/4);

% ===================== constitutive relations =====================
sigma_uu = E/(1-nu^2)*(epsilon_uu + nu*epsilon_vv) + nu/(1-nu)*sigma_rr;
sigma_vv = E/(1-nu^2)*(epsilon_vv + nu*epsilon_uu) + nu/(1-nu)*sigma_rr;
sigma_uv = E/(1+nu)*epsilon_uv;
epsilon_rr = (1+nu)*(1-2*nu)/(E*(1-nu))*sigma_rr - nu/(1-nu)*(epsilon_uu + epsilon_vv);

% ===================== energy density =====================
u_el = (sigma_uu*epsilon_uu + 2*sigma_uv*epsilon_uv + sigma_vv*epsilon_vv ...
        + sigma_rr*epsilon_rr)/2;

% ===================== transverse shears =====================
% FIX: assumption 5 (linear through-thickness profile) must NOT be applied here.
% A function linear in zeta that vanishes at both faces is identically zero, so a
% linear shear profile forces the shear FORCE to vanish whenever the contact
% faces are shear-free -- which is false.  Carry a general quadratic instead:
% u_eq/v_eq have an affine source, so the solution is quadratic up to O(t/r)
% (the circumferential component picks up an exponential, expanded below).
syms s_ur(u,v) s_vr(u,v) ds_ur(u,v) ds_vr(u,v) p_ur(u,v) p_vr(u,v)
sigma_ur = s_ur + zeta/t*ds_ur + p_ur/2*(zeta^2 - t^2/4);
sigma_vr = s_vr + zeta/t*ds_vr + p_vr/2*(zeta^2 - t^2/4);
Q_ur = int(sigma_ur, zeta, -t/2, t/2);       % shear force per unit width
Q_vr = int(sigma_vr, zeta, -t/2, t/2);
X_ur = int(zeta*sigma_ur, zeta, -t/2, t/2);  % first moment of the shear
X_vr = int(zeta*sigma_vr, zeta, -t/2, t/2);

% ===================== moments =====================
M_uu = simplify(int(zeta*sigma_uu, zeta, -t/2, t/2));
M_uv = simplify(int(zeta*sigma_uv, zeta, -t/2, t/2));
M_vv = simplify(int(zeta*sigma_vv, zeta, -t/2, t/2));

% ===================== equilibrium =====================
% FIX: kappa_ij, not k_ij; and the v-equation signs were flipped.
u_eq = diff(sigma_uu,u) + diff(sigma_uv,v) + diff(sigma_ur,zeta) ...
       + kappa_uu*sigma_ur - kappa_uv*sigma_vr == 0;
v_eq = diff(sigma_uv,u) + diff(sigma_vv,v) + diff(sigma_vr,zeta) ...
       - kappa_uv*sigma_ur + kappa_vv*sigma_vr == 0;
r_eq = diff(sigma_ur,u) + diff(sigma_vr,v) + diff(sigma_rr,zeta) ...
       - kappa_uu*sigma_uu + 2*kappa_uv*sigma_uv - kappa_vv*sigma_vv == 0;

% ===================== moment balance =====================
% FIX: kappa_ij, not k_ij.  FIX: the boundary term of the by-parts step does NOT
% cancel -- with shear-free faces it leaves the shear force Q, so the right-hand
% sides are Q_ur and Q_vr, not zero.  (The cancellation used in the original
% assumes a linear shear profile, which forces Q = 0; see the note above.)
u_mom = diff(M_uu,u) + diff(M_uv,v) + kappa_uu*X_ur - kappa_uv*X_vr == Q_ur;
v_mom = diff(M_uv,u) + diff(M_vv,v) - kappa_uv*X_ur + kappa_vv*X_vr == Q_vr;
% The first moment of r_eq: the by-parts term here DOES cancel against the mean,
% but only for the linear part of sigma_rr; the quadratic part survives and this
% equation determines P rather than vanishing.
r_mom = diff(X_ur,u) + diff(X_vr,v) + P*t^3/12 ...
        - kappa_uu*M_uu + 2*kappa_uv*M_uv - kappa_vv*M_vv == 0;
% Projecting u_mom, v_mom on m = [sin_phi; cos_phi] kills the curvature terms,
% because [kappa_uu -kappa_uv; -kappa_uv kappa_vv] = (1/r)*n*n' with
% n = [cos_phi; -sin_phi], and m'*n = 0.  That gives the classical plate relation
%     Q_z = m * [dM_uu/du + dM_uv/dv ; dM_uv/du + dM_vv/dv],
% determining the axial shear force.  The moment balance therefore removes no
% unknown; it is needed only because M_ij is how the bending stress reaches the
% shear source in u_eq/v_eq.

% ===================== compatibility: arc length =====================
% Holds on the mid-surface material line, for each v.
arc_length = int(e_uu, u, 0, L) + L == Lambda;

% Boundary conditions: sigma_uv = sigma_vv = sigma_vr = 0 at v = +- W/2;
% sigma_rr^+ = 0 at u = L and sigma_rr^- = 0 at u = 0 (both ends of the chain).

%% ==================================================================
%% ============================ SOLUTION ============================
%% ==================================================================
% Everything above is the statement of the problem.  What follows solves it.
% No new governing equation is introduced: S0-S2 only manipulate the equations
% and boundary conditions already written down.

%% ---- S0. The two profiles that equilibrium determines --------------
% u_eq and v_eq, read as a first-order system in zeta for (sigma_ur,sigma_vr),
% carry a two-parameter solution family.  Shear-free contact prescribes FOUR
% values (two components on each of two faces), so the system is over-determined
% by two, and the two leftovers are conditions on the source
%     s = ( d(sigma_uu)/du + d(sigma_uv)/dv ,  d(sigma_uv)/du + d(sigma_vv)/dv ).
% Rotating into the circumferential and axial directions decouples them exactly,
% because the curvature matrix [kappa_uu -kappa_uv; -kappa_uv kappa_vv] equals
% (1/r)*n*n' with n = [cos_phi; -sin_phi] -- rank one, since K = 0.  Writing
% sbar and s' for the zeta-average and zeta-slope of the source, the conditions
% are
%       axial:            m'*sbar = 0                       (exact)
%       circumferential:  n'*sbar = -(t^2/12r) n'*s'         (+O(t^4/r^3))
% i.e. thickness-averaged in-plane equilibrium is source-free to O(t^2/r).  The
% profiles that follow are parabolic, vanishing at both faces with a nonzero
% resultant -- the classical Kirchhoff transverse shear.  Similarly the first
% moment of r_eq determines the quadratic coefficient P rather than vanishing:
%       P*t^3/12 = kappa_uu*M_uu - 2*kappa_uv*M_uv + kappa_vv*M_vv
%                  - d(Xi_ur)/du - d(Xi_vr)/dv.
% Both are recorded for completeness; neither survives the interior reduction.

%% ---- S1. Is the system solvable as posed?  No. --------------------
% Take the interior of the ribbon, where the in-plane field is uniform (source-
% free plane elasticity on a long strip with traction-free edges has the uniform
% uniaxial Saint-Venant field as its exact solution; self-equilibrated end data
% decays as exp(-4.212 u/W)).  Then every d/du and d/dv vanishes, so
%   (i)  the shear source vanishes, hence sigma_ur = sigma_vr = 0 identically;
%   (ii) the recursion gives Delta sigma_rr = 2*pi*d(sbar_rr)/d(theta) = 0;
%   (iii) the zeroth moment of r_eq then reads sigma_theta_bar = 0.
% Free edges make sigma_vv_bar = sigma_uv_bar = 0, and with sigma_rr = 0
% Hooke gives sigma_uu_bar = E*e_uu, so sigma_theta_bar = cos_phi^2*E*e_uu.
% Hence e_uu = 0 -- which contradicts the arc length, e_uu = Lambda/L - 1 > 0.
% Rebuild the fields with every (u,v) derivative set to zero.  Doing this from
% scratch, rather than substituting into the general expressions, keeps the
% reduction explicit: constant strains, no pressure gradient, no shear source.
syms e_uu_c e_uv_c e_vv_c s_rr_c P_rr real
eps_uu_c = e_uu_c + zeta*k_uu;
eps_uv_c = e_uv_c + zeta*k_uv;
eps_vv_c = e_vv_c + zeta*k_vv;
% ds_rr = 0 by the recursion (no theta gradient in the interior).  P does NOT
% vanish: r_mom with Xi = 0 gives P*t^3/12 = kappa_uu*M_uu - 2*kappa_uv*M_uv
% + kappa_vv*M_vv, and the Kirchhoff moments are nonzero.  This is the
% bending-induced pressure bump: quadratic in zeta, zero at both faces, so it
% changes no surface value and violates no contact condition.
D      = E*t^3/(12*(1-nu^2));
M_uu_c = D*(k_uu + nu*k_vv);
M_vv_c = D*(k_vv + nu*k_uu);
M_uv_c = D*(1-nu)*k_uv;
P_c    = simplify(12/t^3*(kappa_uu*M_uu_c - 2*kappa_uv*M_uv_c + kappa_vv*M_vv_c));
sig_rr_c = s_rr_c + P_rr/2*(zeta^2 - t^2/4);   % P_rr is a symbol; = P_c below
sig_uu_c = E/(1-nu^2)*(eps_uu_c + nu*eps_vv_c) + nu/(1-nu)*sig_rr_c;
sig_vv_c = E/(1-nu^2)*(eps_vv_c + nu*eps_uu_c) + nu/(1-nu)*sig_rr_c;
sig_uv_c = E/(1+nu)*eps_uv_c;
eps_rr_c = (1+nu)*(1-2*nu)/(E*(1-nu))*sig_rr_c - nu/(1-nu)*(eps_uu_c + eps_vv_c);
u_el_c   = (sig_uu_c*eps_uu_c + 2*sig_uv_c*eps_uv_c + sig_vv_c*eps_vv_c ...
            + sig_rr_c*eps_rr_c)/2;

N_vv = simplify(int(sig_vv_c, zeta, -t/2, t/2)/t);
N_uv = simplify(int(sig_uv_c, zeta, -t/2, t/2)/t);
edge = solve([N_vv == 0, N_uv == 0], [e_vv_c, e_uv_c]);
edge0.e_vv_c = subs(edge.e_vv_c, P_rr, 0);   % pressure-free, for S1
edge0.e_uv_c = subs(edge.e_uv_c, P_rr, 0);
N_uu = simplify(subs(subs(int(sig_uu_c, zeta, -t/2, t/2)/t, P_rr, 0), ...
                     [e_vv_c, e_uv_c], [edge0.e_vv_c, edge0.e_uv_c]));
sigma_theta_bar = simplify(cos_phi^2*N_uu);
fprintf('\nS1  free edges  ->  e_vv = %s ,  e_uv = %s\n', ...
        char(simplify(edge.e_vv_c)), char(simplify(edge.e_uv_c)));
fprintf('S1  sigma_uu_bar    = %s\n', char(N_uu));
fprintf('S1  sigma_theta_bar = %s\n', char(sigma_theta_bar));
fprintf('S1  setting it to zero with s_rr = 0 gives e_uu = %s\n', ...
        char(solve(subs(sigma_theta_bar, s_rr_c, 0) == 0, e_uu_c)));
fprintf('S1  the arc length demands e_uu = Lambda/L - 1 > 0.  INCONSISTENT.\n');

%% ---- S2. The one relaxation ---------------------------------------
% The obstruction is the pair of free ends of the contact chain: sigma_rr = 0 at
% the outer face of the last turn AND at the inner face of the first.  Integrating
% r_eq across the stack between them turns that into  int sigma_theta dR/R = 0,
% which a uniform, one-signed sigma_theta cannot satisfy; representing the
% required sign change needs a radius that varies across the stack, which the
% uniform-r assumption forbids.  Two admissible ways out, neither in closed form:
%   (a) let sigma_uv carry a profile across the width -- the free edges constrain
%       it only AT v = +-W/2 -- which nulls sigma_theta_bar with sigma_uu_bar
%       nonzero, at a cost (1+nu)*sigma_uv_bar^2/E, sigma_uv_bar = E*e_uu*cot(phi)/2;
%   (b) bear the innermost turn on a hub, which supplies the missing pressure and
%       leaves the energy untouched.
% We take sigma_rr = 0 and bound (a) at the end.

%% ---- S3-S5. Free edges and arc length ------------------------------
% Two versions: P kept (the honest field) and P dropped (the closed form).  The
% difference is priced at the end; it is O(1e-4) in F.
e_vv_0 = simplify(subs(edge.e_vv_c, s_rr_c, 0));
e_uv_0 = simplify(subs(edge.e_uv_c, s_rr_c, 0));
u_el_g = subs(u_el_c, [s_rr_c, e_vv_c, e_uv_c], [0, e_vv_0, e_uv_0]);
u_el_g = subs(u_el_g, e_uu_c, Lambda/L - 1);
u_el_P = subs(u_el_g, P_rr, 0);           % the closed form: P dropped
u_el_0 = subs(u_el_g, P_rr, P_c);         % the honest field: P kept

%% ---- S6. Closed-form total energy ---------------------------------
U_total = simplify(L*W*int(u_el_P, zeta, -t/2, t/2));   % P dropped
U_withP = simplify(L*W*int(u_el_0, zeta, -t/2, t/2));   % P kept
fprintf('\nS6  closed-form total elastic energy obtained.\n');
% Cross-check against the report's grouping:
%   U/(E L W t) = tau^2/(24(1-nu^2)) [ (1/rho-1)^2 + 2(1-nu)chi^2/(rho lam^2)
%                                      + Gamma (lam-1)^2 ],  tau = t/r_nat
rho = r*theta_L/L;  chi = x/L;  lam = sqrt(rho^2+chi^2);
tau = t/r_nat;      Gam = 12*(1-nu^2)/tau^2;
U_ref = E*L*W*t*tau^2/(24*(1-nu^2)) * ...
        ( (1/rho-1)^2 + 2*(1-nu)*chi^2/(rho*lam^2) + Gam*(lam-1)^2 );
fprintf('S6  U_total - U_report = %s   (should be 0)\n', ...
        char(simplify(expand(U_total - U_ref))));

%% ---- S7. Minimising radius, fields, and F(x) ----------------------
% dU/dr = 0 at fixed x selects the geometry; F = dU/dx at fixed r (envelope
% theorem), so dr/dx is never needed.
dUdr = simplify(diff(U_total, r));
F_of_x = simplify(diff(U_total, x));

pars = {L, W, t, theta_L, E, nu};
vals = {200, 8, 0.01, 100, 3e5, 0.3};
dUdr_n = matlabFunction(subs(dUdr,  pars, vals), 'Vars', [r x]);
U_n    = matlabFunction(subs(U_total, pars, vals), 'Vars', [r x]);
F_n    = matlabFunction(subs(F_of_x, pars, vals), 'Vars', [r x]);
Lv = 200; thv = 100; rnv = Lv/thv;

fprintf('\n   x/L      rho        eps          sig_uu[N/cm2]   F[N]\n');
for xi = [0.2 0.4 0.6 0.8 0.9 0.95 0.99]
    xv = xi*Lv;
    % bracket the root in r, then bisect
    rs = linspace(1e-3, 0.999*rnv, 4000);
    g  = arrayfun(@(rr) dUdr_n(rr,xv), rs);
    k  = find(sign(g(1:end-1)) ~= sign(g(2:end)));
    rstar = fzero(@(rr) dUdr_n(rr,xv), [rs(k(end)) rs(k(end)+1)]);
    lamv  = sqrt((rstar*thv)^2 + xv^2)/Lv;
    fprintf('  %5.2f  %8.5f  %11.4e  %11.2f  %10.4f\n', ...
            xi, rstar/rnv, lamv-1, 3e5*(lamv-1), F_n(rstar,xv));
end

% Price of dropping P (the bending-induced quadratic pressure):
dUP = matlabFunction(subs(simplify(diff(U_withP,r)), pars, vals), 'Vars',[r x]);
FP  = matlabFunction(subs(simplify(diff(U_withP,x)), pars, vals), 'Vars',[r x]);
fprintf('\n   x/L    dF/F from dropping P\n');
for xi = [0.4 0.8 0.99]
    xv = xi*Lv;  rs = linspace(1e-3,0.999*rnv,4000);
    g0 = arrayfun(@(rr) dUdr_n(rr,xv), rs); k0 = find(sign(g0(1:end-1))~=sign(g0(2:end)));
    r0 = fzero(@(rr) dUdr_n(rr,xv), [rs(k0(end)) rs(k0(end)+1)]);
    g1 = arrayfun(@(rr) dUP(rr,xv), rs);    k1 = find(sign(g1(1:end-1))~=sign(g1(2:end)));
    r1 = fzero(@(rr) dUP(rr,xv), [rs(k1(end)) rs(k1(end)+1)]);
    fprintf('  %5.2f  %+10.3e\n', xi, FP(r1,xv)/F_n(r0,xv)-1);
end

% Bound on the relaxation of S2: add the width-shear energy (1+nu)*sig_uv^2/E
U_shear = U_total + L*W*t*(1+nu)*(E*(Lambda/L-1)*cos_phi/(2*sin_phi))^2/E;
dUs = matlabFunction(subs(simplify(diff(U_shear,r)), pars, vals), 'Vars',[r x]);
Fs  = matlabFunction(subs(simplify(diff(U_shear,x)), pars, vals), 'Vars',[r x]);
fprintf('\n   x/L    dF/F from the S2 relaxation\n');
for xi = [0.4 0.8 0.99]
    xv = xi*Lv;
    rs = linspace(1e-3,0.999*rnv,4000);
    g0 = arrayfun(@(rr) dUdr_n(rr,xv), rs);  k0 = find(sign(g0(1:end-1))~=sign(g0(2:end)));
    r0 = fzero(@(rr) dUdr_n(rr,xv), [rs(k0(end)) rs(k0(end)+1)]);
    g1 = arrayfun(@(rr) dUs(rr,xv), rs);     k1 = find(sign(g1(1:end-1))~=sign(g1(2:end)));
    r1 = fzero(@(rr) dUs(rr,xv), [rs(k1(end)) rs(k1(end)+1)]);
    fprintf('  %5.2f  %+10.4f %%\n', xi, 100*(Fs(r1,xv)/F_n(r0,xv)-1));
end
