syms x

L        = 200;         % Total length of the strip (cm)
r0_val   = 2;            % Base coil radius at x = 0 (cm)
W        = 10;           % Ribbon width (cm)
theta_L  = L/r0_val;      % Total wind angle (rad)
mu       = 0.3;          % Coefficient of friction
t        = 0.01;         % Ribbon thickness (cm)

% Geometry at extension x (Eqs. eq:phi, eq:r)
r   = sqrt(L^2 - x^2)/theta_L;
phi = asin(x/L);

% Curvature and accumulation-system coefficients (Eq. eq:ABC_def)
kappa_uu = cos(phi)^2/r;
A = mu*sin(2*phi)*kappa_uu;
B = mu*sin(2*phi)/(4*r);
C = t*cos(phi)/(pi*r^2);

% Eigenvalues of the (sigma_uu, S_hat) accumulation system (Eq. eq:lambda_pm)
% Always real since B,C > 0 => determinant -B*C < 0 => no oscillation.
Lambda      = sqrt(A^2 + 4*B*C);
lambda_plus  = (A + Lambda)/2;
lambda_minus = (A - Lambda)/2;

% Boundary-condition constants from sigma_rr^+(L,v) = 0 (Eq. eq:sigma_uu_L)
mu_plus  = (lambda_plus  - A) + t*kappa_uu*B;
mu_minus = (lambda_minus - A) + t*kappa_uu*B;

A0 = 1;  % free scale fixing the overall magnitude of the pull

% sigma_uu(L) and the resulting end load (Eqs. eq:sigma_uu_L, eq:F_final)
sigma_uu_L = A0*B*(1 - mu_plus/mu_minus)*exp(lambda_plus*L);
F = simplify(W*sigma_uu_L*sin(phi));

x_vals = linspace(0.01*L, 0.99*L, 100);
F_vals = double(subs(F, x, x_vals));
plot(x_vals, F_vals)
xlabel('x (cm)')
ylabel('F(x)')
title('Force-displacement relation (corrected, non-oscillatory derivation)')
