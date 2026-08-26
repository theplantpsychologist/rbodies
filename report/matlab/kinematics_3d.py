"""
3D visualization of the actual helix kinematics as the active end is pulled
(x increasing), to sanity-check the geometry behind the layer-contact gates
(Eqs. eq:position, eq:phi, eq:r, eq:uminus--eq:vplus). Each (u,v) grid point
is mapped to its 3D position via cylindrical coordinates
    r(x) = sqrt(L^2-x^2)/theta_L,  phi(x) = asin(x/L),
    theta(u,v) = (u*cos(phi) - v*sin(phi)) / r,  z(u,v) = u*sin(phi) + v*cos(phi),
    X = r cos(theta), Y = r sin(theta), Z = z,
and the (u,v) grid is connected as a structured mesh with semi-transparent
faces and visible edges, colored by the same 4-case layer-contact
classification as contact_gate_animation.gif.

This view is rendered true to scale (auto-zoomed each frame to fill the
frame). A top-down (down-the-z-axis) view was tried and dropped: with
theta_L/(2*pi) ~ 16 turns at a nearly constant radius, every turn projects
onto nearly the same circle in that view, so turns just occlude each other
and the case pattern isn't legible there -- the flat (u,v) rendering in
contact_gate_animation.gif is already the right tool for seeing that pattern
clearly; this script's job is to check whether the 3D shape itself (the
radius shrinking, the coil elongating) looks like a sane, non-buggy
kinematic consequence of pulling the helix.
"""
import numpy as np
import pyvista as pv

L = 200.0
r0 = 1.0
W = 5.0
theta_L = L / r0

Nu_viz, Nv_viz = 700, 15
u = np.linspace(0.0, L, Nu_viz)
v = np.linspace(-W / 2, W / 2, Nv_viz)
U, V = np.meshgrid(u, v, indexing="ij")  # (Nu_viz, Nv_viz)


def positions(x):
    phi = np.arcsin(x / L)
    r = np.sqrt(L ** 2 - x ** 2) / theta_L
    theta = (U * np.cos(phi) - V * np.sin(phi)) / r
    z = U * np.sin(phi) + V * np.cos(phi)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Z = z
    return X, Y, Z


def contact_case(x):
    """Same 4-case classification as numerical_solver.py's gate animation,
    evaluated on this visualization grid, for coloring the mesh."""
    phi = np.arcsin(x / L)
    r = np.sqrt(L ** 2 - x ** 2) / theta_L
    shift_u = 2 * np.pi * r * np.cos(phi)
    shift_v = 2 * np.pi * r * np.sin(phi)
    has_outer = (V >= -W / 2 + shift_v) & (U + shift_u <= L)
    has_inner = (V <= W / 2 - shift_v) & (U - shift_u >= 0)
    case = np.full(U.shape, 0)
    case[has_outer & has_inner] = 0
    case[has_outer & ~has_inner] = 1
    case[~has_outer & has_inner] = 2
    case[~has_outer & ~has_inner] = 3
    return case


CMAP = ["#4a7fb5", "#e8a33d", "#5fae5f", "#c0504d"]

if __name__ == "__main__":
    x_vals = np.linspace(1.0, 199.0, 40)

    X0, Y0, Z0 = positions(x_vals[0])
    grid = pv.StructuredGrid(X0, Y0, Z0)
    grid["case"] = contact_case(x_vals[0]).ravel(order="F")

    pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
    pl.add_mesh(grid, scalars="case", cmap=CMAP, clim=[-0.5, 3.5], opacity=0.55,
                show_edges=True, edge_color="gray", line_width=0.3,
                show_scalar_bar=True,
                scalar_bar_args={"title": "0 both / 1 outer / 2 inner / 3 none", "n_labels": 4})
    pl.add_axes()
    pl.camera_position = "iso"
    title_actor = pl.add_text(f"x={x_vals[0]:.1f} cm   (r={np.sqrt(L**2-x_vals[0]**2)/theta_L:.3f} cm)",
                               position="upper_left", font_size=12)
    pl.open_gif("helix_kinematics.gif", fps=8)

    for x in x_vals:
        X, Y, Z = positions(x)
        grid.points = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
        grid["case"] = contact_case(x).ravel(order="F")
        r = np.sqrt(L ** 2 - x ** 2) / theta_L
        pl.remove_actor(title_actor)
        title_actor = pl.add_text(f"x={x:.1f} cm   (r={r:.3f} cm)", position="upper_left", font_size=12)
        pl.reset_camera()
        pl.camera_position = "iso"
        pl.write_frame()

    pl.close()
    print("saved helix_kinematics.gif")
