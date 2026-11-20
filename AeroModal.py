"""
Aircraft modal analysis, Useful for post balancing check
---------------------------------------------------------
Rigid cylinder (fuselage) + two independent rigid wings connected by 3-DOF elastic
rotational joints (flap, twist, sweep).  A point mass (Added_mass) can be attached to
each wing at a chosen Distance from the joint to represent the unbalance source or the added balancing mass.
The program solves a free-free eigenvalue problem and reports natural frequencies associated with mode shapes.

Axes: x = forward, y = span, z = up.
DOFs (12): fuselage [ux uy uz thx thy thz] + right wing [flap twist sweep]
           + left wing [flap twist sweep]   
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

try:                                 
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# ------------------------------------------------------------------ materials
MATERIALS = {            # E [Pa], G [Pa], rho [kg/m^3]
    "aluminium": dict(E=70e9, G=26e9, rho=2700),
    "steel":     dict(E=210e9, G=80e9, rho=7850),
    "cfrp":      dict(E=70e9, G=5e9,  rho=1600),
}
fus_mat = MATERIALS["aluminium"]
wing_mat = MATERIALS["cfrp"]

# ------------------------------------------------------------------ geometry
L_f, R_f = 1.0, 0.06                      # fuselage length, radius [m]
span, chord, thick = 0.8, 0.15, 0.01      # wing size [m]
x_h = 0.05                                # hinge x-position from fuselage COM

# ------------------------------------------------------------------ user input
def ask_float(prompt, default, lo, hi):
    while True:
        s = input(f"{prompt} [default {default}]: ").strip()
        if s == "":
            return default
        try:
            v = float(s)
        except ValueError:
            print("  Please enter a number.")
            continue
        if lo <= v <= hi:
            return v
        print(f"  Value must be between {lo} and {hi}.")

print(f"Wing span = {span} m. Enter 0 for Added_mass to run without it.\n")
added = {}
am_R = ask_float("RIGHT wing  Added_mass [kg]", 0.5, 0.0, 1e3)
ds_R = ask_float("RIGHT wing  Distance from joint [m]", 0.6, 0.0, span)
am_L = ask_float("LEFT  wing  Added_mass [kg] (Enter = same as right)", am_R, 0.0, 1e3)
ds_L = ask_float("LEFT  wing  Distance from joint [m] (Enter = same as right)", ds_R, 0.0, span)
added[+1] = (am_R, ds_R)
added[-1] = (am_L, ds_L)

# ------------------------------------------------------------------ helpers
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# fuselage: solid cylinder
m_f = fus_mat["rho"] * np.pi * R_f**2 * L_f
I_f = np.diag([0.5 * m_f * R_f**2,
               m_f * (3 * R_f**2 + L_f**2) / 12,
               m_f * (3 * R_f**2 + L_f**2) / 12])

# wing: thin box (inertia about its own COM)
m_w = wing_mat["rho"] * span * chord * thick
I_w = np.diag([m_w * (span**2 + thick**2) / 12,
               m_w * (chord**2 + thick**2) / 12,
               m_w * (chord**2 + span**2) / 12])

# joint stiffness from an equivalent cantilever
E, G = wing_mat["E"], wing_mat["G"]
k_flap = E * (chord * thick**3 / 12) / span       # about x
k_twist = G * (chord * thick**3 / 3) / span       # about y (span axis)
k_sweep = E * (thick * chord**3 / 12) / span      # about z
K_joint = np.diag([k_flap, k_twist, k_sweep])

# ------------------------------------------------------------------ assembly
ndof = 12
M = np.zeros((ndof, ndof))
K = np.zeros((ndof, ndof))

# fuselage
Jv = np.zeros((3, ndof)); Jv[:, 0:3] = np.eye(3)
Jw = np.zeros((3, ndof)); Jw[:, 3:6] = np.eye(3)
M += Jv.T @ (m_f * np.eye(3)) @ Jv + Jw.T @ I_f @ Jw

for side, start in ((+1, 6), (-1, 9)):            # +1 right wing, -1 left wing
    r_hinge = np.array([x_h, side * R_f, 0.0])
    sl = slice(start, start + 3)

    # (a) the rigid wing itself (COM at mid-span)
    c = np.array([0.0, side * span / 2, 0.0])     # hinge -> wing COM
    a = r_hinge + c                               # fuselage COM -> wing COM
    Jv = np.zeros((3, ndof))
    Jv[:, 0:3] = np.eye(3)
    Jv[:, 3:6] = -skew(a)
    Jv[:, sl] = -skew(c)
    Jw = np.zeros((3, ndof))
    Jw[:, 3:6] = np.eye(3)
    Jw[:, sl] = np.eye(3)
    M += Jv.T @ (m_w * np.eye(3)) @ Jv + Jw.T @ I_w @ Jw

    # (b) the added point mass, rigidly fixed to the wing at Distance from the joint
    m_add, dist = added[side]
    if m_add > 0:
        c_add = np.array([0.0, side * dist, 0.0])  # hinge -> point mass
        a_add = r_hinge + c_add
        Jp = np.zeros((3, ndof))
        Jp[:, 0:3] = np.eye(3)
        Jp[:, 3:6] = -skew(a_add)
        Jp[:, sl] = -skew(c_add)
        M += Jp.T @ (m_add * np.eye(3)) @ Jp       # point mass: no rotary inertia

    # (c) joint springs
    K[sl, sl] += K_joint

# ------------------------------------------------------------------ eigenvalue problem
vals, vecs = eigh(K, M)
vals[np.abs(vals) < 1e-6] = 0.0
freqs = np.sqrt(np.clip(vals, 0, None)) / (2 * np.pi)

print(f"\nFuselage mass {m_f:.2f} kg | wing mass (each) {m_w:.2f} kg")
print(f"Added mass: right {am_R:.3f} kg at {ds_R:.3f} m | left {am_L:.3f} kg at {ds_L:.3f} m")
print("\n #   f [Hz]")
for i, f in enumerate(freqs):
    print(f"{i+1:2d}  {f:8.3f}" + ("   (rigid body)" if f < 1e-3 else ""))

# ================================================================== MODE SHAPES
# Small-motion kinematics:  fuselage point p : d = u + th_f x p
#                           wing point p     : d = u + th_f x p + phi x (p - r_hinge)
def mode_geometry():
    th = np.linspace(0, 2 * np.pi, 25)
    xs = np.linspace(-L_f / 2, L_f / 2, 15)
    Xc, Tc = np.meshgrid(xs, th)
    fus = np.stack([Xc, R_f * np.cos(Tc), R_f * np.sin(Tc)], axis=-1)
    wings = {}
    for side in (+1, -1):
        xw = np.linspace(x_h - chord / 2, x_h + chord / 2, 4)
        yw = side * np.linspace(R_f, R_f + span, 12)
        Xw, Yw = np.meshgrid(xw, yw)
        wings[side] = np.stack([Xw, Yw, np.zeros_like(Xw)], axis=-1)
    return fus, wings

def point_masses():
    """Undeformed positions of the added masses (dict side -> point or None)."""
    out = {}
    for side in (+1, -1):
        m_add, dist = added[side]
        out[side] = np.array([x_h, side * (R_f + dist), 0.0]) if m_add > 0 else None
    return out

def deform(phi_vec, scale):
    u, thf = phi_vec[0:3], phi_vec[3:6]
    fus, wings = mode_geometry()
    pm = point_masses()
    fus_d = fus + scale * (u + np.cross(thf, fus))
    wings_d, pm_d = {}, {}
    for side, start in ((+1, 6), (-1, 9)):
        r_hinge = np.array([x_h, side * R_f, 0.0])
        phi = phi_vec[start:start + 3]
        pts = wings[side]
        wings_d[side] = pts + scale * (u + np.cross(thf, pts) + np.cross(phi, pts - r_hinge))
        p = pm[side]
        pm_d[side] = None if p is None else p + scale * (u + np.cross(thf, p) + np.cross(phi, p - r_hinge))
    return fus, wings, pm, fus_d, wings_d, pm_d

def draw_state(ax, phi_vec, scale, title):
    fus0, w0, pm0, fusd, wd, pmd = deform(phi_vec, scale)
    ax.plot_wireframe(*[fus0[..., k] for k in range(3)], color="lightgrey", lw=0.4)
    for s in (+1, -1):
        ax.plot_surface(*[w0[s][..., k] for k in range(3)], color="lightgrey", alpha=0.35)
        if pm0[s] is not None:
            ax.scatter(*pm0[s], color="lightgrey", s=60, depthshade=False)
    ax.plot_wireframe(*[fusd[..., k] for k in range(3)], color="dimgrey", lw=0.6)
    for s, col in ((+1, "tab:red"), (-1, "tab:blue")):
        ax.plot_surface(*[wd[s][..., k] for k in range(3)], color=col, alpha=0.8,
                        edgecolor="k", lw=0.3)
        if pmd[s] is not None:
            ax.scatter(*pmd[s], color="black", s=60, depthshade=False)
    ylim = R_f + span + 0.1
    ax.set_xlim(-0.5 * L_f, 0.5 * L_f); ax.set_ylim(-ylim, ylim); ax.set_zlim(-0.35, 0.35)
    ax.set_box_aspect((1, 2 * ylim, 0.7)); ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=-25)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=11)

def plot_modes(first=6, last=12, amp=0.25):
    """Two pictures per mode (+ and - extreme); one figure per mode, shown in turn."""
    for i in range(first, last):
        phi_vec = vecs[:, i]
        _, w0, _, _, wd1, _ = deform(phi_vec, 1.0)
        dmax = max(np.abs(wd1[s] - w0[s]).max() for s in (+1, -1))
        scale = amp * span / dmax                    
        fig = plt.figure(figsize=(13, 5))
        for k, sign in enumerate((+1, -1)):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            draw_state(ax, phi_vec, sign * scale,
                       f"Mode {i+1}, {freqs[i]:.2f} Hz: {'+' if sign > 0 else '-'} extreme")
        fig.suptitle("grey = undeformed | red = right wing (+y) | blue = left wing (-y) | "
                     "black dot = added mass", fontsize=9)
        fig.tight_layout()
        plt.show()

plot_modes()