"""
breaking_bad_model_so3.py -- the 3-axis 'Breaking Bad' case study with the
Home -> Doctor -> Home loop.   Run:  python3 breaking_bad_model_so3.py

Axes (weights, all positive at the start):
    0 Order/Safety     1 Fairness/Justice     2 Autonomy/Self-preservation (hub)
"""
import numpy as np
from scipy.linalg import expm

def make_generator(v0, hub, weights):
    """Skew-symmetric generator g of a context.

    weights = {axis_index: strength}.  Strengths are normalised to length 1 so
    all contexts have comparable strength.  u is the unit vector of
    'valuations attacked'; its sign follows the sign of the baseline valuation,
    so that every attacked valuation moves toward zero while the hub grows.

        g = e_hub u^T - u e_hub^T          (a rotation in the plane of e_hub and u)

    Returns (g, u).
    """
    n = len(v0)
    norm = np.sqrt(sum(w * w for w in weights.values()))
    u = np.zeros(n)
    for axis, w in weights.items():
        u[axis] = np.sign(v0[axis]) * w / norm
    e = np.eye(n)[hub]
    g = np.outer(e, u) - np.outer(u, e)
    return g, u


def gate(v, v0, hub, k=3.2):
    """Completeness r of the return edge of a Street loop, between 0 and 1.

    r = 1 while the hub valuation is at its baseline (the loop closes exactly);
    r falls linearly to 0 as the hub valuation rises above baseline.
    k = 3.2 is an assumed gate strength.
    """
    rise = max(v[hub] - v0[hub], 0.0) / np.linalg.norm(v0)
    return float(np.clip(1.0 - k * rise, 0.0, 1.0))


def simulate(v0, hub, g_doctor, g_street, events, doctor_angle, street_angle):
    """Walk an agent through a sequence of loops.

    events        : list of 'D' (Home->Doctor->Home) and 'S' (Home->Street->Home)
    doctor_angle  : angle of the doctor diagnosis
    street_angle  : angle of every Street trip
    Doctor loops have r = 0 (bad news cannot be undone).
    Street loops have r = gate(...) (r = 1 if gate_on is False).

    Returns (final vector, life rotation W, log).  Each log entry holds the
    edge matrices, the holonomy H of that loop, and the vector before, in the
    middle of (after the outbound edge) and after the loop.
    """
    v = np.array(v0, float)
    W = np.eye(len(v))
    log, d = [], 0
    for ev in events:
        if ev == "D":
            theta = doctor_angle
            d += 1
            g, r = g_doctor, 0.0
        else:
            theta, g = street_angle, g_street
            r = 0.0
        E_out = expm(theta * g)
        E_back = expm(-r * theta * g)
        H = E_back @ E_out
        v_mid = E_out @ v
        v_new = E_back @ v_mid
        W = H @ W
        log.append(dict(event=ev, theta=theta, r=r, E_out=E_out, E_back=E_back,
                        H=H, v_before=v.copy(), v_mid=v_mid, v_after=v_new))
        v = v_new
    return v, W, log


def rotation_angles_deg(R):
    """Principal rotation angles of a rotation matrix, in degrees (largest first)."""
    ang = np.sort(np.abs(np.angle(np.linalg.eigvals(R))))[::2]
    return np.degrees(ang[::-1][: len(R) // 2])


def axis_of_rotation_3d(R):
    """Rotation axis of a 3x3 rotation (unit vector), from its skew part."""
    a = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return a / np.linalg.norm(a)


def axis_of_generator_3d(g):
    """Rotation axis vector a such that g = [a]_x (3x3 only)."""
    return np.array([g[2, 1], g[0, 2], g[1, 0]])


# The story: two diagnoses, each followed by Street trips; 7 loops in total.
STORY = ["D", "S", "S", "D", "S", "S", "S"]
HANK = ["S"] * len(STORY)          # Hank: same number of Street trips, no diagnoses

DOCTOR_ANGLES = 0.872665    # radians; escalating bad news. 50° = 0.872665
STREET_ANGLE = 0.0174533    # radians per Street trip. 1° = 0.0174533

np.set_printoptions(precision=3, suppress=True, linewidth=140)
AXES = ["Order", "Fairness", "Autonomy"]
HUB = 2

# ---- 1. Starting valuation vector (identical for Hank and Walt) --------------
V0 = np.array([0.95, 0.30, 0.05])
V0 = V0 / np.linalg.norm(V0)            # (0.952, 0.301, 0.050), length 1

# ---- 2. Contexts: which valuations does each place attack, and how hard? ----
DOCTOR_W = {0: 0, 1: 0.6}           # a diagnosis affects fairness less
STREET_W = {0: 0.4, 1: 0}           # the street mainy affects order
g_doc, u_doc = make_generator(V0, HUB, DOCTOR_W)
g_str, u_str = make_generator(V0, HUB, STREET_W)

print("=== SETUP ===")
print("start vector (Hank and Walt):", V0)
print("Doctor affects (unit vector u):", u_doc, "  -> generator g_D:\n", g_doc)
print("   rotation axis of Doctor:", axis_of_generator_3d(g_doc))
print("Street affects (unit vector u):", u_str, "  -> generator g_S:\n", g_str)
print("   rotation axis of Street:", axis_of_generator_3d(g_str))
print("Doctor angles (rad):", DOCTOR_ANGLES, " = deg", np.degrees(DOCTOR_ANGLES).round(1))
print("Street angle (rad):", STREET_ANGLE, " = deg", round(np.degrees(STREET_ANGLE), 1))

# ---- 3. Walk both agents through life -----------------------------------------
vW, W_walt, logW = simulate(V0, HUB, g_doc, g_str, STORY, DOCTOR_ANGLES, STREET_ANGLE)
vH, W_hank, logH = simulate(V0, HUB, g_doc, g_str, HANK, DOCTOR_ANGLES, STREET_ANGLE)

print("\n=== EDGE MATRICES OF SELECTED LOOPS ===")
def show(title, rec):
    print(f"\n{title}   angle={rec['theta']:.2f} rad, return fraction r={rec['r']:.2f}")
    print("outbound edge:\n", rec["E_out"])
    print("return edge:\n", rec["E_back"])
    print("holonomy H = return @ outbound:\n", rec["H"])
show("Walt, loop 1: Home->Doctor->Home (first diagnosis)", logW[0])
show("Walt, loop 2: Home->Street->Home (first trip after the diagnosis)", logW[1])
show("Walt, loop 5: Home->Street->Home (gate fully open)", logW[4])
show("Hank, loop 1: Home->Street->Home", logH[0])
print("\nHank: return edge equals transpose of outbound edge?",
      np.allclose(logH[0]["E_back"], logH[0]["E_out"].T))

print("\n=== WALT: EDGES VISITED AND VALUATION VECTOR (Order, Fairness, Autonomy) ===")
print(f"{'#':>2} {'loop':<22} {'r':>4}  {'after outbound edge':<24} {'back home':<24}")
print(f"{'':>2} {'start':<22} {'':>4}  {'':<24} {np.array2string(V0):<24}")
for i, rec in enumerate(logW, 1):
    name = "Home>Doctor>Home" if rec["event"] == "D" else "Home>Street>Home"
    print(f"{i:>2} {name:<22} {rec['r']:>4.2f}  {np.array2string(rec['v_mid']):<24} "
          f"{np.array2string(rec['v_after']):<24}")

print("\n=== HOLONOMIES ===")
for i, rec in enumerate(logW, 1):
    print(f"Walt loop {i:>2} ({rec['event']}): rotation angle of H = "
          f"{rotation_angles_deg(rec['H'])[0]:6.2f} deg")
print("Hank: every loop holonomy is the identity:",
      all(np.allclose(r["H"], np.eye(3)) for r in logH))
print("\nWalt's net life rotation W = H14 @ ... @ H1:\n", W_walt)
print("   angle:", rotation_angles_deg(W_walt)[0].round(1), "deg   axis:",
      axis_of_rotation_3d(W_walt))
print("Hank's net life rotation is the identity:", np.allclose(W_hank, np.eye(3)))

print("\n=== FINAL VALUATION VECTORS ===")
print("start :", V0)
print("Walt without diagnosis  :", vH)
print("Walt with diagnosis  :", vW)
print("distance Walt-Hank:", round(np.linalg.norm(vW - vH), 2),
      "  angle between them:",
      round(np.degrees(np.arccos(vW @ vH / np.linalg.norm(vW) / np.linalg.norm(vH))), 1), "deg")
print("check: lengths are conserved:", np.linalg.norm(vW).round(6), np.linalg.norm(vH).round(6))

# ---- 4. Counterfactuals --------------------------------------------------------
print("\n=== COUNTERFACTUALS ===")
v, _, _ = simulate(V0, HUB, g_doc, g_str, ["S"] * 14, DOCTOR_ANGLES, STREET_ANGLE)
print("Walt without any diagnosis (14 Street trips):", v)
v, _, _ = simulate(V0, HUB, g_doc, g_str, ["D"] * 4 + ["S"] * 10, DOCTOR_ANGLES, STREET_ANGLE)
print("all 4 diagnoses first, then 10 Street trips :", v)
v, _, _ = simulate(V0, HUB, g_doc, g_str, ["S"] * 10 + ["D"] * 4, DOCTOR_ANGLES, STREET_ANGLE)
print("10 Street trips first, then 4 diagnoses     :", v)
v, _, _ = simulate(V0, HUB, g_doc, g_str, STORY, DOCTOR_ANGLES, STREET_ANGLE)
print("Hank given Walt's diagnoses (same rules)    :", v, "  (identical to Walt by construction)")
