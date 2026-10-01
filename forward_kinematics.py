#!/usr/bin/env python3
"""
Forward Kinematics — 5-DOF Gesture-Controlled Robotic Manipulator
=================================================================
DH-Convention implementation with proper robotics coordinate frame.

Coordinate Frame (origin at base centre, ground level)
------------------------------------------------------
  Z-axis : points UP   (vertical, positive above ground)
  X-axis : points RIGHT (direction arm extends at rest)
  Y-axis : follows right-hand rule (points forward/into the robot)

The XACRO model uses a CAD frame where Y is up.  This module applies
a fixed rotation to remap into the standard robotics frame above.

Frame remapping (XACRO → Robotics):
  X_new = -X_xacro     (arm extends in +X at rest)
  Y_new =  Z_xacro     (right-hand rule completion)
  Z_new =  Y_xacro     (Y was up in XACRO → Z is up now)

Kinematic Chain
---------------
  base_link ──(J1 Z)──▸ link_1_1 ──(J2 -X)──▸ link_2_1
  ──(J3 -X)──▸ link_3_1 ──(J4 +X)──▸ link_4_1 ──(J5 -Z)──▸ end_effector_1

DH Parameters (Modified Convention, after frame remapping)
----------------------------------------------------------
  i   θ_i (var)    d_i (mm)     a_i (mm)     α_i
  ─   ─────────    ────────     ────────     ────
  1   θ₁           40.605       28.16        ...
  2   θ₂           ...          ...          ...
  3   θ₃           ...          ...          ...
  4   θ₄           ...          ...          ...
  5   θ₅           ...          ...          ...

  (Full DH decomposition is approximate due to 3D offsets between
   joints; exact transforms from XACRO origins are used instead.)

Usage
-----
  # standalone — interactive 3D plot
  python3 forward_kinematics.py

  # as a library from p1.py
  from forward_kinematics import RoboticArmFK
  fk = RoboticArmFK()
  pos, T = fk.compute(base=90, shoulder=90, elbow=90, yaw=90, roll=90)
  print(f"End-effector: x={pos[0]:.1f} y={pos[1]:.1f} z={pos[2]:.1f} mm")
"""

import numpy as np
import math

# ════════════════════════════════════════════════════════════════
#  FRAME REMAPPING: XACRO (Y-up) → Robotics (Z-up, X-right)
# ════════════════════════════════════════════════════════════════
#
#  The XACRO/CAD frame has Y pointing up.  We want:
#    Z up, X to the right (arm extension direction at rest)
#
#  Remapping matrix R_remap (applied to all XACRO vectors):
#    X_new = -X_xacro
#    Y_new =  Z_xacro
#    Z_new =  Y_xacro
#
R_REMAP = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
])
# det(R_REMAP) = +1 (proper rotation) ✓


# ════════════════════════════════════════════════════════════════
#  LINK DATA (from XACRO joint <origin xyz="…"/> and <axis/>)
# ════════════════════════════════════════════════════════════════
#
#  Raw XACRO values (metres), then remapped to robotics frame.
#
#  Joint   Parent→Child          XACRO xyz (m)                    XACRO axis
#  ─────   ──────────────────    ─────────────────────────────    ──────────
#  J1      base_link→link_1_1   (-0.02816,  0.040605,  0.0    )  (0, +1, 0)
#  J2      link_1_1 →link_2_1   ( 0.01332,  0.050623,  0.00372)  (+1, 0, 0)
#  J3      link_2_1 →link_3_1   ( 0.09250,  0.011623, -0.04354)  (+1, 0, 0)
#  J4      link_3_1 →link_4_1   ( 0.04750, -0.092851, -0.05109)  (-1, 0, 0)
#  J5      link_4_1 →end_eff    (-0.29266,  0.000500, -0.09684)  (0, -1, 0)

_XACRO_ORIGINS_M = [
    np.array([-0.02816,   0.040605,   0.0      ]),   # J1
    np.array([ 0.01332,   0.050623,   0.00372  ]),   # J2
    np.array([ 0.09250,   0.011623,  -0.04354  ]),   # J3
    np.array([ 0.04750,  -0.092851,  -0.05109  ]),   # J4
    np.array([-0.292662,  0.000500,  -0.096842 ]),   # J5
]

_XACRO_AXES = [
    np.array([ 0.0,  1.0,  0.0]),   # J1  +Y
    np.array([ 1.0,  0.0,  0.0]),   # J2  +X
    np.array([ 1.0,  0.0,  0.0]),   # J3  +X
    np.array([-1.0,  0.0,  0.0]),   # J4  -X
    np.array([ 0.0, -1.0,  0.0]),   # J5  -Y
]

# ── Apply frame remapping ──────────────────────────────────────
# Joint origins in the robotics frame (Z-up, X-right), in metres
JOINT_ORIGINS_M = [R_REMAP @ v for v in _XACRO_ORIGINS_M]
# → J1: (+0.02816, 0.0, +0.040605)   — 28.2 mm to the right, 40.6 mm up
# → J2: (-0.01332, +0.00372, +0.050623) — 50.6 mm further up
# → J3: (-0.09250, -0.04354, +0.011623) — 11.6 mm further up
# → J4: (-0.04750, -0.05109, -0.092851) — 92.9 mm DOWN
# → J5: (+0.29266, -0.09684, +0.000500) — 0.5 mm up

# Joint rotation axes in the robotics frame
JOINT_AXES = [R_REMAP @ v for v in _XACRO_AXES]
# → J1: (0, 0, +1)  = +Z  (base swivel about vertical) ✓
# → J2: (-1, 0, 0)  = -X  (shoulder pitch)
# → J3: (-1, 0, 0)  = -X  (elbow pitch)
# → J4: (+1, 0, 0)  = +X  (wrist pitch, reversed)
# → J5: (0, 0, -1)  = -Z  (wrist roll about vertical)

# Joint limits (radians) from XACRO — J1 is continuous
JOINT_LIMITS_RAD = [
    (-np.inf,      np.inf     ),   # J1 continuous
    ( 0.0,         np.pi      ),   # J2  0  → π
    (-np.pi / 2,   np.pi / 2  ),   # J3 -π/2 → π/2
    (-np.pi,       0.0        ),   # J4 -π → 0
    (-np.pi,       0.017453   ),   # J5 -π → ~1°
]

# Servo-angle rest positions (degrees) matching p1.py
REST_ANGLES_DEG = [90.0, 90.0, 90.0, 90.0, 90.0]

# Computed link lengths in mm (invariant under rotation)
LINK_LENGTHS_MM = [round(np.linalg.norm(v) * 1000, 1) for v in JOINT_ORIGINS_M]
# → [49.4, 52.5, 102.9, 116.1, 308.3]  (same as before — rotation preserves lengths)


# ════════════════════════════════════════════════════════════════
#  HOMOGENEOUS TRANSFORM HELPERS (DH-style)
# ════════════════════════════════════════════════════════════════

def _rotation_matrix(axis, angle_rad):
    """Rodrigues' rotation formula for an arbitrary unit axis."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [ 0,       -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    return R


def _homogeneous(R, t):
    """Build 4×4 homogeneous transform from 3×3 rotation + 3-vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# ════════════════════════════════════════════════════════════════
#  FORWARD KINEMATICS CLASS  (DH Convention, Z-up Frame)
# ════════════════════════════════════════════════════════════════

class RoboticArmFK:
    """
    Forward kinematics for the 5-DOF gesture-controlled manipulator.

    All outputs are in the **robotics frame**:
      - Origin at base centre (ground level)
      - Z pointing UP
      - X pointing to the RIGHT (arm extension direction)

    Angles can be supplied in **degrees** (matching p1.py servo angles)
    or in radians.  By default the class maps the servo range [0°–180°]
    to the XACRO joint limits automatically.

    Methods
    -------
    compute(base, shoulder, elbow, yaw, roll, degrees=True)
        Returns (end_effector_xyz_mm, full_4x4_transform)

    compute_all_frames(base, shoulder, elbow, yaw, roll, degrees=True)
        Returns list of (frame_name, xyz_mm) for every joint frame,
        useful for 3D visualisation.
    """

    def __init__(self):
        self.origins = JOINT_ORIGINS_M
        self.axes    = JOINT_AXES
        self.limits  = JOINT_LIMITS_RAD
        self.link_lengths_mm = LINK_LENGTHS_MM

    # ── angle conversion ───────────────────────────────────────
    @staticmethod
    def servo_to_rad(servo_deg, joint_idx):
        """
        Map p1.py servo angle (0-180°) to the XACRO joint angle (radians).

        The servo "rest" at 90° corresponds to the XACRO zero (home) pose.
        So:  joint_rad = (servo_deg - 90) * π / 180

        For joints with reversed axis (-X or -Z), the sign is already
        encoded in the axis vector, so we keep the same mapping.
        """
        return math.radians(servo_deg - 90.0)

    # ── core FK (DH-style chain in Z-up frame) ─────────────────
    def compute(self, base=90, shoulder=90, elbow=90, yaw=90, roll=90,
                degrees=True):
        """
        Compute the end-effector position in the robotics frame.

        The robotics frame has:
          - Origin at base centre (ground level)
          - Z axis pointing UP
          - X axis pointing to the RIGHT

        Parameters
        ----------
        base, shoulder, elbow, yaw, roll : float
            Joint angles.  If degrees=True these are servo degrees (0-180)
            centred at 90°.  Otherwise they are raw radians.

        Returns
        -------
        position_mm : ndarray (3,)   — [x, y, z] in millimetres
        T           : ndarray (4,4)  — full homogeneous transform
        """
        if degrees:
            angles_rad = [self.servo_to_rad(a, i)
                          for i, a in enumerate([base, shoulder, elbow, yaw, roll])]
        else:
            angles_rad = [base, shoulder, elbow, yaw, roll]

        T = np.eye(4)
        for i in range(5):
            # Translation to joint i (in robotics frame)
            T_trans = _homogeneous(np.eye(3), self.origins[i])
            # Rotation about joint axis (in robotics frame)
            R = _rotation_matrix(self.axes[i], angles_rad[i])
            T_rot = _homogeneous(R, np.zeros(3))
            # Chain: T = T_prev * T_translate * T_rotate
            T = T @ T_trans @ T_rot

        position_m  = T[:3, 3]
        position_mm = position_m * 1000.0
        return position_mm, T

    def compute_all_frames(self, base=90, shoulder=90, elbow=90,
                           yaw=90, roll=90, degrees=True):
        """
        Compute positions of every joint frame (for 3D plotting).

        All positions in the robotics frame (Z-up, X-right).

        Returns
        -------
        frames : list of (name: str, xyz_mm: ndarray(3,))
        """
        if degrees:
            angles_rad = [self.servo_to_rad(a, i)
                          for i, a in enumerate([base, shoulder, elbow, yaw, roll])]
        else:
            angles_rad = [base, shoulder, elbow, yaw, roll]

        frame_names = ["base_link", "link_1_1", "link_2_1",
                        "link_3_1", "link_4_1", "end_effector_1"]

        frames = [("base_link", np.array([0.0, 0.0, 0.0]))]
        T = np.eye(4)

        for i in range(5):
            T_trans = _homogeneous(np.eye(3), self.origins[i])
            R = _rotation_matrix(self.axes[i], angles_rad[i])
            T_rot = _homogeneous(R, np.zeros(3))
            T = T @ T_trans @ T_rot
            pos_mm = T[:3, 3] * 1000.0
            frames.append((frame_names[i + 1], pos_mm.copy()))

        return frames

    def workspace_sample(self, n_samples=5000):
        """
        Monte-Carlo sample of reachable end-effector positions.

        Returns ndarray (N, 3) of [x, y, z] in mm (robotics frame).
        """
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n_samples):
            angles = [rng.uniform(0, 180) for _ in range(5)]
            pos, _ = self.compute(*angles)
            points.append(pos)
        return np.array(points)

    def print_summary(self):
        """Print a human-readable kinematic summary."""
        print("=" * 65)
        print("  FORWARD KINEMATICS — 5-DOF Robotic Manipulator")
        print("  Frame: Z-up, X-right (DH Convention)")
        print("=" * 65)
        print()
        print("  Coordinate Frame")
        print("  ────────────────")
        print("    Origin : Base centre at ground level")
        print("    Z-axis : UP   (vertical)")
        print("    X-axis : RIGHT (arm extension direction)")
        print("    Y-axis : follows right-hand rule")
        print()
        print("  Link Lengths (from XACRO joint origins)")
        print("  ────────────────────────────────────────")
        names = [
            "base_link  → link_1_1  (J1)",
            "link_1_1   → link_2_1  (J2)",
            "link_2_1   → link_3_1  (J3)",
            "link_3_1   → link_4_1  (J4)",
            "link_4_1   → end_eff   (J5)",
        ]
        total = 0.0
        for i, (name, length) in enumerate(zip(names, self.link_lengths_mm)):
            axis_idx = np.argmax(np.abs(self.axes[i]))
            axis_str = f"{'XYZ'[axis_idx]}"
            sign = "+" if self.axes[i][axis_idx] > 0 else "-"
            lo, hi = self.limits[i]
            lo_d = f"{math.degrees(lo):.1f}" if np.isfinite(lo) else "-∞"
            hi_d = f"{math.degrees(hi):.1f}" if np.isfinite(hi) else "+∞"
            print(f"  {name}  {length:7.1f} mm   axis {sign}{axis_str}"
                  f"   [{lo_d}° .. {hi_d}°]")
            total += length
        print(f"  {'':40s}  ───────")
        print(f"  {'Total reach (sum)':40s}  {total:.1f} mm")
        print()

        # Remapped joint origins
        print("  Joint Origins (robotics frame, mm)")
        print("  ──────────────────────────────────")
        jnames = ["J1", "J2", "J3", "J4", "J5"]
        for jn, origin in zip(jnames, self.origins):
            o_mm = origin * 1000
            print(f"    {jn}:  X={o_mm[0]:+8.2f}  Y={o_mm[1]:+8.2f}"
                  f"  Z={o_mm[2]:+8.2f} mm")
        print()

        # Rest pose FK
        pos, _ = self.compute()   # all servos at 90°
        print(f"  Rest-pose end-effector (all 90°):")
        print(f"    x = {pos[0]:+8.2f} mm  (horizontal, right)")
        print(f"    y = {pos[1]:+8.2f} mm  (horizontal, forward)")
        print(f"    z = {pos[2]:+8.2f} mm  (vertical, UP)")
        print(f"    |r| = {np.linalg.norm(pos):.2f} mm from base")
        print()


# ════════════════════════════════════════════════════════════════
#  3D VISUALISATION  (standalone mode)
# ════════════════════════════════════════════════════════════════

def plot_arm(fk, angles_deg=None, show_workspace=False):
    """Interactive 3D plot with optional workspace cloud."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#1e1e2e')

    ax = fig.add_subplot(111, projection='3d')

    # ── workspace point cloud (optional) ──
    if show_workspace:
        print("[INFO] Sampling workspace (this may take a moment)...")
        ws = fk.workspace_sample(8000)
        ax.scatter(ws[:, 0], ws[:, 1], ws[:, 2],
                   c=ws[:, 2], cmap='plasma', s=0.4, alpha=0.15)

    # ── slider setup ──
    init = angles_deg if angles_deg else [90, 90, 90, 90, 90]
    slider_labels = ["Base (J1)", "Shoulder (J2)", "Elbow (J3)",
                     "Yaw (J4)", "Roll (J5)"]
    slider_axes = []
    sliders = []
    for i in range(5):
        sax = fig.add_axes([0.15, 0.02 + i * 0.035, 0.55, 0.025])
        sax.set_facecolor('#2d2d44')
        s = Slider(sax, slider_labels[4 - i], 0, 180,
                   valinit=init[4 - i], valstep=1,
                   color='#89b4fa')
        slider_axes.append(sax)
        sliders.append(s)
    sliders.reverse()          # so index 0 = base
    slider_axes.reverse()

    # ── info text ──
    info_text = fig.text(0.78, 0.10, "", fontsize=9, color='#cdd6f4',
                         fontfamily='monospace',
                         bbox=dict(facecolor='#313244', edgecolor='#585b70',
                                   boxstyle='round,pad=0.5', alpha=0.9))

    def update(val=None):
        ax.cla()

        angles = [s.val for s in sliders]
        frames = fk.compute_all_frames(*angles)

        xs = [f[1][0] for f in frames]
        ys = [f[1][1] for f in frames]
        zs = [f[1][2] for f in frames]

        # draw links
        ax.plot(xs, ys, zs, '-o', color='#89b4fa', linewidth=3,
                markersize=8, markerfacecolor='#f38ba8',
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=5)

        # label each joint
        for name, pos in frames:
            short = name.replace("link_", "L").replace("_1", "")
            ax.text(pos[0], pos[1], pos[2] + 8, short,
                    fontsize=7, color='#a6e3a1', ha='center')

        # end-effector marker
        ee = frames[-1][1]
        ax.scatter(*ee, color='#f9e2af', s=120, marker='*',
                   edgecolors='white', linewidths=0.8, zorder=10)

        # base frame axes (Z=UP, X=RIGHT, Y=FORWARD)
        axis_len = 40
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='#f38ba8',
                  arrow_length_ratio=0.15, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='#a6e3a1',
                  arrow_length_ratio=0.15, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='#89b4fa',
                  arrow_length_ratio=0.15, linewidth=1.5)
        ax.text(axis_len + 5, 0, 0, "X (Right)", color='#f38ba8', fontsize=8)
        ax.text(0, axis_len + 5, 0, "Y (Fwd)", color='#a6e3a1', fontsize=8)
        ax.text(0, 0, axis_len + 5, "Z (Up)", color='#89b4fa', fontsize=8)

        # Draw ground plane reference
        ground_size = 80
        gx = np.array([-ground_size, ground_size, ground_size, -ground_size, -ground_size])
        gy = np.array([-ground_size, -ground_size, ground_size, ground_size, -ground_size])
        gz = np.zeros_like(gx)
        ax.plot(gx, gy, gz, color='#585b70', linewidth=0.8, alpha=0.5)

        # styling
        ax.set_xlabel("X — Right (mm)", color='#cdd6f4', fontsize=9)
        ax.set_ylabel("Y — Forward (mm)", color='#cdd6f4', fontsize=9)
        ax.set_zlabel("Z — Up (mm)", color='#cdd6f4', fontsize=9)
        ax.set_title("5-DOF Robotic Arm — Forward Kinematics (DH, Z-Up)",
                      color='#cdd6f4', fontsize=13, fontweight='bold', pad=15)

        ax.set_facecolor('#1e1e2e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.tick_params(colors='#6c7086', labelsize=7)
        ax.grid(True, alpha=0.2, color='#585b70')

        # auto-scale with some padding
        all_coords = np.array([xs, ys, zs])
        span = max(np.ptp(all_coords, axis=1).max(), 100)
        mid = all_coords.mean(axis=1)
        half = span * 0.7
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)

        # info panel
        dist = np.linalg.norm(ee)
        info_text.set_text(
            f"End-Effector (Z-up)\n"
            f"  X: {ee[0]:+8.2f} mm\n"
            f"  Y: {ee[1]:+8.2f} mm\n"
            f"  Z: {ee[2]:+8.2f} mm\n"
            f"  |r|: {dist:.1f} mm\n"
            f"\nLink Lengths\n"
            + "\n".join(f"  L{i+1}: {l:.1f} mm"
                        for i, l in enumerate(fk.link_lengths_mm))
            + f"\n  Total: {sum(fk.link_lengths_mm):.1f} mm"
        )

        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    update()

    plt.subplots_adjust(bottom=0.22, right=0.75)
    plt.show()


# ════════════════════════════════════════════════════════════════
#  MAIN  — run standalone for testing / visualisation
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fk = RoboticArmFK()

    # Print kinematic summary
    fk.print_summary()

    # Show a few example poses
    print("  Example Poses (Z-up, X-right frame)")
    print("  ────────────────────────────────────")
    test_poses = [
        ("All 90° (rest)",     [90, 90, 90, 90, 90]),
        ("Base swept to 0°",   [ 0, 90, 90, 90, 90]),
        ("Base swept to 180°", [180, 90, 90, 90, 90]),
        ("Shoulder up (45°)",  [90, 45, 90, 90, 90]),
        ("Elbow bent (135°)",  [90, 90, 135, 90, 90]),
        ("Full extend attempt",[90, 135, 45, 135, 90]),
    ]
    for label, angles in test_poses:
        pos, _ = fk.compute(*angles)
        dist = np.linalg.norm(pos)
        print(f"  {label:25s}  →  x={pos[0]:+8.2f}  y={pos[1]:+8.2f}"
              f"  z={pos[2]:+8.2f}  |r|={dist:.1f} mm")
    print()

    # Verify: base sweep should keep Z constant
    print("  Verification: Base sweep (Z should stay constant)")
    print("  ─────────────────────────────────────────────────")
    for base_deg in [0, 45, 90, 135, 180]:
        pos, _ = fk.compute(base=base_deg)
        print(f"    Base={base_deg:3d}°  →  x={pos[0]:+8.2f}  y={pos[1]:+8.2f}"
              f"  z={pos[2]:+8.2f} mm")
    print()

    # Launch interactive 3D viewer
    print("[INFO] Launching 3D visualiser with sliders…")
    print("[INFO] Frame: Z=Up, X=Right, Y=Forward")
    print("[INFO] Drag the sliders to move each joint (0°-180°).")
    print("[INFO] Close the window to exit.\n")
    plot_arm(fk, show_workspace=False)
