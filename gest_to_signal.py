#!/usr/bin/env python3
"""
Gesture-Controlled 6-DOF Robotic Manipulator
=============================================
LEFT HAND  -> Joint Selection (which joint to move)
RIGHT HAND -> Movement Control + Gripper + Rest Position

Hardware: 6x MG90S servos, ESP32 controller, 5V 10A PSU
Speed: 60 deg/s (configurable)
"""

import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import os
import numpy as np
from forward_kinematics import RoboticArmFK

# Forward Kinematics engine (uses link lengths from XACRO)
fk = RoboticArmFK()

# ================================================================
#  CONDITION 1: EDIT MIN / MAX / REST ANGLES FOR EACH JOINT HERE
# ================================================================
JOINT_CONFIG = {
    "base":     {"min": 0,   "max": 180, "rest": 90},
    "shoulder": {"min": 0,   "max": 180, "rest": 90},
    "elbow":    {"min": 0,   "max": 180, "rest": 90},
    "yaw":      {"min": 0,   "max": 180, "rest": 90},   # end-effector yaw
    "roll":     {"min": 0,   "max": 180, "rest": 90},   # end-effector roll
    "jaws":     {"min": 10,  "max": 120, "rest": 40},   # gripper
}

# Condition 5: 60 deg/s  (MG90S max ~600 deg/s, we use 1/10)
MOVEMENT_SPEED_DPS = 60

SERIAL_BAUDRATE = 115200
GESTURE_CONF_THRESH = 0.35        # lowered for faster recognition
DEBOUNCE_FRAMES = 1               # minimal debounce for instant response
SERIAL_SEND_INTERVAL = 0.05       # send at most every 50ms (20Hz)
CAMERA_WIDTH = 640                 # lower resolution = faster processing
CAMERA_HEIGHT = 480

# ================================================================
#  GESTURE MAP
# ================================================================
# LEFT HAND  -> selects active joint
#   Pointing_Up  -> base
#   Victory      -> shoulder
#   ILoveYou     -> elbow
#   Thumb_Up     -> yaw
#   Thumb_Down   -> roll
#
# RIGHT HAND -> controls movement
#   Thumb_Up     -> increment selected joint
#   Thumb_Down   -> decrement selected joint
#   Open_Palm    -> open jaws       (Condition 3)
#   Closed_Fist  -> close jaws      (Condition 3)
#   ILoveYou     -> return to rest  (Condition 4)

LEFT_JOINT_MAP = {
    "Pointing_Up": "base",
    "Victory":     "shoulder",
    "ILoveYou":    "elbow",
    "Thumb_Up":    "yaw",
    "Thumb_Down":  "roll",
}

# Hand skeleton connections (21 landmarks)
HAND_CONNS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

FINGERTIP_IDS = {4:"THB", 8:"IDX", 12:"MID", 16:"RNG", 20:"PNK"}

# ================================================================
#  AUTO-DETECT SERIAL PORT
# ================================================================
def find_serial_port():
    """Scan for any USB-serial adapter on Linux."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        info = f"{p.description} {p.hwid}".lower()
        if any(k in info for k in ['usb','ch340','cp210','ftdi','pl2303','acm','esp','arduino']):
            return p.device
    for path in ['/dev/ttyUSB0','/dev/ttyACM0','/dev/ttyUSB1','/dev/ttyACM1']:
        if os.path.exists(path):
            return path
    return None

port_path = find_serial_port()
arduino = None
if port_path:
    # Retry connection up to 3 times (ESP32 may need a moment after reset)
    for attempt in range(3):
        try:
            arduino = serial.Serial(port=port_path, baudrate=SERIAL_BAUDRATE, timeout=0.1)
            time.sleep(2)  # Wait for ESP32 bootloader to finish
            # Flush any bootloader garbage
            arduino.reset_input_buffer()
            arduino.reset_output_buffer()
            print(f"[INFO] Serial connected on {port_path}")
            # Read and print any ESP32 startup messages
            while arduino.in_waiting:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[ESP32] {line}")
            break
        except Exception as e:
            print(f"[WARN] Serial attempt {attempt+1}/3 failed on {port_path}: {e}")
            arduino = None
            if attempt < 2:
                time.sleep(1)
    if arduino is None:
        print("[WARN] All serial attempts failed. Running in simulation mode.")
        print("[HINT] Make sure Arduino IDE Serial Monitor is CLOSED.")
        print("[HINT] Try unplugging and re-plugging the USB cable.")
else:
    print("[WARN] No USB-serial port found. Running in simulation mode.")

# ================================================================
#  STATE
# ================================================================
JOINT_NAMES = ["base", "shoulder", "elbow", "yaw", "roll"]
current_angles = {name: JOINT_CONFIG[name]["rest"] for name in JOINT_NAMES}
current_angles["jaws"] = JOINT_CONFIG["jaws"]["rest"]

active_joint = "base"
prev_time = time.time()
last_send_time = 0.0
last_sent_angles = None

# Debounce buffers per hand
left_buf = []
right_buf = []

def send_angles(angles, force=False):
    """Condition 2: always send angles as output (rate-limited)."""
    global last_send_time, last_sent_angles
    now = time.time()
    # Rate-limit serial writes to avoid flooding
    if not force and (now - last_send_time) < SERIAL_SEND_INTERVAL:
        return
    # Skip if angles haven't changed
    angle_tuple = tuple(round(angles[k], 1) for k in ["base","shoulder","elbow","yaw","roll","jaws"])
    if not force and last_sent_angles == angle_tuple:
        return
    # Format with exactly 1 decimal place to keep line short (ESP32 buffer is 64 chars)
    cmd = (f"{angles['base']:.1f},{angles['shoulder']:.1f},{angles['elbow']:.1f},"
           f"{angles['yaw']:.1f},{angles['roll']:.1f},{angles['jaws']:.1f}\n")
    if arduino and arduino.is_open:
        try:
            arduino.write(cmd.encode('utf-8'))
            # Read any response from ESP32 (non-blocking)
            while arduino.in_waiting:
                resp = arduino.readline().decode('utf-8', errors='ignore').strip()
                if resp:
                    print(f"[ESP32] {resp}")
        except serial.SerialException as e:
            print(f"[ERR] Serial write failed: {e}")
    last_send_time = now
    last_sent_angles = angle_tuple

def clamp_joint(name, value):
    """Clamp to configurable min/max (Condition 1)."""
    cfg = JOINT_CONFIG[name]
    return max(cfg["min"], min(cfg["max"], value))

# ================================================================
#  DRAWING HELPERS
# ================================================================
def draw_hand(frame, landmarks, is_left_physical, h, w):
    """Draw skeleton, joints, fingertip labels, and bounding box."""
    pts = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts.append((cx, cy))

    jc = (255, 200, 50) if is_left_physical else (50, 255, 100)   # joint color
    bc = (255, 100, 50) if is_left_physical else (50, 200, 50)    # bone color
    label = "L" if is_left_physical else "R"

    # Draw bones
    for a, b in HAND_CONNS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], bc, 2, cv2.LINE_AA)

    # Draw joints
    for i, (cx, cy) in enumerate(pts):
        r = 6 if i in FINGERTIP_IDS else 3
        cv2.circle(frame, (cx, cy), r, jc, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r+1, (255,255,255), 1, cv2.LINE_AA)

    # Fingertip labels
    for idx, name in FINGERTIP_IDS.items():
        if idx < len(pts):
            cv2.putText(frame, name, (pts[idx][0]+8, pts[idx][1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

    # Bounding box
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 15
        x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
        x2, y2 = min(w, max(xs)+pad), min(h, max(ys)+pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{label} Hand", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, jc, 1, cv2.LINE_AA)

def draw_hud(frame, active, angles, left_g, right_g, speed, fps, h, w, ee_pos=None):
    """Draw telemetry overlay with joint angles and FK end-effector position."""
    # ── Top bar background ──
    top_roi = frame[0:110, 0:w].copy()
    cv2.rectangle(top_roi, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.addWeighted(top_roi, 0.6, frame[0:110, 0:w], 0.4, 0, frame[0:110, 0:w])

    # ── Right-side panel: Joint Angles + FK Coordinates ──
    panel_w = 210
    panel_x = w - panel_w
    panel_h = h - 110           # from below top bar to bottom
    panel_y = 110
    # Semi-transparent dark panel
    side_roi = frame[panel_y:panel_y + panel_h, panel_x:w].copy()
    cv2.rectangle(side_roi, (0, 0), (panel_w, panel_h), (15, 15, 30), -1)
    cv2.addWeighted(side_roi, 0.75, frame[panel_y:panel_y + panel_h, panel_x:w],
                    0.25, 0, frame[panel_y:panel_y + panel_h, panel_x:w])

    # ── Section 1: Joint Angles ──
    sx = panel_x + 10
    sy = panel_y + 22
    cv2.putText(frame, "JOINT ANGLES", (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    # Divider line
    cv2.line(frame, (sx, sy + 6), (w - 10, sy + 6), (80, 80, 120), 1, cv2.LINE_AA)

    joint_labels = {
        "base": "Base",
        "shoulder": "Shoulder",
        "elbow": "Elbow",
        "yaw": "Yaw",
        "roll": "Roll",
        "jaws": "Jaws",
    }
    bar_w = 120
    row_y = sy + 28
    for jn in JOINT_NAMES + ["jaws"]:
        cfg = JOINT_CONFIG[jn]
        is_active = (jn == active)
        label_color = (0, 255, 255) if is_active else (200, 200, 200)
        bar_color = (0, 255, 255) if is_active else (100, 140, 160)
        angle_val = int(angles[jn])

        # Joint name + angle value
        label = f"{joint_labels[jn]:>8s}: {angle_val:>3d} deg"
        cv2.putText(frame, label, (sx, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)

        # Progress bar below label
        bar_y = row_y + 5
        frac = (angles[jn] - cfg["min"]) / max(1, cfg["max"] - cfg["min"])
        cv2.rectangle(frame, (sx, bar_y), (sx + bar_w, bar_y + 6), (50, 50, 70), -1)
        cv2.rectangle(frame, (sx, bar_y), (sx + int(bar_w * frac), bar_y + 6),
                      bar_color, -1)

        # Active joint marker
        if is_active:
            cv2.circle(frame, (sx + bar_w + 12, row_y - 3), 4, (0, 255, 255), -1,
                       cv2.LINE_AA)

        row_y += 32

    # ── Section 2: FK End-Effector Coordinates ──
    if ee_pos is not None:
        dist = np.linalg.norm(ee_pos)
        fk_y = row_y + 8
        cv2.putText(frame, "END-EFFECTOR (FK)", (sx, fk_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 140, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (sx, fk_y + 6), (w - 10, fk_y + 6), (80, 80, 120), 1,
                 cv2.LINE_AA)

        fk_y += 24
        cv2.putText(frame, f"X: {ee_pos[0]:+8.1f} mm", (sx, fk_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 180, 255), 1, cv2.LINE_AA)
        fk_y += 20
        cv2.putText(frame, f"Y: {ee_pos[1]:+8.1f} mm", (sx, fk_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 255, 180), 1, cv2.LINE_AA)
        fk_y += 20
        cv2.putText(frame, f"Z: {ee_pos[2]:+8.1f} mm", (sx, fk_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 180, 100), 1, cv2.LINE_AA)
        fk_y += 20
        cv2.putText(frame, f"Dist: {dist:.1f} mm", (sx, fk_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Top-left info ──
    cv2.putText(frame, f"ACTIVE JOINT: {active.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Left: {left_g}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Right: {right_g}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Speed: {speed} deg/s | FPS: {fps:.0f}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def debounce(buf, gesture, n=DEBOUNCE_FRAMES):
    """Return stable gesture if last n entries match."""
    buf.append(gesture)
    if len(buf) > n:
        buf.pop(0)
    if len(buf) == n and all(g == buf[0] for g in buf):
        return buf[0]
    return "None"

# ================================================================
#  MEDIAPIPE SETUP (using both: Tasks API for gestures)
# ================================================================
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path='gesture_recognizer.task',
        delegate=BaseOptions.Delegate.CPU,          # explicit CPU (or GPU if available)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.40,   # lower = faster hand pickup
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40,         # lower = less re-detection overhead
)

# Send initial resting position (Condition 2)
print("[INFO] Sending resting position...")
send_angles(current_angles)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)                     # request 30fps from camera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                # minimize frame buffer latency
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit(1)

print("[INFO] Starting gesture control. Press 'q' to quit.")
print("[INFO] Show LEFT hand to select joint, RIGHT hand to control.")

# ================================================================
#  MAIN LOOP
# ================================================================
with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB once for MediaPipe (avoids internal conversion)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int(time.time() * 1000)

        result = recognizer.recognize_for_video(mp_image, ts_ms)

        # Time-based speed control
        now = time.time()
        dt = now - prev_time
        prev_time = now
        max_step = MOVEMENT_SPEED_DPS * dt  # max degrees this frame

        left_gesture = "None"
        right_gesture = "None"

        # Process each detected hand
        if result.gestures and result.handedness and result.hand_landmarks:
            for i in range(len(result.gestures)):
                gesture_name = result.gestures[i][0].category_name
                confidence = result.gestures[i][0].score
                # MediaPipe handedness label (in flipped image, labels are swapped)
                mp_label = result.handedness[i][0].category_name
                landmarks = result.hand_landmarks[i]

                if confidence < GESTURE_CONF_THRESH:
                    gesture_name = "None"

                # After cv2.flip: physical LEFT hand -> MediaPipe "Right"
                #                  physical RIGHT hand -> MediaPipe "Left"
                is_left_physical = (mp_label == "Right")

                # Draw hand skeleton
                draw_hand(frame, landmarks, is_left_physical, h, w)

                if is_left_physical:
                    left_gesture = gesture_name
                else:
                    right_gesture = gesture_name

        # Debounce
        stable_left = debounce(left_buf, left_gesture)
        stable_right = debounce(right_buf, right_gesture)

        # ---- LEFT HAND: Joint Selection ----
        if stable_left in LEFT_JOINT_MAP:
            new_joint = LEFT_JOINT_MAP[stable_left]
            if new_joint != active_joint:
                active_joint = new_joint
                print(f"[JOINT] Switched to: {active_joint.upper()}")

        # ---- RIGHT HAND: Movement Control ----
        if stable_right == "Thumb_Up":
            current_angles[active_joint] += max_step
            current_angles[active_joint] = clamp_joint(active_joint,
                                                       current_angles[active_joint])

        elif stable_right == "Thumb_Down":
            current_angles[active_joint] -= max_step
            current_angles[active_joint] = clamp_joint(active_joint,
                                                       current_angles[active_joint])

        # Condition 3: Gripper control (right hand, not joint-selected)
        elif stable_right == "Open_Palm":
            current_angles["jaws"] -= max_step
            current_angles["jaws"] = clamp_joint("jaws", current_angles["jaws"])

        elif stable_right == "Closed_Fist":
            current_angles["jaws"] += max_step
            current_angles["jaws"] = clamp_joint("jaws", current_angles["jaws"])

        # Condition 4: Return to resting position
        elif stable_right == "ILoveYou":
            for jn in JOINT_NAMES + ["jaws"]:
                rest = JOINT_CONFIG[jn]["rest"]
                diff = rest - current_angles[jn]
                if abs(diff) > 0.5:
                    step = min(max_step, abs(diff))
                    current_angles[jn] += step if diff > 0 else -step
                    current_angles[jn] = clamp_joint(jn, current_angles[jn])
            cv2.putText(frame, ">> RETURNING TO REST <<", (w//2-160, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2, cv2.LINE_AA)

        # Round angles for servo precision
        for jn in current_angles:
            current_angles[jn] = round(current_angles[jn], 1)

        # Condition 2: ALWAYS send angles (rate-limited to avoid bottleneck)
        send_angles(current_angles)

        # Calculate FPS
        fps = 1.0 / dt if dt > 0 else 0

        # Compute Forward Kinematics (end-effector position from current angles)
        ee_pos, _ = fk.compute(
            base=current_angles["base"],
            shoulder=current_angles["shoulder"],
            elbow=current_angles["elbow"],
            yaw=current_angles["yaw"],
            roll=current_angles["roll"]
        )

        # Draw HUD with FK end-effector readout
        draw_hud(frame, active_joint, current_angles,
                 stable_left, stable_right, MOVEMENT_SPEED_DPS, fps, h, w,
                 ee_pos=ee_pos)

        cv2.imshow('Robotic Arm - Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
print("[INFO] Shutdown complete.")