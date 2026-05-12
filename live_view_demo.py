#!/usr/bin/env python3
"""
liveview_overlay.py
-------------------
Combines the MOAD liveview monitor with the PyBullet scene overlay.

Reads live DSLR frames from the Canon EDSDK C++ backend (via the lockfile-safe
evf_live.jpg protocol), and composites a cached PyBullet render on top.

The render is only recomputed when the active camera or turntable angle changes.
Between re-renders, the latest live frame is read every loop and the cached
RGBA is composited onto it in real time.

Intrinsics are scaled from the calibration resolution to match the live view
image size, with an aspect ratio check to catch misconfiguration early.

Controls:
    Q / ESC         quit
    , / LEFT        previous camera
    . / RIGHT       next camera
    1-9             jump to camera by index
    A / D           turntable Z-rotation +/- large step
    W / S           turntable Z-rotation +/- small step
    I / K           global Z offset +/-
    J / L           global X offset +/-
    U / O           global Y offset +/-
    SPACE           reset turntable + offset
    R               force re-render

Usage:
    python3 liveview_overlay.py
"""

import os
import sys
import json
import time
import math
import numpy as np
import cv2
from pathlib import Path

try:
    from scene_replica_notag import TaglessSceneReplica, Rt_to_T, T_inv
    import pybullet as p
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    raise SystemExit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

LIVEVIEW_ROOT   = "/home/csrobot/moad_control/moad_cui/live_view_filestream"
LIVE_FILENAME   = "evf_live.jpg"
LOCK_FILENAME   = "evf.lock"

CALIBRATION     = "test_55mm"
SCRIPT_DIR      = Path(__file__).parent.resolve()
CAM_PARAMS_PATH = SCRIPT_DIR / f"../moad_cui/calibration/{CALIBRATION}/cam_parameters.json"
SCENE_CONFIG    = SCRIPT_DIR / "config/scene_cfg_notag_55mm.json"

WINDOW_NAME         = "MOAD Liveview Overlay"
READ_TIMEOUT        = 0.15      # seconds to wait for lock to clear
LOCK_RETRY          = 0.005     # seconds between lock-check retries
POLL_MS             = 30        # cv2.waitKey interval in ms
LIVE_STALE_S        = 2.0       # seconds before a feed is considered dead
ASPECT_TOL          = 0.01      # tolerance for aspect ratio mismatch check

OFFSET_STEP         = 0.005     # metres per keypress
ROTATION_STEP_LARGE = 10.0      # degrees per A/D keypress
ROTATION_STEP_SMALL = 5.0       # degrees per W/S keypress

CV_TO_GL = np.diag([1.0, 1.0, 1.0, 1.0])


# =============================================================================
# CAMERA NAME MAPPING
# =============================================================================

def folder_name_to_cam_key(folder_name: str) -> str:
    """
    Convert a liveview folder name to a cam_parameters.json key.
    e.g. "Camera 3"  →  "cam3"
         "camera3"   →  "cam3"
    """
    import re
    digits = re.search(r'\d+', folder_name)
    if not digits:
        raise ValueError(f"Cannot extract camera index from folder name: '{folder_name}'")
    return f"cam{digits.group()}"


def cam_key_to_folder_name(cam_key: str, cameras_lv: list[dict]) -> dict | None:
    """
    Find the liveview camera dict whose folder maps to the given cam key.
    Returns the dict or None if not found.
    """
    for cam in cameras_lv:
        if folder_name_to_cam_key(cam["name"]) == cam_key:
            return cam
    return None


# =============================================================================
# CAMERA DISCOVERY
# =============================================================================

def discover_cameras(root: str) -> list[dict]:
    """
    Scan the liveview root for active camera folders.
    A folder is active if evf_live.jpg exists and was written within LIVE_STALE_S.
    """
    cameras = []
    if not os.path.isdir(root):
        print(f"[ERROR] Liveview root not found: {root}")
        return cameras

    entries = sorted(
        [e for e in os.scandir(root) if e.is_dir()],
        key=lambda e: e.name
    )

    for entry in entries:
        live_path = os.path.join(entry.path, LIVE_FILENAME)
        lock_path = os.path.join(entry.path, LOCK_FILENAME)

        if not os.path.exists(live_path):
            print(f"  [SKIP]  {entry.name} — evf_live.jpg not found")
            continue

        age = time.time() - os.stat(live_path).st_mtime
        if age > LIVE_STALE_S:
            print(f"  [SKIP]  {entry.name} — last frame {age:.1f}s ago (feed inactive)")
            continue

        try:
            cam_key = folder_name_to_cam_key(entry.name)
        except ValueError as e:
            print(f"  [SKIP]  {entry.name} — {e}")
            continue

        cameras.append({
            "name":      entry.name,
            "cam_key":   cam_key,
            "live_path": live_path,
            "lock_path": lock_path,
        })
        print(f"  [LIVE]  {entry.name}  →  key='{cam_key}'  (last frame {age:.2f}s ago)")

    return cameras


# =============================================================================
# SAFE FRAME READER
# =============================================================================

def read_latest_frame(live_path: str, lock_path: str) -> np.ndarray | None:
    deadline = time.monotonic() + READ_TIMEOUT
    while time.monotonic() < deadline:
        if os.path.exists(lock_path):
            time.sleep(LOCK_RETRY)
            continue
        if not os.path.exists(live_path):
            time.sleep(LOCK_RETRY)
            continue
        frame = cv2.imread(live_path)
        if frame is not None:
            return frame
        time.sleep(LOCK_RETRY)
    return None


# =============================================================================
# CALIBRATION LOADING
# =============================================================================

def load_cam_parameters(path: Path) -> tuple[dict, dict, dict]:
    with open(path, "r") as f:
        data = json.load(f)

    intrinsics = data["intrinsics"]
    scale_info = data["_info"]["scaling"]
    cameras    = {}
    for cam_name, cam_data in data["cameras"].items():
        cameras[cam_name] = {
            "c2w": np.array(cam_data["extrinsics"]["c2w"], dtype=np.float64),
            "w2c": np.array(cam_data["extrinsics"]["w2c"], dtype=np.float64),
        }

    print(f"  Loaded {len(cameras)} cameras from calibration: {sorted(cameras.keys())}")
    return cameras, intrinsics, scale_info


def check_aspect_ratio(intrinsics: dict, live_frame: np.ndarray) -> float:
    """
    Verify the live frame aspect ratio matches the calibration intrinsics.
    Returns the scale factor (live_width / calib_width).
    Raises ValueError if the aspect ratios diverge beyond ASPECT_TOL.
    """
    calib_w  = intrinsics["width"]
    calib_h  = intrinsics["height"]
    live_h, live_w = live_frame.shape[:2]

    calib_ar = calib_w / calib_h
    live_ar  = live_w  / live_h

    if abs(calib_ar - live_ar) > ASPECT_TOL:
        raise ValueError(
            f"Aspect ratio mismatch: calibration={calib_w}x{calib_h} ({calib_ar:.4f}) "
            f"vs live frame={live_w}x{live_h} ({live_ar:.4f}). "
            f"Check your liveview downscale settings."
        )

    scale = live_w / calib_w
    print(f"  Aspect ratio OK ({calib_ar:.4f}). "
          f"Intrinsic scale: {calib_w}x{calib_h} → {live_w}x{live_h}  (×{scale:.4f})")
    return scale


def scale_intrinsics(intrinsics: dict, scale: float) -> dict:
    return {
        **intrinsics,
        "fx":     intrinsics["fx"] * scale,
        "fy":     intrinsics["fy"] * scale,
        "cx":     intrinsics["cx"] * scale,
        "cy":     intrinsics["cy"] * scale,
        "width":  int(round(intrinsics["width"]  * scale)),
        "height": int(round(intrinsics["height"] * scale)),
    }


def build_cam_K(intrinsics: dict) -> np.ndarray:
    return np.array([
        [intrinsics["fx"],             0.0, intrinsics["cx"]],
        [            0.0, intrinsics["fy"], intrinsics["cy"]],
        [            0.0,             0.0,             1.0],
    ], dtype=np.float64)


# =============================================================================
# CAMERA POSE
# =============================================================================

def make_z_rotation(degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4)
    R[0, 0] =  c;  R[0, 1] = -s
    R[1, 0] =  s;  R[1, 1] =  c
    return R


def get_camera_pose_from_extrinsics(
    cam_key:       str,
    cameras:       dict,
    turntable_deg: float,
    global_offset: np.ndarray,
    scale:         float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    T_w2c = cameras[cam_key]["w2c"].copy()
    T_w2c[:3, 3] *= scale

    T_w2c_cv = CV_TO_GL @ T_w2c

    T_scene = make_z_rotation(turntable_deg)
    T_scene[:3, 3] = global_offset * scale

    T_final = T_w2c_cv @ T_scene
    return T_final[:3, :3], T_final[:3, 3]


# =============================================================================
# COMPOSITING
# =============================================================================

def overlay_rgba_on_bgr(background: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    bg      = background.astype(np.float32)
    ov      = overlay_rgba.astype(np.float32)
    ov_bgra = ov[:, :, [2, 1, 0, 3]]
    alpha   = ov_bgra[:, :, 3:4] / 255.0
    out     = ov_bgra[:, :, :3] * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# HUD
# =============================================================================

def draw_hud(
    frame:         np.ndarray,
    cam_index:     int,
    cameras_lv:    list[dict],
    turntable_deg: float,
    global_offset: np.ndarray,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cam  = cameras_lv[cam_index]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    lines = [
        f"  {cam['name']}  [{cam_index + 1}/{len(cameras_lv)}]   "
        f"rot={turntable_deg:+.1f}deg   "
        f"offset=[{global_offset[0]:+.3f}, {global_offset[1]:+.3f}, {global_offset[2]:+.3f}] m"#,
        # f"  A/D: rot {ROTATION_STEP_LARGE:.0f}deg   W/S: {ROTATION_STEP_SMALL:.0f}deg   "
        # f"IJKL/UO: offset   SPACE: reset   R: re-render   Q/ESC: quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (0, 18 + i * 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1, cv2.LINE_AA)

    # Camera index dots
    dot_r  = 5
    dot_y  = 42
    dot_x0 = w - (len(cameras_lv) * (dot_r * 2 + 6)) - 12
    for i in range(len(cameras_lv)):
        cx     = dot_x0 + i * (dot_r * 2 + 6) + dot_r
        color  = (80, 200, 80) if i == cam_index else (55, 55, 55)
        cv2.circle(frame, (cx, dot_y), dot_r, color, -1)

    # Bottom bar — navigation hint
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 24), (w, h), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay2, 0.72, frame, 0.28, 0) 
    nav = "  , / L-Arr  prev camera        . / R-Arr  next camera        1-9  jump"
    cv2.putText(frame, nav, (0, h - 8),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (110, 110, 110), 1, cv2.LINE_AA)

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "─" * 60)
    print("  MOAD Liveview Overlay")
    print("─" * 60)

    # ── Discover live cameras ─────────────────────────────────────────────────
    print(f"\n  Scanning liveview root: {LIVEVIEW_ROOT}\n")
    cameras_lv = discover_cameras(LIVEVIEW_ROOT)
    if not cameras_lv:
        print("\n[ERROR] No active liveview feeds found. "
              "Is the C++ backend running with liveview started?\n")
        sys.exit(1)
    print(f"\n  {len(cameras_lv)} active feed(s) found.\n")

    # ── Load calibration ──────────────────────────────────────────────────────
    print(f"  Loading calibration: {CAM_PARAMS_PATH}\n")
    cameras_cal, intrinsics_full, scale_info = load_cam_parameters(CAM_PARAMS_PATH)

    # Verify all live cameras have a calibration entry
    for lv_cam in cameras_lv:
        key = lv_cam["cam_key"]
        if key not in cameras_cal:
            print(f"[ERROR] Live camera '{lv_cam['name']}' maps to key '{key}' "
                  f"which is not in cam_parameters.json. "
                  f"Available keys: {sorted(cameras_cal.keys())}")
            sys.exit(1)

    # ── Load scene config ─────────────────────────────────────────────────────
    with open(SCENE_CONFIG, "r") as f:
        scene_cfg = json.load(f)

    scene_scale   = scale_info["scale"]
    global_offset = np.asarray(scene_cfg["CALIBRATION_OFFSET"], dtype=np.float64)

    # ── Read one live frame to determine actual live resolution ───────────────
    print("  Reading a reference frame to determine live resolution...")
    current_idx   = 0
    ref_cam       = cameras_lv[current_idx]
    ref_frame     = None
    deadline      = time.monotonic() + 5.0
    while ref_frame is None and time.monotonic() < deadline:
        ref_frame = read_latest_frame(ref_cam["live_path"], ref_cam["lock_path"])
        if ref_frame is None:
            print("    Waiting for first frame...")
            time.sleep(0.2)
    if ref_frame is None:
        print("[ERROR] Could not read a reference frame within 5 seconds.")
        sys.exit(1)

    # ── Scale intrinsics to match live resolution ─────────────────────────────
    print(f"  Reference frame size: {ref_frame.shape[1]}×{ref_frame.shape[0]}")
    intr_scale   = check_aspect_ratio(intrinsics_full, ref_frame)

    # Apply scene render_scale on top of the live-frame scale
    combined_scale = intr_scale# * scene_cfg.get("render_scale", 1.0)
    render_scale   = scene_cfg.get("render_scale", 1.0)
    intrinsics     = scale_intrinsics(intrinsics_full, combined_scale)
    live_w         = int(ref_frame.shape[1] * render_scale)
    live_h         = int(ref_frame.shape[0] * render_scale)
    print(f"  Render resolution: {intrinsics['width']}×{intrinsics['height']}")

    # ── Initialise PyBullet scene ─────────────────────────────────────────────
    print("\n  Initialising PyBullet scene...")
    cam_K       = build_cam_K(intrinsics)
    initial_key = cameras_lv[current_idx]["cam_key"]
    R_init, t_init = get_camera_pose_from_extrinsics(
        initial_key, cameras_cal, 0.0, global_offset, scene_scale
    )
    scene = TaglessSceneReplica(
        cam_K        = cam_K,
        W            = intrinsics["width"],
        H            = intrinsics["height"],
        R_cw_cv      = R_init,
        t_cw_cv      = t_init,
        scene_config = scene_cfg,
    )
    scene.load_scene(scene_cfg["scene_file"])
    print("  Scene loaded.\n")

    # ── OpenCV window ─────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, live_w, live_h)

    # ── State ─────────────────────────────────────────────────────────────────
    turntable_deg  = 0.0
    needs_render   = True
    rgba_cache     = None
    last_good_frame = ref_frame

    print("  Running. Press Q or ESC to quit.\n")

    while True:
        lv_cam  = cameras_lv[current_idx]
        cam_key = lv_cam["cam_key"]

        # ── Re-render only when state changed ────────────────────────────────
        if needs_render:
            R_cw_cv, t_cw_cv = get_camera_pose_from_extrinsics(
                cam_key, cameras_cal, turntable_deg, global_offset, scene_scale
            )
            print(f"  [render] cam={cam_key}  rot={turntable_deg:+.1f}°  "
                  f"offset={np.round(global_offset, 3)}")
            rgba_cache   = scene.update_and_render(R_cw_cv, t_cw_cv)
            needs_render = False

        # ── Read latest live frame ────────────────────────────────────────────
        frame = read_latest_frame(lv_cam["live_path"], lv_cam["lock_path"])
        if frame is None:
            frame = last_good_frame.copy()
            cv2.putText(frame, "Waiting for frame...", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (40, 40, 200), 2)
        else:
            # Resize live frame to render resolution if needed
            if frame.shape[1] != intrinsics["width"] or frame.shape[0] != intrinsics["height"]:
                frame = cv2.resize(
                    frame,
                    (intrinsics["width"], intrinsics["height"]),
                    interpolation=cv2.INTER_LINEAR
                )
            last_good_frame = frame.copy()

        # ── Composite PyBullet render onto live frame ─────────────────────────
        if rgba_cache is not None:
            display = np.ascontiguousarray(overlay_rgba_on_bgr(frame, rgba_cache))
        else:
            display = frame.copy()

        draw_hud(display, current_idx, cameras_lv, turntable_deg, global_offset)
        if render_scale != 1.0: 
            outsize = (int(intrinsics["width"]*render_scale), int(intrinsics["height"]*render_scale))
            print(outsize)
            finaldisplay = cv2.resize(
                    display,
                    outsize,
                    interpolation=cv2.INTER_LINEAR
            )
        else:
            finaldisplay = display
        cv2.imshow(WINDOW_NAME, finaldisplay)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(POLL_MS) & 0xFF

        if key in (ord('q'), 27):                            # Q / ESC — quit
            break
        elif key in (ord(','), 81):                          # , / LEFT — prev
            current_idx   = (current_idx - 1) % len(cameras_lv)
            last_good_frame = None if last_good_frame is None else last_good_frame
            needs_render  = True
            print(f"  → {cameras_lv[current_idx]['name']}")
        elif key in (ord('.'), 83):                          # . / RIGHT — next
            current_idx   = (current_idx + 1) % len(cameras_lv)
            needs_render  = True
            print(f"  → {cameras_lv[current_idx]['name']}")
        elif ord('1') <= key <= ord('9'):                    # number jump
            idx = key - ord('1')
            if idx < len(cameras_lv):
                current_idx  = idx
                needs_render = True
                print(f"  → {cameras_lv[current_idx]['name']}")
        elif key == ord('d'):
            turntable_deg += ROTATION_STEP_LARGE;  needs_render = True
        elif key == ord('a'):
            turntable_deg -= ROTATION_STEP_LARGE;  needs_render = True
        elif key == ord('w'):
            turntable_deg += ROTATION_STEP_SMALL;  needs_render = True
        elif key == ord('s'):
            turntable_deg -= ROTATION_STEP_SMALL;  needs_render = True
        elif key == ord('l'):
            global_offset[0] += OFFSET_STEP;       needs_render = True
        elif key == ord('j'):
            global_offset[0] -= OFFSET_STEP;       needs_render = True
        elif key == ord('o'):
            global_offset[1] += OFFSET_STEP;       needs_render = True
        elif key == ord('u'):
            global_offset[1] -= OFFSET_STEP;       needs_render = True
        elif key == ord('i'):
            global_offset[2] += OFFSET_STEP;       needs_render = True
        elif key == ord('k'):
            global_offset[2] -= OFFSET_STEP;       needs_render = True
        elif key == ord(' '):
            turntable_deg  = 0.0
            global_offset  = np.asarray(scene_cfg["CALIBRATION_OFFSET"], dtype=np.float64)
            needs_render   = True
        elif key == ord('r'):
            needs_render   = True

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    try:
        p.disconnect(scene.pb)
    except Exception:
        pass
    print("\n  Overlay closed.\n")


if __name__ == "__main__":
    main()