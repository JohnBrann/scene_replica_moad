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
    C               confirm scene replica — copy scene .npz + config.json
                    into <output_dir>/<object_name>/<current_pose>/scene_replica/

Usage:
    python3 liveview_overlay.py
"""

import os
import sys
import json
import time
import math
import argparse
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
# The values below that depend on user/rig setup (paths, calibration name,
# step sizes, fallback behaviour, etc.) have been moved to argparse arguments
# in main() — see ARG DEFAULTS there. The constants below are intentionally
# left as fixed module-level constants because they are not expected to change:
#   - SCRIPT_DIR     : derived from the file's own location
#   - CV_TO_GL       : a fixed coordinate convention matrix
#   - ASPECT_TOL     : a small numerical tolerance, not a tunable setting
#   - timing values  : internal polling/retry intervals
#   - WINDOW_NAME    : cosmetic only

SCRIPT_DIR = Path(__file__).parent.resolve()

WINDOW_NAME  = "MOAD Liveview Overlay"
READ_TIMEOUT = 0.15      # seconds to wait for lock to clear
LOCK_RETRY   = 0.005     # seconds between lock-check retries
POLL_MS      = 30        # cv2.waitKey interval in ms
LIVE_STALE_S = 2.0       # seconds before a feed is considered dead
ASPECT_TOL   = 0.01      # tolerance for aspect ratio mismatch check

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

def discover_cameras(root: str, live_filename: str, lock_filename: str) -> list[dict]:
    """
    Scan the liveview root for active camera folders.
    A folder is active if `live_filename` exists and was written within LIVE_STALE_S.
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
        live_path = os.path.join(entry.path, live_filename)
        lock_path = os.path.join(entry.path, lock_filename)

        if not os.path.exists(live_path):
            print(f"  [SKIP]  {entry.name} — {live_filename} not found")
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


def load_background_images(fallback_dir: str, position: int = 0, scale: float = 1.0) -> list[dict]:
    """
    Build a synthetic 'cameras_lv' list from static background images when no
    live feeds are available.

    Looks for files named "{cam#}_{position}_img.jpg" in `fallback_dir`,
    e.g. "cam3_000_img.jpg". One entry is created per camera found, sorted by
    camera number. Each entry has a "static_path" key (instead of
    "live_path"/"lock_path") pointing at the background image to use.

    Args:
        fallback_dir: directory containing the static background images
        position:     turntable position index to use for the filename
                       (zero-padded to 3 digits, matching the calibration
                       capture convention, e.g. cam2_000_img.jpg)

    Returns:
        List of camera dicts in the same shape as discover_cameras() output,
        but with "static_path" set and "live_path"/"lock_path" set to None.
        Empty list if the directory doesn't exist or no matching files found.
    """
    import re

    cameras = []
    if not os.path.isdir(fallback_dir):
        print(f"[ERROR] Fallback background image directory not found: {fallback_dir}")
        return cameras

    pos_str  = f"{position:03d}"
    pattern  = re.compile(rf"^cam(\d+)_{pos_str}_img\.jpg$", re.IGNORECASE)
    print(f"Searching \'{fallback_dir}\' for turntable position {pos_str}...")

    matches = []
    for entry in os.scandir(fallback_dir):
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            matches.append((int(m.group(1)), entry.name))

    matches.sort(key=lambda x: x[0])

    for cam_num, filename in matches:
        cam_key = f"cam{cam_num}"
        static_path = os.path.join(fallback_dir, filename)
        static_img = cv2.imread(static_path)
        h,w,c = static_img.shape
        static_img = cv2.resize(static_img,(int(w*scale),int(h*scale)))
        cameras.append({
            "name":        f"Camera {cam_num} (static)",
            "cam_key":     cam_key,
            "live_path":   None,
            "lock_path":   None,
            "static_image": static_img,
        })
        print(f"  [STATIC] {filename}  →  key='{cam_key}'")
        print(f"    Original Size: {w}x{h}  →  Scaled to: {int(w*scale)}x{int(h*scale)}")

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


def read_camera_frame(cam: dict) -> np.ndarray | None:
    """
    Read the current frame for a camera entry, whether it's a live feed
    (has "live_path"/"lock_path") or a static fallback image (has
    "static_path"). Returns None if the frame can't be read.
    """
    if cam.get("static_image",None) is not None:
        return cam["static_image"]
    return read_latest_frame(cam["live_path"], cam["lock_path"])


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
# SCENE REPLICA CONFIRMATION
# =============================================================================
 
def find_pose_folder(object_dir: str) -> str | None:
    """
    Search `object_dir` for subfolders matching "pose-[a-z]" (case-insensitive).
 
    Returns:
        - None if no matching folders are found (caller should error out)
        - The single matching folder name if exactly one is found
        - Otherwise, prompts the user via terminal input to choose one
          from the list of matches and returns that folder name
    """
    import re
 
    pattern = re.compile(r"^pose-[a-zA-Z]$")
 
    matches = sorted([
        entry.name for entry in os.scandir(object_dir)
        if entry.is_dir() and pattern.match(entry.name)
    ])
 
    if not matches:
        return None
 
    if len(matches) == 1:
        return matches[0]
 
    # Multiple candidates — ask the user to choose
    print(f"\n  Multiple pose folders found in: {object_dir}")
    for i, name in enumerate(matches, start=1):
        print(f"    [{i}] {name}")
 
    while True:
        choice = input(f"  Select a pose folder [1-{len(matches)}]: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(matches):
                return matches[idx - 1]
        print("  Invalid selection, please try again.")
 
 

 
def confirm_scene_replica(
    moad_config_path: str,
    scene_config_path: Path,
    scene_npz_path: Path,
) -> None:
    """
    Confirm that the current scene has been successfully replicated and copy
    the files needed for later annotation generation into the target data
    folder's "scene_replica" subdirectory.
 
    The target folder is resolved as:
        os.path.join(output_dir, object_name, <pose_folder>)
 
    where output_dir/object_name come from moad_config.json, and
    <pose_folder> is found by searching output_dir/object_name for
    subfolders matching "pose-[a-z]" (see find_pose_folder()).
 
    Copies:
        - the scene .npz file referenced by the scene config ("scene_file")
        - the scene config .json file itself
 
    Args:
        moad_config_path:  path to moad_config.json
        scene_config_path: path to the scene config JSON currently in use
                            (the file passed via --scene-config)
        scene_npz_path:    resolved path to the scene .npz file currently
                            loaded (scene_cfg["scene_file"], resolved relative
                            to the scene_replica repo's assets/scenes dir)
    """
    import shutil
 
    if not os.path.isfile(moad_config_path):
        print(f"  [ERROR] moad_config.json not found: {moad_config_path}")
        return
    
    print("  Loading current MOAD config file...")
    with open(moad_config_path, "r") as f:
        moad_cfg = json.load(f)
 
    try:
        output_dir  = moad_cfg["output_dir"]
        object_name = moad_cfg["object_name"]
    except KeyError as e:
        print(f"  [ERROR] moad_config.json missing expected key: {e}")
        return
 
    object_dir = os.path.join(output_dir, object_name)
 
    if not os.path.isdir(object_dir):
        print(f"  [ERROR] Object folder does not exist: {object_dir}")
        print(f"          (derived from output_dir / object_name)")
        return
 
    # ── Locate the pose folder ────────────────────────────────────────────────
    pose_folder = find_pose_folder(object_dir)
    if pose_folder is None:
        print(f"  [ERROR] No pose folders matching 'pose-[a-z]' found in: {object_dir}")
        return
    
    target_dir  = os.path.join(object_dir, pose_folder)
    print(f"  > Target Object: {object_name}")
    print(f"  > Target Pose: {pose_folder}")
    print(f"  > Target Directory: {target_dir}")
    replica_dir = os.path.join(target_dir, "scene_replica")
    
    if os.path.isdir(replica_dir):
        print("[WARNING] Pose Replica folder had already been created here.")
        input("Continue? (Ctrl-C to quit)")
    os.makedirs(replica_dir, exist_ok=True)
 
    # Copy the scene .npz file
    if scene_npz_path is not None and os.path.isfile(scene_npz_path):
        dest_npz = os.path.join(replica_dir, os.path.basename(scene_npz_path))
        shutil.copy2(scene_npz_path, dest_npz)
        print(f"  [COPIED] {scene_npz_path}  →  {dest_npz}")
    else:
        print(f"  [WARNING] Scene .npz file not found, skipping: {scene_npz_path}")
 
    # Copy the scene config .json file
    if os.path.isfile(scene_config_path):
        dest_cfg = os.path.join(replica_dir, os.path.basename(scene_config_path))
        shutil.copy2(scene_config_path, dest_cfg)
        print(f"  [COPIED] {scene_config_path}  →  {dest_cfg}")
    else:
        print(f"  [WARNING] Scene config file not found, skipping: {scene_config_path}")
 
    print(f"  Scene replica confirmed for: {object_name} / {pose_folder}")
    print(f"  → {replica_dir}")
 



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
    not_live: bool
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
        if not_live: # Color camera dots gray if using static fallback images
            color  = (150, 150, 150) if i == cam_index else (55, 55, 55)
        else: # Color dots green if live
            color  = (80, 200, 80) if i == cam_index else (55, 55, 55)
        cv2.circle(frame, (cx, dot_y), dot_r, color, -1)

    # Bottom bar — navigation hint
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 24), (w, h), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay2, 0.72, frame, 0.28, 0) 
    nav = "  , / L-Arr  prev camera        . / R-Arr  next camera        1-9  jump        C  confirm replica"
    cv2.putText(frame, nav, (0, h - 8),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (110, 110, 110), 1, cv2.LINE_AA)

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    print("\n" + "─" * 60)
    print("  MOAD Liveview Overlay")
    print("─" * 60)

    # ── Resolve paths from arguments ──────────────────────────────────────────
    liveview_root   = Path(args.liveview_root)
    cam_params_path = Path(args.calib_root) / args.calibration / "cam_parameters.json"
    scene_config    = Path(args.scene_config)

    # ── Discover live cameras ─────────────────────────────────────────────────
    print(f"\n  Scanning liveview root: {liveview_root}\n")
    cameras_lv = discover_cameras(liveview_root, args.live_filename, args.lock_filename)

    using_static = False
    if not cameras_lv:
        print("\n[WARNING] No live views found. Reverting to static background images...\n")
        time.sleep(1.0)
        cameras_lv = load_background_images(args.fallback_images_dir,args.fallback_images_pos, args.fallback_images_scale)
        if not cameras_lv:
            print("\n[ERROR] No active liveview feeds found, and no static background "
                  "images available either. "
                  "Is the C++ backend running with liveview started? "
                  f"Check --fallback-images-dir ({args.fallback_images_dir}).\n")
            sys.exit(1)
        using_static = True
        print(f"\n  {len(cameras_lv)} static background image(s) loaded.\n")
    else:
        print(f"\n  {len(cameras_lv)} active feed(s) found.\n")

    # ── Load calibration ──────────────────────────────────────────────────────
    print(f"  Loading calibration: {cam_params_path}\n")
    cameras_cal, intrinsics_full, scale_info = load_cam_parameters(cam_params_path)

    # Verify all live cameras have a calibration entry
    for lv_cam in cameras_lv:
        key = lv_cam["cam_key"]
        if key not in cameras_cal:
            print(f"[ERROR] Live camera '{lv_cam['name']}' maps to key '{key}' "
                  f"which is not in cam_parameters.json. "
                  f"Available keys: {sorted(cameras_cal.keys())}")
            sys.exit(1)

    # ── Load scene config ─────────────────────────────────────────────────────
    with open(scene_config, "r") as f:
        scene_cfg = json.load(f)

    scene_scale   = scale_info["scale"]
    global_offset = np.asarray(scene_cfg["CALIBRATION_OFFSET"], dtype=np.float64)

    # ── Read one frame to determine actual resolution ─────────────────────────
    print("  Reading a reference frame to determine resolution...")
    current_idx   = 0
    ref_cam       = cameras_lv[current_idx]
    ref_frame     = None

    if using_static:
        # Static images are read once — no need to poll/wait.
        ref_frame = read_camera_frame(ref_cam)
        if ref_frame is None:
            print(f"[ERROR] Could not read static background image: {ref_cam['static_path']}")
            sys.exit(1)
    else:
        deadline = time.monotonic() + 5.0
        while ref_frame is None and time.monotonic() < deadline:
            ref_frame = read_camera_frame(ref_cam)
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

    running = True
    while running:
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

        # ── Read latest frame (live feed or static fallback) ──────────────────
        frame = read_camera_frame(lv_cam)
        if frame is None:
            frame = last_good_frame.copy()
            if not using_static:
                cv2.putText(frame, "Waiting for frame...", (20, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (40, 40, 200), 2)
        else:
            # Resize frame to render resolution if needed
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

        draw_hud(display, current_idx, cameras_lv, turntable_deg, global_offset,using_static)
        if render_scale != 1.0: 
            outsize = (int(intrinsics["width"]*render_scale), int(intrinsics["height"]*render_scale))
            # print(outsize)
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
            running = False
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
            turntable_deg += args.rotation_step_large;  needs_render = True
        elif key == ord('a'):
            turntable_deg -= args.rotation_step_large;  needs_render = True
        elif key == ord('w'):
            turntable_deg += args.rotation_step_small;  needs_render = True
        elif key == ord('s'):
            turntable_deg -= args.rotation_step_small;  needs_render = True
        elif key == ord('l'):
            global_offset[0] += args.offset_step;       needs_render = True
        elif key == ord('j'):
            global_offset[0] -= args.offset_step;       needs_render = True
        elif key == ord('o'):
            global_offset[1] += args.offset_step;       needs_render = True
        elif key == ord('u'):
            global_offset[1] -= args.offset_step;       needs_render = True
        elif key == ord('i'):
            global_offset[2] += args.offset_step;       needs_render = True
        elif key == ord('k'):
            global_offset[2] -= args.offset_step;       needs_render = True
        elif key == ord(' '):
            turntable_deg  = 0.0
            global_offset  = np.asarray(scene_cfg["CALIBRATION_OFFSET"], dtype=np.float64)
            needs_render   = True
        elif key == ord('r'):
            needs_render   = True
        elif key == ord('c'):
            # Confirm the scene has been correctly replicated and copy the
            # scene .npz + scene config .json into the target data folder
            # for later annotation generation.
            print("\n  Confirming scene replica...")
            scene_npz_path = os.path.join(scene.scene_path, scene_cfg["scene_file"])
            confirm_scene_replica(
                moad_config_path  = args.moad_config,
                scene_config_path = scene_config,
                scene_npz_path    = scene_npz_path,
            )
            print()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    try:
        p.disconnect(scene.pb)
    except Exception:
        pass
    print("\n  Overlay closed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MOAD liveview overlay: composites a PyBullet scene render "
                     "on top of live DSLR feeds for scene alignment verification."
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    parser.add_argument("--liveview-root", default=str(SCRIPT_DIR / "../moad_cui/live_view_filestream"),
                        help="Directory containing per-camera liveview folders")
    parser.add_argument("--calib-root", default=str(SCRIPT_DIR / "../moad_cui/calibration"),
                        help="Root calibration directory")
    parser.add_argument("--calibration", default="55mm",
                        help="Folder name of camera calibration to use")
    parser.add_argument("--scene-config", default=str(SCRIPT_DIR / "config/scene_cfg_notag_55mm.json"),
                        help="Path to the scene config JSON (CALIBRATION_OFFSET, scene file, etc.)")
    parser.add_argument("--moad-config", default=str(SCRIPT_DIR / "../moad_cui/config/moad_config.json"),
                        help="Path to moad_config.json, used to resolve the target "
                             "data folder when confirming a scene replica")
    
    # ── Fallback Images (When live is not available) ────────────────────────────────────────────────
    parser.add_argument("--fallback-images-dir", default="assets/fallback_images/55mm",
                        help="Directory containing static background images "
                             "(cam#_NNN_img.jpg) used when no live views are found")
    parser.add_argument("--fallback-images-pos", default=0,
                        help="Turntable position (NNN) of fallback images.")
    parser.add_argument("--fallback-images-scale", default=0.16,
                        help="Scale fallback images (and intrinsics) for better rendering performance.")
    
    # ── Liveview file protocol ────────────────────────────────────────────────
    parser.add_argument("--live-filename", default="evf_live.jpg",
                        help="Filename of the live frame written by the C++ backend")
    parser.add_argument("--lock-filename", default="evf.lock",
                        help="Filename of the lockfile used while the live frame is being written")

    # ── Interaction step sizes ────────────────────────────────────────────────
    parser.add_argument("--offset-step", type=float, default=0.005,
                        help="Metres per keypress for global offset adjustment (I/J/K/L/U/O)")
    parser.add_argument("--rotation-step-large", type=float, default=10.0,
                        help="Degrees per keypress for turntable rotation (A/D)")
    parser.add_argument("--rotation-step-small", type=float, default=5.0,
                        help="Degrees per keypress for turntable rotation (W/S)")

    args = parser.parse_args()
    main(args)