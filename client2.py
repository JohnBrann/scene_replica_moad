"""
moad_demo_static.py — MOAD PyBullet Overlay Demo (Static Image Version)

Renders a PyBullet scene overlaid on a fixed background image, using
precomputed AprilTag detections and known camera parameters instead of
a live video stream.

The anchor tag detection from scene_anchors.json gives us the tag pose
in each camera's coordinate frame (R_cw_cv, t_cw_cv). This is passed
directly into MnetSceneReplica to compute the correct view matrix.

Virtual camera switching works by re-expressing the tag pose from the
anchor camera into the target virtual camera's frame, using the known
extrinsics from cam_parameters.json.

Controls:
    Q / ESC         quit
    N / P           next / previous camera view
    A / D           turntable Z-rotation +/- 5 degrees
    W / S           turntable Z-rotation +/- 1 degree
    I / K           global Z offset +/-
    J / L           global X offset +/-
    U / O           global Y offset +/-
    SPACE           reset turntable + offset
    R               reload / re-render current view
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

try:
    from scene_replica import MnetSceneReplica, Rt_to_T, T_inv, make_transparent
    import pybullet as p
except Exception as e:
    print(f"Import error: {e}")
    raise SystemExit(1)


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR      = Path(__file__).parent.resolve()
CAM_PARAMS_PATH = SCRIPT_DIR / "../moad_cui/calibration/55mm" / "cam_parameters.json"
ANCHORS_PATH    = SCRIPT_DIR / "../moad_cui/calibration/55mm" / "scene_anchors.json"
ANCHOR_CAM      = "cam4"

SCENE_SCALE = 8.14#1.0#8.14   # tune this — metres to NeRF scene units
# Background image shown behind the render — swap this path as needed
# BACKGROUND_IMAGE = SCRIPT_DIR / "background.jpg"
IMAGE_DIR = "/home/csrobot/MOAD_DATA/artag_test/pose-a/DSLR/"
IMAGE_POSITION = 0
BACKGROUND_IMAGE = f"/home/csrobot/MOAD_DATA/artag_test/pose-a/DSLR/{ANCHOR_CAM}_000_img.jpg"

SCENE_ID        = "1.npz"              # scene file to load
RENDER_SCALE   = 0.25                 # scale factor for display window (images are large)
DISPLAY_SCALE   = 1.0                # scale factor for display window (images are large)
OFFSET_STEP     = 0.005               # metres per keypress
ROTATION_STEP_LARGE = 10.0             # degrees per A/D keypress
ROTATION_STEP_SMALL = 5.0             # degrees per W/S keypress


# =============================================================================
# DATA LOADING
# =============================================================================
 
def load_cam_parameters(path: Path) -> tuple[dict, dict]:
    with open(path, "r") as f:
        data = json.load(f)

    # DEBUG TESTING ===================
    cams = data["cameras"]
    names = sorted(cams.keys())
    for name in names:
        t = np.array(cams[name]["extrinsics"]["c2w"])[:3, 3]
        print(f"{name}: {t}  |t|={np.linalg.norm(t):.3f}")
    # Estimate scale from two cameras whose physical separation you know
    t_cam3 = np.array(cams["cam3"]["extrinsics"]["c2w"])[:3, 3]
    t_cam5 = np.array(cams["cam5"]["extrinsics"]["c2w"])[:3, 3]
    nerf_dist = np.linalg.norm(t_cam3 - t_cam5)
    physical_dist_metres = 0.6  # measure this physically NOTE: THIS IS A ROUGH ESTIMATE
    scale = nerf_dist / physical_dist_metres
    print(f"Scale factor: {scale:.3f}")
    # =====================
    intrinsics = data["intrinsics"]
    cameras    = {}
    for cam_name, cam_data in data["cameras"].items():
        cameras[cam_name] = {
            "intrinsics": intrinsics,
            "c2w": np.array(cam_data["extrinsics"]["c2w"], dtype=np.float64),
            "w2c": np.array(cam_data["extrinsics"]["w2c"], dtype=np.float64),
        }
    print(f"  Loaded {len(cameras)} cameras: {sorted(cameras.keys())}")
    return cameras, intrinsics
 
 
def load_anchor_detections(path: Path) -> tuple[dict, str]:
    """
    Load precomputed AprilTag detections from scene_anchors.json.
    Applies the same R_FLIP_WORLD that detect_apriltag() applies so
    output is consistent with what MnetSceneReplica expects.
    """
    with open(path, "r") as f:
        data = json.load(f)
 
    R_FLIP_WORLD = np.diag([1.0, -1.0, -1.0])
    anchors = {}
 
    for cam_name, det_data in data["detections"].items():
        if det_data is None:
            continue
        for tag in det_data["tags"]:
            R_raw   = np.array(tag["pose"]["R"], dtype=np.float64)
            t_raw   = np.array(tag["pose"]["t"], dtype=np.float64).reshape(3)
            anchors[cam_name] = {
                "tag_id":   tag["tag_id"],
                "R_cw_cv":  R_raw @ R_FLIP_WORLD.T,
                "t_cw_cv":  t_raw,
                "corners":  np.array(tag["corners"], dtype=np.int32),
                "center":   tag["center"],
            }
            break   # use first tag per camera
 
    print(f"  Loaded anchor detections for: {sorted(anchors.keys())}")
    return anchors, data.get("object_name", "unknown")
 
 
# =============================================================================
# VIRTUAL CAMERA SYNTHESIS
# =============================================================================
def scale_intrinsics(intrinsics: dict, scale: float) -> dict:
    """
    Returns a copy of intrinsics scaled for a downsampled image.
    Rotations and translations are unaffected — only pixel-space
    values (fx, fy, cx, cy, width, height) change.
    """
    return {
        **intrinsics,                              # preserve distortion, camera_model etc.
        "fx":     intrinsics["fx"]     * scale,
        "fy":     intrinsics["fy"]     * scale,
        "cx":     intrinsics["cx"]     * scale,
        "cy":     intrinsics["cy"]     * scale,
        "width":  int(intrinsics["width"]  * scale),
        "height": int(intrinsics["height"] * scale),
    }
 
def get_tag_pose_in_camera(
    anchor_cam: str,
    target_cam: str,
    anchors:    dict,
    cameras:    dict,
    scene_scale,
    verbose = True
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Re-express the AprilTag pose from the anchor camera's frame into the
    target camera's frame using known extrinsics.
    """
    # OpenCV ↔ OpenGL axis convention flip
    CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
    if verbose:
        print(f"Re-expressing tag position from {anchor_cam} -> {target_cam}")
    if anchor_cam not in anchors:
        print(f"  WARNING: No anchor detection for {anchor_cam}")
        return None
    if anchor_cam not in cameras or target_cam not in cameras:
        print(f"  WARNING: Camera not found in parameters.")
        return None
 
    anchor      = anchors[anchor_cam]
    T_ct_anchor = Rt_to_T(anchor["R_cw_cv"], anchor["t_cw_cv"])

    # Scale tag translation into NeRF scene units for the world-space step
    T_ct_scaled = T_ct_anchor.copy()
    T_ct_scaled[:3, 3] *= scene_scale

    # Into world, then into target camera
    T_wt        = cameras[anchor_cam]["c2w"] @ CV_TO_GL @ T_ct_scaled
    T_ct_target = CV_TO_GL @ cameras[target_cam]["w2c"] @ T_wt

    # Scale translation back to metres for MnetSceneReplica
    T_ct_target[:3, 3] /= scene_scale

    tag_T, tag_R = T_ct_target[:3, :3], T_ct_target[:3, 3]
    print(f"New Transform ({target_cam}): \n{T_ct_target}")
    return tag_T, tag_R
 
def build_cam_K(intrinsics: dict) -> np.ndarray:
    return np.array([
        [intrinsics["fx"],             0.0, intrinsics["cx"]],
        [            0.0, intrinsics["fy"], intrinsics["cy"]],
        [            0.0,             0.0,             1.0],
    ], dtype=np.float64)
 
 
# =============================================================================
# TURNTABLE / OFFSET
# =============================================================================
 
def make_z_rotation(degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    R = np.eye(4)
    R[0, 0] =  c;  R[0, 1] = -s
    R[1, 0] =  s;  R[1, 1] =  c
    return R
 
 
def apply_turntable_and_offset(
    R_cw_cv:       np.ndarray,
    t_cw_cv:       np.ndarray,
    turntable_deg: float,
    global_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T_ct           = Rt_to_T(R_cw_cv, t_cw_cv)
    T_local        = make_z_rotation(turntable_deg)
    T_local[:3, 3] = global_offset
    T_modified     = T_ct @ T_local
    return T_modified[:3, :3], T_modified[:3, 3]
 
 
# =============================================================================
# OVERLAY & HUD
# =============================================================================
def load_background_images(
    cam_names:  list,
    image_dir:  str,
    position:   int,
    image_size: tuple[int, int],
) -> dict:
    """
    Loads background images for each camera from a specified directory.
    Expects filenames in the format: cam{N}_{position:03d}_img.jpg

    Args:
        cam_names:  list of camera names e.g. ["cam1", "cam2", ...]
        image_dir:  directory containing the images
        position:   turntable position index (used in filename)
        image_size: (width, height) to resize images to
    """
    backgrounds = {}
    for cam_name in cam_names:
        # Extract camera index from name e.g. "cam3" → 3
        cam_idx  = int(cam_name.replace("cam", ""))
        filename = f"cam{cam_idx}_{position:03d}_img.jpg"
        path     = Path(image_dir) / filename
        print(f" > Loading: {path}...")
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARNING: Could not load background for {cam_name}: {path}")
            continue
        if img.shape[1] != image_size[0] or img.shape[0] != image_size[1]:
            img = cv2.resize(img, image_size)
        backgrounds[cam_name] = img
        print(f"  {cam_name}: {path}")

    return backgrounds
 
def overlay_rgba_on_bgr(background_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    bg      = background_bgr.astype(np.float32)
    ov      = overlay_rgba.astype(np.float32)
    ov_bgra = ov[:, :, [2, 1, 0, 3]]
    alpha   = ov_bgra[:, :, 3:4] / 255.0
    out     = ov_bgra[:, :, :3] * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def draw_hud(
    image:         np.ndarray,
    anchor_cam:    str,
    current_cam:   str,
    turntable_deg: float,
    global_offset: np.ndarray,
    cam_names:     list,
) -> np.ndarray:
    out   = image.copy()
    lines = [
        f"Anchor cam:  {anchor_cam}",
        f"View cam:    {current_cam}  [{cam_names.index(current_cam)+1}/{len(cam_names)}]",
        f"Turntable:   {turntable_deg:+.1f} deg",
        f"Offset XYZ:  [{global_offset[0]:+.3f}, {global_offset[1]:+.3f}, {global_offset[2]:+.3f}] m",
    ]
    controls = [
        "N/P: next/prev cam view",
        "A/D: rotate +/-5deg   W/S: +/-1deg",
        "IJKL: X/Z offset   UO: Y offset",
        "SPACE: reset   R: re-render   Q/ESC: quit",
    ]
 
    y = 22
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),       2, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 20
 
    h  = out.shape[0]
    yc = h - len(controls) * 18 - 6
    for line in controls:
        cv2.putText(out, line, (8, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0),       2, cv2.LINE_AA)
        cv2.putText(out, line, (8, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)
        yc += 18
 
    return out
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    # --- Load data ---
    print("\nLoading camera parameters...")
    cameras, intrinsics_full = load_cam_parameters(CAM_PARAMS_PATH)
    cam_names = sorted(cameras.keys())

    # Scale intrinsics (images are scaled accordingly when loaded)
    intrinsics = scale_intrinsics(intrinsics_full, RENDER_SCALE)
    
    # Load AR Anchor Calibrations, set anchor camera
    print("Loading anchor detections...")
    anchors, object_name = load_anchor_detections(ANCHORS_PATH)
    anchor_cam = next((c for c in cam_names if c in anchors), None)
    if anchor_cam is None:
        print("ERROR: No anchor detections found for any camera in cam_parameters.")
        sys.exit(1)
    # Manually set anchor cam
    anchor_cam = ANCHOR_CAM
    print(f"  Anchor camera: {anchor_cam}")
    
    # Load Background Images for all camera, 
    bkg_images = {}
    print("Loading background images...")
    image_size  = (intrinsics["width"], intrinsics["height"])
    backgrounds = load_background_images(cam_names, IMAGE_DIR, IMAGE_POSITION, image_size)
    black_frame = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    background = backgrounds.get(anchor_cam, black_frame)

 
    target_w, target_h = intrinsics["width"], intrinsics["height"]

    # --- Interactive state ---
    current_cam_idx = cam_names.index(anchor_cam)
    turntable_deg   = 0.0
    global_offset   = np.zeros(3, dtype=np.float64)
    needs_render    = True
    rgba_cache      = None
 
    # --- Initialise scene ONCE ---
    print("\nInitialising PyBullet scene (one-time load)...")
    cam_K  = build_cam_K(intrinsics)
    anchor = anchors[anchor_cam]
 
    scene = MnetSceneReplica(
        cam_K   = cam_K,
        W       = intrinsics["width"],
        H       = intrinsics["height"],
        det     = None,
        tag_id  = anchor["tag_id"],
        corners = anchor["corners"],
        R_cw_cv = anchor["R_cw_cv"],
        t_cw_cv = anchor["t_cw_cv"],
    )
    scene.load_scene(SCENE_ID)
    print("  Scene loaded.\n")
 
    # --- CV window ---
    WINDOW = f"MOAD Demo — {object_name}"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    disp_w = int(target_w * DISPLAY_SCALE)
    disp_h = int(target_h * DISPLAY_SCALE)
    cv2.resizeWindow(WINDOW, disp_w, disp_h)
    print(f"  Display: {disp_w}x{disp_h}  (scale={DISPLAY_SCALE})")
    print("Running. Press Q or ESC to quit.\n")
 
    while True:
        current_cam = cam_names[current_cam_idx]
 
        # --- Re-render only when state changed ---
        if needs_render:
            background = backgrounds.get(current_cam, black_frame)
            tag_pose = get_tag_pose_in_camera(
                anchor_cam, current_cam, anchors, cameras, SCENE_SCALE
            )
            if tag_pose is not None:
                R_cw_cv, t_cw_cv = tag_pose
                R_mod, t_mod = apply_turntable_and_offset(
                    R_cw_cv, t_cw_cv, turntable_deg, global_offset
                )
                print(f"  Rendering: view={current_cam}"
                      f"  rot={turntable_deg:+.1f}deg"
                      f"  offset={np.round(global_offset, 3)}")
                rgba_cache = scene.update_and_render(R_mod, t_mod)
            else:
                print(f"  Cannot compute pose for {current_cam}")
                rgba_cache = None
            needs_render = False
 
        # --- Composite + HUD ---
        if rgba_cache is not None:
            display = overlay_rgba_on_bgr(background, rgba_cache)
            display = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        
        else:
            display = background.copy()
            display = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        
            cv2.putText(display, f"No render for {current_cam}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
 
        display = draw_hud(
            display, anchor_cam, current_cam,
            turntable_deg, global_offset, cam_names,
        )
        # display_small = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, display)
 
        # --- Key handling ---
        key = cv2.waitKey(30) & 0xFF
 
        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            current_cam_idx = (current_cam_idx + 1) % len(cam_names)
            needs_render = True
        elif key == ord('p'):
            current_cam_idx = (current_cam_idx - 1) % len(cam_names)
            needs_render = True
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
            turntable_deg = 0.0
            global_offset[:] = 0.0
            needs_render = True
        elif key == ord('r'):
            needs_render = True
 
    # --- Cleanup ---
    cv2.destroyAllWindows()
    try:
        p.disconnect(scene.pb)
    except Exception:
        pass
    print("Done.")
 
 
if __name__ == "__main__":
    main()
