"""
client3.py — MOAD PyBullet Overlay Demo (Calibration-Direct Version)

Renders a PyBullet scene overlaid on real camera images using known camera
extrinsics directly — no AprilTag detection required.

Since the alignment transform has already been applied in cam_parameters.json,
the turntable centre sits precisely at the world origin and the XY plane is
level. The virtual scene is anchored at its own origin. The camera extrinsics
therefore directly define the correct view of the scene for each camera,
without any intermediate AR tag pose.

The view matrix for each camera is derived by converting the calibrated w2c
extrinsic into the (R_cw_cv, t_cw_cv) form expected by MnetSceneReplica,
with the appropriate OpenCV/OpenGL axis convention flip.

Turntable rotation and global offset are applied as a local transform in
world space around the scene origin.

Controls:
    Q / ESC         quit
    N / P           next / previous camera view
    A / D           turntable Z-rotation +/- large step
    W / S           turntable Z-rotation +/- small step
    I / K           global Z offset +/-
    J / L           global X offset +/-
    U / O           global Y offset +/-
    SPACE           reset turntable + offset
    R               force re-render current view
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

try:
    from scene_replica import MnetSceneReplica, Rt_to_T, T_inv
    import pybullet as p
except Exception as e:
    print(f"Import error: {e}")
    raise SystemExit(1)


# =============================================================================
# CONFIG
# =============================================================================

CALIBRATION     = "test"
SCRIPT_DIR      = Path(__file__).parent.resolve()
CAM_PARAMS_PATH = SCRIPT_DIR / f"../moad_cui/calibration/{CALIBRATION}" / "cam_parameters.json"

IMAGE_DIR       = f"/home/csrobot/MOAD_DATA/artag_test_18mm/pose-a/DSLR/"
IMAGE_POSITION  = 10

SCENE_ID            = "moad_gears_close.npz"
RENDER_SCALE        = 0.25      # scale intrinsics and images to this fraction
DISPLAY_SCALE       = 1.0       # additional display scaling (1.0 = no extra scaling)
OFFSET_STEP         = 0.005     # metres per keypress
ROTATION_STEP_LARGE = 10.0      # degrees per A/D keypress
ROTATION_STEP_SMALL = 5.0       # degrees per W/S keypress

# OpenCV ↔ OpenGL axis flip — applied when converting between NeRF/COLMAP
# convention (camera looks along -Z, Y up) and OpenCV convention (camera
# looks along +Z, Y down) used by MnetSceneReplica.
# CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
CV_TO_GL = np.diag([1.0, 1.0, 1.0, 1.0])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cam_parameters(path: Path) -> tuple[dict, dict]:
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

    print(f"  Loaded {len(cameras)} cameras: {sorted(cameras.keys())}")

    # Print camera positions and inter-camera distances for reference
    names = sorted(cameras.keys())
    for name in names:
        t = cameras[name]["c2w"][:3, 3]
        print(f"    {name}: position={np.round(t, 3)}  |t|={np.linalg.norm(t):.3f}")

    return cameras, intrinsics, scale_info


def scale_intrinsics(intrinsics: dict, scale: float) -> dict:
    return {
        **intrinsics,
        "fx":     intrinsics["fx"]    * scale,
        "fy":     intrinsics["fy"]    * scale,
        "cx":     intrinsics["cx"]    * scale,
        "cy":     intrinsics["cy"]    * scale,
        "width":  int(intrinsics["width"]  * scale),
        "height": int(intrinsics["height"] * scale),
    }


def load_background_images(
    cam_names:  list,
    image_dir:  str,
    position:   int,
    image_size: tuple[int, int],
) -> dict:
    backgrounds = {}
    for cam_name in cam_names:
        cam_idx  = int(cam_name.replace("cam", ""))
        filename = f"cam{cam_idx}_{position:03d}_img.jpg"
        path     = Path(image_dir) / filename
        print(f"  Loading: {path}")
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARNING: Could not load background for {cam_name}: {path}")
            continue
        if img.shape[1] != image_size[0] or img.shape[0] != image_size[1]:
            img = cv2.resize(img, image_size)
        backgrounds[cam_name] = img
    return backgrounds


def build_cam_K(intrinsics: dict) -> np.ndarray:
    return np.array([
        [intrinsics["fx"],             0.0, intrinsics["cx"]],
        [            0.0, intrinsics["fy"], intrinsics["cy"]],
        [            0.0,             0.0,             1.0],
    ], dtype=np.float64)


# =============================================================================
# CALIBRATION-DIRECT CAMERA POSE
# =============================================================================

def get_camera_pose_from_extrinsics(
    cam_name:      str,
    cameras:       dict,
    turntable_deg: float,
    global_offset: np.ndarray,
    scale:          float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive (R_cw_cv, t_cw_cv) for MnetSceneReplica directly from the
    calibrated camera extrinsics, with turntable rotation and offset applied.

    The extrinsics in cam_parameters.json are in NeRF/COLMAP convention:
        c2w: camera-to-world, camera looks along -Z (OpenGL)

    MnetSceneReplica expects OpenCV convention:
        R_cw_cv, t_cw_cv: world-origin-to-camera transform, camera looks along +Z

    Conversion:
        T_w2c_nerf  = inv(c2w)                  # world → camera, OpenGL
        T_w2c_cv    = CV_TO_GL @ T_w2c_nerf      # flip to OpenCV axes

    Turntable and offset are applied as a world-space transform around the
    scene origin before projecting into camera space:
        T_world_modified = T_turntable_and_offset
        T_w2c_final      = T_w2c_cv @ T_world_modified

    Args:
        cam_name:      camera to render from
        cameras:       dict from load_cam_parameters
        turntable_deg: Z-rotation of the scene around the world origin (degrees)
        global_offset: XYZ translation of the scene origin (metres)

    Returns:
        (R_cw_cv, t_cw_cv) ready to pass to scene.update_and_render()
    """
    # scale = 0.5
    T_c2w    = cameras[cam_name]["c2w"].copy()      # [4x4] camera→world (OpenGL/NeRF)
    T_w2c    = cameras[cam_name]["w2c"].copy()      # [4x4] world→camera (OpenGL/NeRF)
    
    # Scale the translational part of both extrinsic matrices.
    # The rotation columns are dimensionless and must not be scaled.
    # c2w[:3,3] is the camera position in world space.
    # w2c[:3,3] is -(R^T @ t_world), i.e. the negated rotated translation —
    # scaling it is equivalent to scaling the original world translation.
    # T_c2w[:3, 3] *= scale
    T_w2c[:3, 3] *= scale

    # Flip to OpenCV convention: camera looks along +Z instead of -Z
    T_w2c_cv = CV_TO_GL @ T_w2c            # [4x4] world→camera (OpenCV)

    # Build world-space scene transform: rotate around Z, then translate
    T_scene  = make_z_rotation(turntable_deg)
    T_scene[:3, 3] = global_offset * scale

    # Apply scene transform: project from the rotated/offset scene origin
    T_final  = T_w2c_cv @ T_scene          # [4x4] scene-origin→camera (OpenCV)

    R_cw_cv  = T_final[:3, :3]
    t_cw_cv  = T_final[:3,  3]

    return R_cw_cv, t_cw_cv


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


# =============================================================================
# OVERLAY & HUD
# =============================================================================

def overlay_rgba_on_bgr(background_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    bg      = background_bgr.astype(np.float32)
    ov      = overlay_rgba.astype(np.float32)
    ov_bgra = ov[:, :, [2, 1, 0, 3]]
    alpha   = ov_bgra[:, :, 3:4] / 255.0
    out     = ov_bgra[:, :, :3] * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_hud(
    image:         np.ndarray,
    current_cam:   str,
    turntable_deg: float,
    global_offset: np.ndarray,
    cam_names:     list,
) -> np.ndarray:
    out   = image.copy()
    lines = [
        f"View cam:    {current_cam}  [{cam_names.index(current_cam)+1}/{len(cam_names)}]",
        f"Turntable:   {turntable_deg:+.1f} deg",
        f"Offset XYZ:  [{global_offset[0]:+.3f}, {global_offset[1]:+.3f}, {global_offset[2]:+.3f}] m",
    ]
    controls = [
        "N/P: next/prev cam view",
        "A/D: rotate +/-10deg   W/S: +/-5deg",
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
    # --- Load camera parameters ---
    print("\nLoading camera parameters...")
    cameras, intrinsics_full, scale_info = load_cam_parameters(CAM_PARAMS_PATH)
    cam_names = sorted(cameras.keys())

    intrinsics = scale_intrinsics(intrinsics_full, RENDER_SCALE)
    print(f"  Render resolution: {intrinsics['width']} x {intrinsics['height']}")

    # --- Load background images ---
    print("\nLoading background images...")
    image_size  = (intrinsics["width"], intrinsics["height"])
    backgrounds = load_background_images(cam_names, IMAGE_DIR, IMAGE_POSITION, image_size)
    black_frame = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # --- Interactive state ---
    current_cam_idx = 0
    turntable_deg   = 0.0
    global_offset   = np.zeros(3, dtype=np.float64)
    scene_scale =   scale_info["scale"]
    needs_render    = True
    rgba_cache      = None

    # --- Initialise PyBullet scene ONCE ---
    # MnetSceneReplica needs an initial pose to set up its view matrix.
    # We use the first camera's pose — it will be updated immediately on
    # the first render pass via update_and_render().
    print("\nInitialising PyBullet scene (one-time load)...")
    cam_K        = build_cam_K(intrinsics)
    initial_cam  = cam_names[current_cam_idx]
    R_init, t_init = get_camera_pose_from_extrinsics(
        initial_cam, cameras, turntable_deg, global_offset, scene_scale
    )

    scene = MnetSceneReplica(
        cam_K   = cam_K,
        W       = intrinsics["width"],
        H       = intrinsics["height"],
        det     = None,
        tag_id  = 0,        # unused — no AR tag involved
        corners = None,
        R_cw_cv = R_init,
        t_cw_cv = t_init,
    )
    scene.load_scene(SCENE_ID)
    print("  Scene loaded.\n")

    # --- CV window ---
    target_w = intrinsics["width"]
    target_h = intrinsics["height"]
    disp_w   = int(target_w * DISPLAY_SCALE)
    disp_h   = int(target_h * DISPLAY_SCALE)

    WINDOW = "MOAD Demo (calibration-direct)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, disp_w, disp_h)
    print(f"  Display: {disp_w}x{disp_h}")
    print("Running. Press Q or ESC to quit.\n")

    while True:
        current_cam = cam_names[current_cam_idx]

        # --- Re-render when state changed ---
        if needs_render:
            background = backgrounds.get(current_cam, black_frame)

            R_cw_cv, t_cw_cv = get_camera_pose_from_extrinsics(
                current_cam, cameras, turntable_deg, global_offset, scene_scale
            )
            print(f"  Rendering: view={current_cam}"
                  f"  rot={turntable_deg:+.1f}deg"
                  f"  offset={np.round(global_offset, 3)}")

            rgba_cache   = scene.update_and_render(R_cw_cv, t_cw_cv)
            needs_render = False

        # --- Composite + HUD ---
        if rgba_cache is not None:
            display = overlay_rgba_on_bgr(background, rgba_cache)
        else:
            display = background.copy()
            cv2.putText(display, f"No render for {current_cam}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        display = draw_hud(display, current_cam, turntable_deg, global_offset, cam_names)
        display = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
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
