import os
import json
import numpy as np

try:
    import cv2
    from scene_replica import detect_apriltag, MnetSceneReplica
except Exception as e:
    print(f"Import error: {e}")
    raise SystemExit(1)

# rgba to bgr conversion for cv
def overlay_rgba_on_bgr(background_bgr, overlay_rgba):
    bg = background_bgr.astype(np.float32)
    ov = overlay_rgba.astype(np.float32)

    # Convert RGBA -> BGRA for OpenCV-style compositing
    ov_bgra = ov[:, :, [2, 1, 0, 3]]

    alpha = ov_bgra[:, :, 3:4] / 255.0
    out = ov_bgra[:, :, :3] * alpha + bg * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(out)

# Get Scene Replica Overlay
def initialize_overlay(frame, cam_K, cam_width, cam_height, scene_id):
    
    # Detect April Tag
    apriltag_detected = detect_apriltag(frame, cam_K)
    if not apriltag_detected:
        return None

    det, tag_id, corners, R_cw_cv, t_cw_cv = apriltag_detected
    print(f"AprilTag detected. Tag ID: {tag_id}")

    scene_render = MnetSceneReplica(
        cam_K,
        cam_width,
        cam_height,
        det,
        tag_id,
        corners,
        R_cw_cv,
        t_cw_cv,
    )

    scene_render.load_scene(scene_id)
    rgba = scene_render.render_scene_image()
    return rgba


def main():
    # Open Config File
    with open("./config/camera_cfg.json", "r") as f:
        cfg = json.load(f)

    # Camera Config 
    cam_K = np.array(cfg["camera"]["cam_K"], dtype=float)
    cam_width = int(cfg["camera"]["width"])
    cam_height = int(cfg["camera"]["height"])

    # Load Scene
    with open("./config/scene_replica_cfg.json", "r") as f:
        cfg = json.load(f)

    scene_id = f'{cfg["scene"]["scene_number"]}.npz'

    # Intialize cv2 Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    if not cap.isOpened():
        print("Could not open video stream.")
        return


    # Read exactly one startup frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        cv2.destroyAllWindows()
        return

    if frame.shape[1] != cam_width or frame.shape[0] != cam_height:
        frame = cv2.resize(frame, (cam_width, cam_height))

    # Initialize only from the first frame
    overlay_rgba = initialize_overlay(
        frame, cam_K, cam_width, cam_height, scene_id
    )

    if overlay_rgba is None:
        print("AprilTag not detected in first frame. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        if frame.shape[1] != cam_width or frame.shape[0] != cam_height:
            frame = cv2.resize(frame, (cam_width, cam_height))

        # Reuse the same overlay every frame at the same pixel location
        display = overlay_rgba_on_bgr(frame, overlay_rgba)
        display = np.ascontiguousarray(display)
        cv2.imshow("Live Scene Overlay", display)

        key = cv2.waitKey(1) & 0xFF
        # If 's' is pressed, recapture the scene overlay
        if key == ord("s"):
            overlay_rgba = initialize_overlay(
            frame, cam_K, cam_width, cam_height, scene_id
        )

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()