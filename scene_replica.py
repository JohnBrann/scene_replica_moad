# Implementation of the class for rendering real world scene layouts for grasping in clutters task
# DO NOT MODIFY THIS FILE
# Contact: support@manipulation-net.org

import os
import sys
import glob
import time
import json
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pybullet as p
    from pupil_apriltags import Detector

except Exception as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and properly configured.")
    exit()


# Open Config File
with open("./config/scene_replica_cfg.json", "r") as f:
    cfg = json.load(f)

APRILTAG_SIZE = cfg["april_tag"]["APRILTAG_SIZE"]
APRILTAG_FAMILY = cfg["april_tag"]["APRILTAG_FAMILY"]
PB_RENDER = getattr(p, cfg["april_tag"]["PB_RENDER"])
WORLD_OFFSET = np.array(cfg["april_tag"]["WORLD_OFFSET"])
CAMERA_NEAR = cfg["april_tag"]["CAMERA_NEAR"]
CAMERA_FAR = cfg["april_tag"]["CAMERA_FAR"]


def Rt_to_T(R, t):
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def detect_apriltag(apriltag_img, cam_K: np.ndarray):
    """
    Detect the apriltag in the image and return the tag id, corners, and the pose of the tag in the camera coordinate
    """
    assert cam_K.shape == (3, 3), "Camera intrinsic matrix must be a 3x3 matrix"
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    gray = cv2.cvtColor(apriltag_img, cv2.COLOR_BGR2GRAY)
    det = Detector(families="tag36h11").detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=APRILTAG_SIZE,
    )
    detections = det

    if not det:
        return None

    det = max(detections, key=lambda d: getattr(d, "decision_margin", 0.0))
    tag_id = getattr(det, "tag_id", 0)
    corners = np.int32(getattr(det, "corners", [])) if hasattr(det, "corners") else None
    R_cw_cv = det.pose_R
    t_cw_cv = np.asarray(det.pose_t, dtype=float).reshape(3)
    R_FLIP_WORLD = np.diag([1.0, -1.0, -1.0])  # flip the world coordinate

    R_cw_cv = R_cw_cv @ R_FLIP_WORLD.T

    return det, tag_id, corners, R_cw_cv, t_cw_cv


def make_transparent(rgb, seg, alpha=0.6):
    """
    Combine PyBullet RGB and segmentation images into RGBA with transparent background.
    """
    rgb = np.asarray(rgb)
    seg = np.asarray(seg)

    # Drop any existing alpha channel
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    alpha = np.where(seg == -1, 0, 255 * alpha).astype(np.uint8)
    rgba = np.dstack((rgb, alpha)).astype(np.uint8)
    return rgba


def seg_stats(seg):
    seg = np.asarray(seg).astype(np.int32, copy=False)
    uniq, counts = np.unique(seg, return_counts=True)
    print("Unique IDs (value: count):")
    for u, c in zip(uniq, counts):
        print(f"{u:>12}: {c}")


def resize_rgba(arr, h, w):
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[:2] == (h, w):
        return a.astype(np.uint8, copy=False)
    return a.reshape(h, w, 4).astype(np.uint8)


def resize_seg(arr, h, w):
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape == (h, w):
        return a.astype(np.int32, copy=False)
    return a.reshape(h, w).astype(np.int32)


class MnetSceneReplica:
    def __init__(
        self, cam_K, W, H, det, tag_id, corners, R_cw_cv, t_cw_cv
    ):
        self.pb = p.connect(p.DIRECT)
        self.cam_K = cam_K
        self.W = W
        self.H = H
        self.det = det
        self.tag_id = tag_id
        self.corners = corners
        self.R_cw_cv = R_cw_cv
        self.t_cw_cv = t_cw_cv
        self.model_lib = {}
        self.projection_matrix = None
        self.view_matrix = None
        self.object_model_path = os.path.join("assets", "object_sets", "moad")
        self.scene_path = os.path.join("assets", "scenes")
        self.near = CAMERA_NEAR
        self.far = CAMERA_FAR
        self.urdf_models = []
        self.scene_layouts = []
        self.load_assets()
        self._compute_projection_matrix()
        self._compute_view_matrix()

    def create_visual_only_bars(self):
        length = 0.5          # length of each bar
        half_length = length / 2.0
        height = 0.005       # bar height
        z_pos = height / 2.0  
        color = [1, 0, 0, 1]  # red

        bar_visual_x = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_length, height / 2, height / 2],
            rgbaColor=color
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bar_visual_x,
            baseCollisionShapeIndex=-1,
            basePosition=np.array([0, +half_length, z_pos])+WORLD_OFFSET
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bar_visual_x,
            baseCollisionShapeIndex=-1,
            basePosition=np.array([0, -half_length, z_pos])+WORLD_OFFSET
        )

        bar_visual_y = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[height / 2, half_length, height / 2],
            rgbaColor=color
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bar_visual_y,
            baseCollisionShapeIndex=-1,
            basePosition=np.array([+half_length, 0, z_pos])+WORLD_OFFSET
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bar_visual_y,
            baseCollisionShapeIndex=-1,
            basePosition=np.array([-half_length, 0, z_pos])+WORLD_OFFSET
        )
    def create_visual_only_circle(self, diameter: float = 0.5, notch_size: float = 0.005):
        """
        Creates a visual-only circle (approximated by thin box segments) with a
        small notch box indicating the zero rotational position (along +X axis).

        Args:
            diameter:   diameter of the circle in metres
            notch_size: side length of the notch box in metres
        """
        radius     = diameter / 2.0
        thickness  = 0.005          # bar cross-section
        z_pos      = thickness / 2.0
        num_segs   = 32             # more segments = smoother circle
        color_ring = [1, 0, 0, 1]  # red ring
        color_notch= [1, 0.5, 0, 1]  # orange notch

        for i in range(num_segs):
            angle_start = (2 * np.pi * i)       / num_segs
            angle_end   = (2 * np.pi * (i + 1)) / num_segs
            angle_mid   = (angle_start + angle_end) / 2.0

            # Centre of this segment
            cx = radius * np.cos(angle_mid)
            cy = radius * np.sin(angle_mid)

            # Chord length between the two endpoints
            chord = 2 * radius * np.sin(np.pi / num_segs)

            # Half extents: long axis = half chord, short axes = thickness
            half_extents = [chord / 2.0, thickness / 2.0, thickness / 2.0]

            vis = p.createVisualShape(
                shapeType  = p.GEOM_BOX,
                halfExtents= half_extents,
                rgbaColor  = color_ring,
            )

            # Quaternion rotating the segment to sit tangent to the circle
            # Each segment is rotated by angle_mid around Z
            # quat = p.getQuaternionFromEuler([0, 0, angle_mid])
            quat = p.getQuaternionFromEuler([0, 0, angle_mid + np.pi / 2])

            p.createMultiBody(
                baseMass               = 0,
                baseVisualShapeIndex   = vis,
                baseCollisionShapeIndex= -1,
                basePosition           = np.array([cx, cy, z_pos]) + WORLD_OFFSET,
                baseOrientation        = quat,
            )

        # --- Notch box at +X (zero rotation position) ---
        if notch_size is not None:
            notch_vis = p.createVisualShape(
                shapeType  = p.GEOM_BOX,
                halfExtents= [0.1, notch_size / 2.0, notch_size / 2.0],
                rgbaColor  = color_notch,
            )
            p.createMultiBody(
                baseMass               = 0,
                baseVisualShapeIndex   = notch_vis,
                baseCollisionShapeIndex= -1,
                # Sit just outside the ring at angle=0 (+X axis)
                basePosition           = np.array([radius + notch_size * 0.75, 0, z_pos]) + WORLD_OFFSET,
            )

    def load_assets(self):
        self.model_lib = {}

        # object_sets/<object_set>/<object_name>/fused/<object_name>.urdf
        self.urdf_models = glob.glob(
            os.path.join(self.object_model_path, "**", "fused", "*.urdf"),
            recursive=True,
        )

        for urdf_path in self.urdf_models:
            fused_dir = os.path.dirname(urdf_path)
            object_dir = os.path.dirname(fused_dir)

            object_name = os.path.basename(object_dir)

            # canonical key
            self.model_lib[object_name] = urdf_path

            # Also allow lookup by URDF filename without extension
            urdf_key = os.path.splitext(os.path.basename(urdf_path))[0]
            self.model_lib[urdf_key] = urdf_path

        self.scene_layouts = glob.glob(os.path.join(self.scene_path, "*.npz"))

    def load_scene(self, scene_file):
        p.resetSimulation()
        # self.create_visual_only_bars()
        self.create_visual_only_circle(diameter=0.61,notch_size=None)
        self.create_visual_only_circle(diameter=0.45)
        data = np.load(os.path.join(self.scene_path, scene_file), allow_pickle=True)
        model_names = data["model_names"]
        poses = data["poses"]
        for model_name, pose in zip(model_names, poses):
            p.loadURDF(
                self.model_lib[model_name],
                basePosition=np.array(pose[:3]) + WORLD_OFFSET,
                baseOrientation=np.array(pose[3:]),
                useFixedBase=True,
            )

    def _compute_projection_matrix(self):
        self.fx = self.cam_K[0, 0]
        self.fy = self.cam_K[1, 1]
        self.cx = self.cam_K[0, 2]
        self.cy = self.cam_K[1, 2]
        left = -self.cx * self.near / self.fx
        right = (self.W - self.cx) * self.near / self.fx
        bottom = -(self.H - self.cy) * self.near / self.fy
        top = self.cy * self.near / self.fy
        P = p.computeProjectionMatrix(left, right, bottom, top, self.near, self.far)
        self.projection_matrix = P

    def _compute_view_matrix(self):
        # OpenCV cam -> OpenGL cam
        S = np.diag([1, -1, -1])
        R_cw_gl = S @ self.R_cw_cv
        t_cw_gl = S @ np.asarray(self.t_cw_cv, dtype=float).reshape(3)

        T_cw_gl = Rt_to_T(R_cw_gl, t_cw_gl)
        T_wc_gl = T_inv(T_cw_gl)

        R_wc = T_wc_gl[:3, :3]
        t_wc = T_wc_gl[:3, 3]
        eye = t_wc
        # camera -Z in world
        forward = -R_wc[:, 2]
        # +Y in world
        up = R_wc[:, 1]
        target = eye + forward

        self.view_matrix = p.computeViewMatrix(
            eye.tolist(), target.tolist(), up.tolist()
        )

    def render_scene_image(self):
        img = p.getCameraImage(
            self.W,
            self.H,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            shadow=0,
            renderer=PB_RENDER,
        )

        self.rgb = resize_rgba(img[2], self.H, self.W)
        self.seg = resize_seg(img[4], self.H, self.W)
        self.rgba = make_transparent(self.rgb, self.seg)
        return self.rgba

    def draw_apriltag_frame(self, rgba_img, axis_len=0.02):
        def draw_axes(img, K, rvec, tvec, axis_len=0.02):
            axes_3d = np.float32(
                [[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]
            )
            pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, np.zeros(5, dtype=float))
            p0, px, py, pz = pts.reshape(-1, 2).astype(int)
            cv2.line(img, tuple(p0), tuple(px), (255, 0, 0, 255), 2)  # X red
            cv2.line(img, tuple(p0), tuple(py), (0, 255, 0, 255), 2)  # Y green
            cv2.line(img, tuple(p0), tuple(pz), (0, 0, 255, 255), 2)  # Z blue

        self.scene_with_axis = rgba_img.copy()

        center = tuple(
            np.int32(getattr(self.det, "center", np.mean(self.corners, axis=0)))
        )

        cv2.putText(
            self.scene_with_axis,
            f"tag {self.tag_id}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0, 255),
            2,
        )
        rvec, _ = cv2.Rodrigues(self.R_cw_cv)
        tvec = self.t_cw_cv.reshape(3, 1)
        draw_axes(
            self.scene_with_axis, self.cam_K, rvec, tvec, axis_len=APRILTAG_SIZE * 0.6
        )
        return self.scene_with_axis
    
    def update_camera(self, R_cw_cv: np.ndarray, t_cw_cv: np.ndarray):
        """
        Update the view matrix with a new tag pose and re-render.
        Allows camera switching without reloading the scene.

        Args:
            R_cw_cv:  [3x3] rotation matrix (tag→camera, OpenCV convention, pre-flipped)
            t_cw_cv:  [3]   translation vector (tag position in camera frame)
        """
        self.R_cw_cv = R_cw_cv
        self.t_cw_cv = np.asarray(t_cw_cv, dtype=float).reshape(3)
        self._compute_view_matrix()

    def update_and_render(self, R_cw_cv: np.ndarray, t_cw_cv: np.ndarray) -> np.ndarray:
        """
        Convenience method: update camera pose and immediately return a fresh render.

        Args:
            R_cw_cv:  [3x3] rotation matrix
            t_cw_cv:  [3]   translation vector

        Returns:
            rgba: [H, W, 4] uint8 RGBA image
        """
        self.update_camera(R_cw_cv, t_cw_cv)
        return self.render_scene_image()