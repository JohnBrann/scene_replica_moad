# Implementation of the class for rendering real world scene layouts for grasping in clutters task

import os
import glob
import math

try:
    # import cv2
    import numpy as np
    import pybullet as p

except Exception as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and properly configured.")
    exit()

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


class TaglessSceneReplica:
    def __init__(
        self, cam_K, W, H, R_cw_cv, t_cw_cv, scene_config, scene_path=None, model_library_path=None
    ):  
        print("\n=== Initializing Scene Replica Object ===")
        # Initial state
        self.pb = p.connect(p.DIRECT)
        self.cam_K = cam_K
        self.W = W
        self.H = H
        self.R_cw_cv = R_cw_cv
        self.t_cw_cv = t_cw_cv

        # Configuration
        # self.config = scene_config
        if model_library_path is None:
            self.object_model_path = scene_config["model_library_dir"]
        else:
            self.object_model_path = model_library_path
        if scene_path is None:
            self.scene_path = scene_config["scenes_dir"]
        else:
            self.scene_path = scene_path
        self.near = scene_config["CAMERA_NEAR"] 
        self.far = scene_config["CAMERA_FAR"] 
        self.WORLD_OFFSET = scene_config["WORLD_OFFSET"]
        print(f"WORLD_OFFSET = {self.WORLD_OFFSET}")
        self.PB_RENDER = scene_config["PB_RENDER"]
        self.render_alpha = scene_config["render_alpha"]

        self.viz_rings = scene_config["visualize_rings"]
        self.viz_rotation = scene_config["visualize_rotation"]
        self.viz_floor = scene_config["visualize_floor"]
        self.viz_center = scene_config["visualize_center"]

        # Internal states
        self.model_lib = {}
        self.projection_matrix = None
        self.view_matrix = None
        self.urdf_models = []
        self.scene_layouts = []
        self.load_assets()
        self._compute_projection_matrix()
        self._compute_view_matrix()

        # Print self.
        print(f"Attributes: ")
        print_len_only = ['urdf_models','model_lib']
        for k, v in self.__dict__.items():
            if k in print_len_only:
                print(f" > {k}: [{len(v)} items]")
            else:
                print(f" > {k}: {v}")

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
            basePosition=np.array([0, +half_length, z_pos])+ self.WORLD_OFFSET
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bar_visual_x,
            baseCollisionShapeIndex=-1,
            basePosition=np.array([0, -half_length, z_pos])+ self.WORLD_OFFSET
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
            basePosition=np.array([+half_length, 0, z_pos])+ self.WORLD_OFFSET
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bar_visual_y,
            baseCollisionShapeIndex=-1,
            basePosition=np.array([-half_length, 0, z_pos])+ self.WORLD_OFFSET
        )

    def create_visual_only_circle(self, diameter: float = 0.5):
        """
        Creates a visual-only circle (approximated by thin box segments) with a
        small notch box indicating the zero rotational position (along +X axis).

        Args:
            diameter:   diameter of the circle in metres
            notch_size: side length of the notch box in metres
        """
        radius     = diameter / 2.0
        thickness  = 0.002          # bar cross-section
        z_pos      = thickness / 2.0
        num_segs   = 32             # more segments = smoother circle
        color_ring = [1, 0, 0, 1]  # red ring

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
                basePosition           = np.array([cx, cy, z_pos]),# + self.WORLD_OFFSET,
                baseOrientation        = quat,
            )

        

    def create_visual_only_rotation_indicator(self,dist_from_center=0.2,length=0.2,thickness=0.005):
        color_notch= [1, 0.5, 0, 1]  # orange notch
        z_pos      = thickness / 2.0
        # --- Notch box at +X (zero rotation position) ---
        notch_vis = p.createVisualShape(
            shapeType  = p.GEOM_BOX,
            halfExtents= [length / 2.0, thickness / 2.0, thickness / 2.0],
            rgbaColor  = color_notch,
        )
        p.createMultiBody(
            baseMass               = 0,
            baseVisualShapeIndex   = notch_vis,
            baseCollisionShapeIndex= -1,
            # Positioned at angle=0 (+X axis)
            basePosition           = np.array([dist_from_center + (length / 2.0), 0, z_pos]),# + self.WORLD_OFFSET,
        )
        
    def create_visual_only_floor(self, width: float = 0.5, thickness: float = 0.001, color = [1, 0, 0, 0.8]):
        z_pos      = thickness / 2.0
        half_extents1 = [width / 2.0, width / 2.0, thickness / 2.0]

        vis = p.createVisualShape(
            shapeType  = p.GEOM_BOX,
            halfExtents= half_extents1,
            rgbaColor  = color,
        )
        half_extents2 = [width / 20.0, width / 20.0, thickness / 2.0]

        vis2 = p.createVisualShape(
            shapeType  = p.GEOM_BOX,
            halfExtents= half_extents2,
            rgbaColor  = [0,1,0,0.5],
        )
        quat = p.getQuaternionFromEuler([0, 0, 0])
        p.createMultiBody(
                baseMass               = 0,
                baseVisualShapeIndex   = vis,
                baseCollisionShapeIndex= -1,
                basePosition           = np.array([0, 0, z_pos]) + self.WORLD_OFFSET, #TODO: Apply calibration offset?
                baseOrientation        = quat,
            )
        p.createMultiBody(
                baseMass               = 0,
                baseVisualShapeIndex   = vis2,
                baseCollisionShapeIndex= -1,
                basePosition           = np.array([0, 0, z_pos+0.0001]) + self.WORLD_OFFSET, #TODO: Apply calibration offset?
                baseOrientation        = quat,
            )
        
    def create_visual_only_center_post(self,height=0.2,thickness=0.002):
        color_notch= [1, 0.5, 0, 1]  # orange post
        z_pos      = height / 2.0
        # --- Center post showing +Z (zero rotation position) ---
        notch_vis = p.createVisualShape(
            shapeType  = p.GEOM_BOX,
            halfExtents= [thickness / 2.0, thickness / 2.0, height / 2.0],
            rgbaColor  = color_notch,
        )
        p.createMultiBody(
            baseMass               = 0,
            baseVisualShapeIndex   = notch_vis,
            baseCollisionShapeIndex= -1,
            # Positioned at center (+Z axis)
            basePosition           = np.array([0, 0, z_pos]),# + self.WORLD_OFFSET,
        )
        
    def load_assets(self, verbose=True):
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
        if verbose: print(f"Found {len(self.model_lib.keys())} object model urdf files...")

        self.scene_layouts = glob.glob(os.path.join(self.scene_path, "*.npz"))
        if verbose: print(f"Found {len(self.scene_layouts)} scene layout npz files...")

    def load_scene(self, scene_file, verbose=True):
        if verbose: print(f"Loading Scene: \"{scene_file}\"...")
        p.resetSimulation()
        # Create Visual Markers
        # self.create_visual_only_bars()
        if self.viz_rings:
            self.create_visual_only_circle(diameter=0.6)
            self.create_visual_only_circle(diameter=0.45) # TODO: Separate out rotation indicator to separate function
        if self.viz_rotation:
            self.create_visual_only_rotation_indicator(dist_from_center=0.1,length=0.25)
        if self.viz_floor:
            self.create_visual_only_floor(width=0.33,color=[0,0,1,0.5])
        if self.viz_center:
            self.create_visual_only_center_post(height=0.2)

        # Load Scene Objects
        data = np.load(os.path.join(self.scene_path, scene_file), allow_pickle=True)
        model_names = data["model_names"]
        poses = data["poses"]
        for model_name, pose in zip(model_names, poses):
            p.loadURDF(
                self.model_lib[model_name],
                basePosition=np.array(pose[:3]) + self.WORLD_OFFSET,
                baseOrientation=np.array(pose[3:]),
                useFixedBase=True,
            )
        

        if verbose: self.print_scene_objects()

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
        RENDERER = getattr(p, self.PB_RENDER)
        img = p.getCameraImage(
            self.W,
            self.H,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            shadow=0,
            renderer=RENDERER,
        )

        self.rgb = resize_rgba(img[2], self.H, self.W)
        self.seg = resize_seg(img[4], self.H, self.W)
        self.rgba = make_transparent(self.rgb, self.seg, alpha=self.render_alpha)
        return self.rgba

    
    
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
    

    def print_scene_objects(self, physics_client: int = 0) -> None:
        """
        Print the position and orientation of every body currently loaded in a
        PyBullet simulation, formatted for easy reading.

        Args:
            physics_client: PyBullet physics client ID (default 0).
        """
        num_bodies = p.getNumBodies(physicsClientId=physics_client)

        print("\n" + "═" * 64)
        print(f"  SCENE OBJECTS  ({num_bodies} bodies total, ignoring debug visuals)")
        print("═" * 64)

        for i in range(num_bodies):
            body_id   = p.getBodyUniqueId(i, physicsClientId=physics_client)
            body_name = p.getBodyInfo(body_id, physicsClientId=physics_client)[1].decode("utf-8")
            pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
            if body_name == '': continue
            # Convert quaternion (x,y,z,w) → Euler angles in degrees for readability
            euler_rad = p.getEulerFromQuaternion(quat)
            euler_deg = tuple(math.degrees(a) for a in euler_rad)

            print(f"\n  [{i}]  '{body_name}'  (id={body_id})")
            print(f"       Position  :  x={pos[0]:+.4f}   y={pos[1]:+.4f}   z={pos[2]:+.4f}")
            print(f"       Quaternion:  x={quat[0]:+.4f}   y={quat[1]:+.4f}   z={quat[2]:+.4f}   w={quat[3]:+.4f}")
            print(f"       Euler (°) :  r={euler_deg[0]:+.2f}   p={euler_deg[1]:+.2f}   y={euler_deg[2]:+.2f}")

        print("\n" + "═" * 64 + "\n")