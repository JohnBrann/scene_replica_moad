"""
view_scene.py — Simple interactive PyBullet scene viewer

Loads a .npz scene file and opens a PyBullet GUI window for free exploration.

Usage:
    python view_scene.py <scene.npz>
    python view_scene.py                  # prompts for scene file

Mouse controls (PyBullet native):
    Left drag       orbit camera
    Right drag      pan camera
    Scroll          zoom
    Ctrl+drag       change pitch

Keyboard:
    Q / ESC         quit
"""

import sys
import os
import glob
import time
import numpy as np
from pathlib import Path

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not installed. Run: pip install pybullet")
    sys.exit(1)

# =============================================================================
# CONFIG — adjust paths to match your asset layout
# =============================================================================

OBJECT_MODEL_PATH = os.path.join("assets", "object_sets", "moad-atb1")
SCENE_PATH        = os.path.join("assets", "scenes")
WORLD_OFFSET      = np.array([0.0, 0.0, 0.0])  # match your scene_replica_cfg

# Initial camera view
CAMERA_DISTANCE   = 1.2     # metres from target
CAMERA_YAW        = 45.0    # degrees
CAMERA_PITCH      = -30.0   # degrees
CAMERA_TARGET     = [0, 0, 0]


# =============================================================================
# ASSET LOADING
# =============================================================================

def build_model_lib(object_model_path: str) -> dict:
    """Scan YCB object folder and build name → urdf_path mapping."""
    model_pattern = os.path.join(object_model_path, "*/fused/*.urdf")
    urdf_files = glob.glob(model_pattern)

    model_lib  = {}
    for urdf_path in urdf_files:
        folder = os.path.basename(os.path.dirname(os.path.dirname(urdf_path)))
        print(f"Found Model: {folder}")
        model_lib[folder] = urdf_path
        # Also map the name without leading numeric prefix (e.g. "004_sugar_box" → "sugar_box")
        if "_" in folder and folder.split("_", 1)[0].isdigit():
            short_key = folder.split("_", 1)[1]
            model_lib[short_key] = urdf_path
    return model_lib


def load_scene(scene_file: str, model_lib: dict) -> int:
    """Load all objects from npz into the current pybullet simulation. Returns object count."""
    data        = np.load(scene_file, allow_pickle=True)
    model_names = data["model_names"]
    poses       = data["poses"]
    count       = 0

    for model_name, pose in zip(model_names, poses):
        if model_name not in model_lib:
            print(f"  WARNING: model '{model_name}' not found in library, skipping.")
            continue
        urdf_path = model_lib[model_name]
        pos  = np.array(pose[:3]) + WORLD_OFFSET
        orn  = np.array(pose[3:])
        p.loadURDF(urdf_path, basePosition=pos, baseOrientation=orn, useFixedBase=True)
        count += 1

    return count


def add_reference_frame(size=0.1):
    """Draw XYZ axes at the world origin as thin coloured boxes."""
    half = size / 2
    th   = 0.004
    axes = [
        ([half, 0,    0   ], [half, th, th], [1, 0, 0, 1]),  # X red
        ([0,    half, 0   ], [th, half, th], [0, 1, 0, 1]),  # Y green
        ([0,    0,    half], [th, th, half], [0, 0, 1, 1]),  # Z blue
    ]
    for pos, ext, color in axes:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=color)
        p.createMultiBody(0, -1, vis, np.array(pos) + WORLD_OFFSET)


def add_ground_plane():
    """Add a subtle grey ground plane for visual reference."""
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[1.0, 1.0, 0.001],
        rgbaColor=[0.5, 0.5, 0.5, 0.3],
    )
    p.createMultiBody(0, -1, vis, [0, 0, -0.001])


# =============================================================================
# SCENE SELECTION
# =============================================================================

def pick_scene(scene_dir: str, arg: str | None) -> str | None:
    """Resolve scene file path from CLI arg or interactive prompt."""
    if arg:
        path = Path(arg)
        if path.exists():
            return str(path)
        # Try looking in SCENE_PATH
        candidate = Path(scene_dir) / arg
        if candidate.exists():
            return str(candidate)
        print(f"ERROR: Scene file not found: {arg}")
        return None

    # No arg — list available scenes and prompt
    scenes = sorted(glob.glob(os.path.join(scene_dir,"*", "scene_replica.npz")))
    if not scenes:
        print(f"No .npz scenes found in {scene_dir}")
        print("Usage: python view_scene.py <path_to_scene.npz>")
        return None

    print("\nAvailable scenes:")
    for i, s in enumerate(scenes):
        # print(f"  [{i+1:>2}] {Path(s).name}")
        print(f"  [{i+1:>2}] {Path(s)}")

    try:
        choice = int(input("\nEnter number (or 0 to cancel): "))
        if choice == 0:
            return None
        return scenes[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    # --- Resolve scene file ---
    arg        = sys.argv[1] if len(sys.argv) > 1 else None
    scene_file = pick_scene(SCENE_PATH, arg)
    if scene_file is None:
        sys.exit(0)

    print(f"\nLoading scene: {Path(scene_file).name}")

    # --- Build asset library ---
    model_lib = build_model_lib(OBJECT_MODEL_PATH)
    print(f"  Found {len(model_lib)} models in library.")

    # --- Start PyBullet GUI ---
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)      # hide default sliders

    # --- Load scene content ---
    p.resetSimulation()
    add_ground_plane()
    add_reference_frame(size=0.15)
    count = load_scene(scene_file, model_lib)
    print(f"  Loaded {count} objects.")

    # --- Set initial camera view ---
    p.resetDebugVisualizerCamera(
        cameraDistance    = CAMERA_DISTANCE,
        cameraYaw         = CAMERA_YAW,
        cameraPitch       = CAMERA_PITCH,
        cameraTargetPosition = CAMERA_TARGET,
    )

    print(f"\nScene ready. Use mouse to navigate.")
    print("Press Q or ESC in the PyBullet window to quit.\n")

    # --- Main loop ---
    try:
        while True:
            keys = p.getKeyboardEvents()

            # Q or ESC to quit
            if (ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED) or \
               (27 in keys and keys[27] & p.KEY_WAS_TRIGGERED):
                break

            p.stepSimulation()
            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        pass

    p.disconnect(client)
    print("Done.")


if __name__ == "__main__":
    main()