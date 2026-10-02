#!/usr/bin/env python3
"""
moad_to_bop.py
--------------
Converts one MOAD scene/pose (with generated annotations) into BOP-scenewise
format, ready for direct use with the BOP toolkit.

USAGE
    python3 moad_to_bop.py \\
        --data-root  /home/csrobot/MOAD_DATA \\
        --object     batch1_007 \\
        --pose       pose-a \\
        --calib-root /home/csrobot/moad_control/moad_cui/calibration/55mm_joint \\
        --bop-root   /home/csrobot/moad_eval/bop_data \\
        --model-dir  /home/csrobot/moad_eval/assets/object_sets/moad-atb1

OUTPUT STRUCTURE
    <bop-root>/moad/                  ← dataset root (name must match CSV filename)
    ├── camera.json                   # reference camera params (DSLR, for rendering)
    ├── dataset_info.json             # dataset metadata
    ├── test_targets_bop19.json       # accumulated evaluation targets
    ├── models_eval/                  # BOP toolkit expects models here
    │   ├── models_info.json          # diameter + symmetries per object
    │   └── obj_000001.ply            # pre-converted PLY files
    ├── meta/                         # MOAD-specific persistent indices
    │   ├── object_index.json         # name → BOP id mapping (grows over runs)
    │   └── scene_index.json          # object/pose → scene_id mapping
    └── test/                         # BOP split folder (toolkit looks for "test")
        └── <scene_id:06d>/           # e.g. 000001
            ├── rgb_dslr/             # DSLR colour images (000001.png …)
            ├── depth_dslr/           # DSLR depth images (uint16, mm) — optional
            ├── depth_dslr_info.json  # provenance of depth_dslr/ (source, model)
            ├── scene_camera_dslr.json  # DSLR intrinsics per image
            ├── scene_gt_dslr.json    # GT poses relative to DSLR cameras
            ├── scene_gt_realsense.json  # GT poses relative to RS cameras
            ├── scene_gt_info.json    # visibility info for DSLR images
            ├── rgb_realsense/        # RS colour images
            ├── depth_realsense/      # RS depth images (uint16, mm)
            ├── scene_camera_realsense.json
            └── scene_gt_info_realsense.json

MODALITY CONVENTIONS
    DSLR        : image IDs 1…360, written to rgb_dslr/ and scene_camera_dslr.json
    RealSense   : image IDs 1…360, written to rgb_realsense/, depth_realsense/
                  and scene_camera_realsense.json  (independent ID space)
    Each sensor's GT poses are expressed in that sensor's camera frame —
    scene_gt_dslr.json and scene_gt_realsense.json contain different R/t
    values for the same physical object because each camera has different
    extrinsics. Targets files are also per-sensor (test_targets_moad_dslr.json
    and test_targets_moad_realsense.json) since visibility fractions differ.

DSLR DEPTH (optional)
    MOAD DSLR frames have no native depth; depth aligned to the DSLR views is
    generated separately (RealSense reprojection, COLMAP stereo, or Depth
    Anything 3) into a per-scan folder such as DSLR_depth/ or DSLR_depth_da3/.
    Pass --dslr-depth-subdir to copy one of those into depth_dslr/.

    The folder name stays depth_dslr/ whichever generator produced it; the
    provenance is recorded in depth_dslr_info.json (source dir plus the
    generator's own meta.json: backend, model, geometry, config hash), because
    BOP itself has no field for it and an unattributed depth folder cannot be
    reproduced later.

    Depth is matched to colour by FILE STEM (frame_00001.jpg -> frame_00001.png),
    never by independent directory listing order: a single missing depth frame
    would otherwise shift every subsequent image_id and pair depth with the
    wrong pose.

UNITS
    All translation vectors are in MILLIMETRES (BOP convention).
    Source annotations are in metres — the script multiplies t by 1000.
    Depth images are uint16 millimetres for both sensors, so depth_scale = 1.0
    (BOP: pixel_value * depth_scale = millimetres).

MODEL UNITS  (a silent-failure trap — read before changing)
    BOP is millimetres EVERYWHERE: model vertices, cam_t_m2c, models_info
    diameters, depth images and the MSSD/MSPD thresholds. MOAD source meshes
    are authored in METRES, so they are scaled on the way into models_eval/
    (--model-units, default 'meters').

    A metre-scale model does NOT fail loudly:
      * MSSD/MSPD on a GT-vs-GT run still score 1.0, because prediction ==
        ground truth gives zero distance whatever the units are;
      * a real estimator's MSSD errors come out 1000x too small, so every
        pose passes the 2-20 mm thresholds and scores look excellent;
      * only VSD notices, because it alone compares the rendered model
        against measured depth in mm — hence "VSD ~ 0 while MSSD/MSPD = 1.0",
        which is the signature of this bug.

    Scaling the vertices is not enough on its own: models_info diameters must
    be recomputed from the SCALED mesh (VSD normalises tau by diameter), and
    any symmetries_discrete translations must be scaled by the same factor.
    All three are handled together in copy_and_scale_ply()/build_models_info_entry().

MODEL FRAME  (orientation of the BOP meshes vs. the annotations)
    The scene annotations (R, t per object) were produced against the meshes
    used for scene replication, which are Y-UP. The source meshes in
    --model-dir are Z-UP. Same object, different model frame, so applying an
    annotated pose to an unrotated source mesh is off by a fixed rotation.

    --model-frame-rot declares R_fix, the rotation taking source-mesh
    coordinates into the annotation frame:  v_annot = R_fix @ v_source.
    It is applied to the MESH as it is copied into models_eval/, alongside
    the unit scale, so the BOP meshes share the annotation frame and every
    pose is written exactly as annotated:

        models_eval vertex = R_fix @ (scale * v_source)
        cam_R_m2c, cam_t_m2c = the annotation, unchanged

    One frame then holds everywhere: annotations, scene replication, BOP
    evaluation and every pose-estimator wrapper that loads models_eval/.

    The rotation is about the MODEL ORIGIN (CloudCompare's
    Edit > Apply Transformation; its interactive rotate tool pivots about
    the bounding-box centre instead).

    CONSEQUENCE: symmetries in models_info.json are defined in the
    models_eval frame, so they must be measured on (or transformed into) the
    ROTATED meshes. Changing --model-frame-rot invalidates them.

PERSISTENT CONVERSION SETTINGS
    Scenes accumulate in one dataset across many runs. Settings that change
    what the files MEAN (model units, model-frame rotation) are recorded in
    meta/conversion_settings.json on first use, and every later run must
    match them — otherwise scenes converted with and without a correction
    would sit side by side with nothing in the files to tell them apart.

INTRINSICS
    cam_K in scene_camera_*.json describes the RESOLUTION OF THE IMAGES ACTUALLY
    WRITTEN. DSLR calibration is stored at full resolution (6000x4000) while the
    converted images come from images_4 (1500x1000), so the intrinsics are
    rescaled to the real image size — see scale_intrinsics().

OBJECT IDs
    Maintained in meta/object_index.json. New object names encountered during
    conversion are assigned the next available integer ID. IDs are stable once
    assigned — re-running on a different scene will not change existing mappings.

SCENE IDs
    Maintained in meta/scene_index.json. Each (object, pose) pair that has not
    been converted before is assigned the next available 6-digit integer ID.
"""

import os
import re
import sys
import glob
import json
import shutil
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

METRES_TO_MM = 1000.0   # BOP stores translations in millimetres


# ---------------------------------------------------------------------------
# Persistent index helpers
# ---------------------------------------------------------------------------

def load_json(path: str, default) -> dict:
    """Load a JSON file or return default if it doesn't exist."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_or_assign_object_id(
    object_index: dict,
    object_name: str,
) -> int:
    """
    Return the BOP integer ID for object_name, creating a new entry if needed.
    object_index is modified in-place.
    """
    if object_name not in object_index:
        next_id = max(object_index.values(), default=0) + 1
        object_index[object_name] = next_id
        print(f"  [NEW OBJ] '{object_name}' → obj_id={next_id}")
    return object_index[object_name]


def get_or_assign_scene_id(
    scene_index: dict,
    scene_key: str,
) -> int:
    """
    Return the BOP 6-digit scene ID for scene_key (e.g. 'batch1_007/pose-a'),
    creating a new entry if needed. scene_index is modified in-place.
    """
    if scene_key not in scene_index:
        next_id = max(scene_index.values(), default=0) + 1
        scene_index[scene_key] = next_id
        print(f"  [NEW SCENE] '{scene_key}' → scene_id={next_id:06d}")
    return scene_index[scene_key]


# ---------------------------------------------------------------------------
# models_info.json helpers
# ---------------------------------------------------------------------------

# Plausible extents (mm) for a MOAD turntable object: roughly 1 cm to 1 m.
# Used only to catch unit mistakes, not to validate geometry.
PLAUSIBLE_EXTENT_MM = (10.0, 1000.0)


def read_ply_vertices(ply_path: str) -> np.ndarray:
    """Vertex positions of a PLY as (N,3), via trimesh/open3d, else ASCII parse."""
    try:
        import trimesh
        mesh = trimesh.load(ply_path, process=False)
        return np.asarray(mesh.vertices, dtype=np.float64)
    except Exception:
        pass
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_triangle_mesh(ply_path).vertices, dtype=np.float64)
    except Exception:
        pass
    verts, in_vertex = [], False
    with open(ply_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "end_header":
                in_vertex = True
                continue
            if in_vertex:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        verts.append([float(x) for x in parts[:3]])
                    except ValueError:
                        pass
    return np.array(verts, dtype=np.float64)


def describe_extent(verts: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis-aligned bounding box size and its diagonal length."""
    if len(verts) == 0:
        return np.zeros(3), 0.0
    size = verts.max(axis=0) - verts.min(axis=0)
    return size, float(np.linalg.norm(size))


def check_extent_units(diag_mm: float, obj_name: str) -> None:
    """
    Warn loudly when a converted model's size is implausible for millimetres.

    A metre-scale mesh that slipped through shows up here as a diagonal of a
    fraction of a millimetre; a double-scaled one as tens of metres.
    """
    lo, hi = PLAUSIBLE_EXTENT_MM
    if diag_mm < lo or diag_mm > hi:
        guess = ("looks like METRES that were not scaled" if diag_mm < lo else
                 "looks like it was scaled twice, or is not a single object")
        print(f"  {'!' * 74}")
        print(f"  !! IMPLAUSIBLE MODEL SIZE for '{obj_name}': bbox diagonal "
              f"{diag_mm:.3f} mm")
        print(f"  !! expected {lo:.0f}-{hi:.0f} mm for a MOAD object — {guess}.")
        print(f"  !! BOP requires MILLIMETRES. VSD will score ~0 while "
              f"MSSD/MSPD still look perfect.")
        print(f"  {'!' * 74}")


def copy_and_scale_ply(src: str, dst: str, scale: float, obj_name: str,
                       R_fix: np.ndarray | None = None) -> bool:
    """
    Copy a source mesh into models_eval/, converting units to mm and
    rotating it into the annotation frame:  v = R_fix @ (scale * v_source).

    With scale 1 and no rotation the file is copied byte-for-byte (no
    re-encode, preserving colours/normals exactly). Otherwise the mesh is
    loaded, transformed about the origin, and written back out.
    """
    rotate = R_fix is not None and not np.allclose(R_fix, np.eye(3))
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    src_verts = read_ply_vertices(src)
    src_size, src_diag = describe_extent(src_verts)

    if scale == 1.0 and not rotate:
        shutil.copy2(src, dst)
    else:
        try:
            import trimesh
            mesh = trimesh.load(src, process=False)
            T = np.eye(4)
            T[:3, :3] = (R_fix if rotate else np.eye(3)) * scale
            mesh.apply_transform(T)          # scale + rotate about the origin
            mesh.export(dst)
        except Exception as e:
            print(f"  [ERROR] could not rescale {src}: {type(e).__name__}: {e}")
            print(f"          install trimesh, or pre-scale the mesh to mm")
            return False

    dst_verts = read_ply_vertices(dst)
    dst_size, dst_diag = describe_extent(dst_verts)

    print(f"  [MODEL SCALE] {obj_name}")
    print(f"      source : {src}")
    print(f"      factor : x{scale:g}   "
          f"({'copied unchanged' if scale == 1.0 and not rotate else 'vertices transformed'})")
    if rotate:
        print(f"      rotate : R_fix into annotation frame "
              f"(source +Z -> {np.round(R_fix @ [0, 0, 1], 3).tolist()})")
    print(f"      bbox   : {src_size[0]:.4g} x {src_size[1]:.4g} x {src_size[2]:.4g} "
          f"(diag {src_diag:.4g})  ->  "
          f"{dst_size[0]:.1f} x {dst_size[1]:.1f} x {dst_size[2]:.1f} mm "
          f"(diag {dst_diag:.1f} mm)")
    check_extent_units(dst_diag, obj_name)
    return True


def object_diameter_from_ply(ply_path: str) -> float:
    """
    Approximate the BOP diameter (largest pairwise vertex distance) for a PLY
    file by reading vertex positions and computing the max distance between
    any two vertices. For large meshes we subsample to keep it fast.

    Call this on the CONVERTED mesh in models_eval/ (already scaled to mm),
    never on the metre-scale source: VSD normalises tau by this diameter, so a
    metre-scale diameter breaks VSD independently of the vertex scaling.
    """
    verts = read_ply_vertices(ply_path)

    if len(verts) == 0:
        print(f"  [WARN] No vertices found in {ply_path}, diameter=0")
        return 0.0

    # Subsample for speed
    if len(verts) > 5000:
        idx   = np.random.choice(len(verts), 5000, replace=False)
        verts = verts[idx]

    # Max pairwise distance via broadcasting on subsampled set
    diff = verts[:, None, :] - verts[None, :, :]   # (N, N, 3)
    diam = float(np.sqrt((diff ** 2).sum(axis=-1)).max())
    return diam


def build_models_info_entry(
    obj_id:   int,
    obj_name: str,
    ply_path: str,
) -> dict:
    """
    Build a models_info.json entry for one object.
    Symmetry fields are templated as empty — fill in manually.
    """
    diam = object_diameter_from_ply(ply_path) if os.path.isfile(ply_path) else 0.0
    if diam > 0:
        check_extent_units(diam, obj_name)

    return {
        "obj_id":   obj_id,
        "obj_name": obj_name,
        "diameter": round(diam, 4),
        # ── Symmetry fields — fill in manually ──────────────────────────────
        # discrete: list of {"R": [[...]], "t": [0,0,0]} dicts (one per
        #           non-identity symmetry transform).  Leave [] if none.
        # continuous: list of {"axis": [ax, ay, az], "offset": [0,0,0]} dicts.
        #             Leave [] if no continuous symmetry axis.
        "symmetries_discrete":   [],
        "symmetries_continuous": [],
    }


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def load_annotation_files(annot_dir: str) -> dict[int, dict]:
    """
    Load all annotation JSON files from annot_dir.
    Returns a dict mapping image_id (1-based int) → annotation dict.
    Files are sorted and assigned IDs in sorted order.
    """
    paths = sorted(glob.glob(os.path.join(annot_dir, "*.json")))
    result = {}
    for image_id, path in enumerate(paths, start=1):
        with open(path) as f:
            result[image_id] = json.load(f)
    return result


# ---------------------------------------------------------------------------
# BOP JSON builders
# ---------------------------------------------------------------------------

def scale_intrinsics(intr: dict, out_w: int, out_h: int) -> dict:
    """
    Rescale calibration intrinsics to the resolution of the images actually
    written into the BOP scene.

    The DSLR calibration is expressed at full sensor resolution (6000x4000)
    but the converted images come from images_4 (1500x1000); cam_K must
    describe the grid the pixels are on, or every consumer that projects
    through it (VSD rendering, any pose estimator) is wrong by the scale
    factor. Pixel-centre aware: the centre of pixel 0 sits at 0, hence
    (c + 0.5) * s - 0.5.
    """
    sx = out_w / float(intr["width"])
    sy = out_h / float(intr["height"])
    return {
        "fx":     intr["fx"] * sx,
        "fy":     intr["fy"] * sy,
        "cx":     (intr["cx"] + 0.5) * sx - 0.5,
        "cy":     (intr["cy"] + 0.5) * sy - 0.5,
        "width":  int(out_w),
        "height": int(out_h),
    }


def image_size(path: str) -> tuple[int, int] | None:
    """(width, height) of an image on disk, or None if unreadable."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def build_scene_camera_entry(
    intrinsics: dict,
    depth_scale: float = 1.0,
    include_depth_scale: bool = True,
) -> dict:
    """
    Build one entry for scene_camera.json from an intrinsics dict.
    cam_K is stored row-wise as a flat 9-element list (BOP convention).
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    entry = {
        "cam_K": [
            fx,  0.0, cx,
            0.0, fy,  cy,
            0.0, 0.0, 1.0,
        ],
    }
    if include_depth_scale:
        entry["depth_scale"] = depth_scale
    return entry


def parse_frame_rotation(spec: str | None) -> np.ndarray:
    """
    Parse a model-frame rotation spec into a 3x3 matrix.

    Format: comma-separated "axis:degrees" terms applied in order, e.g.
        "x:-90"          single rotation about x
        "x:-90,z:180"    rotate about x, then about z
    "none" / "" / None returns the identity.

    Rotations are about the MODEL ORIGIN, matching CloudCompare's
    Edit > Apply Transformation.
    """
    R = np.eye(3)
    if not spec or spec.strip().lower() in ("none", "identity", "0"):
        return R
    for term in spec.split(","):
        try:
            axis, deg = term.strip().lower().split(":")
            a = np.deg2rad(float(deg))
        except ValueError:
            raise ValueError(f"bad --model-frame-rot term {term!r}; use e.g. 'x:-90'")
        c, s_ = np.cos(a), np.sin(a)
        Ri = {"x": np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]),
              "y": np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]),
              "z": np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])}.get(axis)
        if Ri is None:
            raise ValueError(f"bad axis {axis!r} in --model-frame-rot; use x, y or z")
        R = Ri @ R                     # each term applied after the previous
    R[np.abs(R) < 1e-12] = 0.0         # exact zeros for clean JSON
    return R


def check_conversion_settings(meta_dir: str, settings: dict, force: bool) -> None:
    """
    Record conversion settings on first use; require later runs to match.

    These settings change what the stored poses and meshes MEAN, so a dataset
    must be converted with one consistent set. Mismatch is an error unless
    --force-settings is given (e.g. when deliberately reconverting everything).
    """
    path = os.path.join(meta_dir, "conversion_settings.json")
    existing = load_json(path, None)
    if existing is None:
        save_json(path, settings)
        print(f"  conversion settings recorded: {path}")
        return
    diffs = [k for k in settings if existing.get(k) != settings[k]]
    if not diffs:
        return
    print(f"\n  {'!' * 74}")
    print(f"  !! CONVERSION SETTINGS DIFFER from those this dataset was built with:")
    for k in diffs:
        print(f"  !!   {k}: dataset {existing.get(k)!r}  vs  this run {settings[k]!r}")
    print(f"  !! Mixing them leaves scenes whose poses mean different things.")
    if not force:
        print(f"  !! Re-run with matching flags, or pass --force-settings after "
              f"deleting the old scenes.")
        print(f"  {'!' * 74}")
        sys.exit(1)
    print(f"  !! --force-settings given: overwriting recorded settings.")
    print(f"  {'!' * 74}")
    save_json(path, settings)


def build_scene_gt_entries(
    annotation:    dict,
    object_index:  dict,
) -> list[dict]:
    """
    Build the list of GT entries for one image in scene_gt.json.
    Translation is converted from metres to millimetres.
    R is stored row-wise as a flat 9-element list.

    Poses are written exactly as annotated: the meshes in models_eval/ are
    rotated into the annotation frame instead (see MODEL FRAME above).
    """
    entries = []
    for obj in annotation.get("objects", []):
        obj_name = obj.get("object_name", "")
        if not obj_name:
            continue
        obj_id = object_index.get(obj_name)
        if obj_id is None:
            print(f"  [WARN] Object '{obj_name}' not in object_index, skipping")
            continue

        R = np.array(obj["R"], dtype=np.float64)
        t = np.array(obj["t"], dtype=np.float64) * METRES_TO_MM

        entries.append({
            "obj_id":    obj_id,
            "cam_R_m2c": R.flatten().tolist(),
            "cam_t_m2c": t.tolist(),
        })
    return entries


def build_scene_gt_info_entries(annotation: dict) -> list[dict]:
    """
    Build the list of GT info entries for one image in scene_gt_info.json.
    Requires that the annotation has been generated with visibility fields
    (px_count_all, px_count_visib, visib_fract, bbox_obj, bbox_visib).
    """
    entries = []
    for obj in annotation.get("objects", []):
        entry = {
            "bbox_obj":       obj.get("bbox_obj",       [0, 0, 0, 0]),
            "bbox_visib":     obj.get("bbox_visib",     [0, 0, 0, 0]),
            "px_count_all":   obj.get("px_count_all",   0),
            "px_count_visib": obj.get("px_count_visib", 0),
            "visib_fract":    obj.get("visib_fract",    0.0),
        }
        # BOP also expects px_count_valid (pixels with valid depth); for
        # DSLR (no depth) we set it equal to px_count_visib. For RS, the
        # caller can override this by post-processing the depth image.
        entry["px_count_valid"] = obj.get("px_count_valid",
                                          entry["px_count_visib"])
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# test_targets_bop19.json builder
# ---------------------------------------------------------------------------

def build_test_targets(
    scene_id:      int,
    annotations:   dict[int, dict],
    object_index:  dict,
    visib_thresh:  float = 0.1,
) -> list[dict]:
    """
    Build test_targets_bop19.json entries for one scene.
    Only instances with visib_fract >= visib_thresh are included.
    inst_count is the number of qualifying instances of each object per image.
    """
    targets = []
    for image_id, ann in annotations.items():
        # Count qualifying instances per object in this image
        instance_counts: dict[int, int] = {}
        for obj in ann.get("objects", []):
            obj_name = obj.get("object_name", "")
            obj_id   = object_index.get(obj_name)
            if obj_id is None:
                continue
            vf = obj.get("visib_fract", 1.0)
            if vf < visib_thresh:
                continue
            instance_counts[obj_id] = instance_counts.get(obj_id, 0) + 1

        for obj_id, count in instance_counts.items():
            targets.append({
                "scene_id":   scene_id,
                "im_id":      image_id,
                "obj_id":     obj_id,
                "inst_count": count,
            })
    return targets


# ---------------------------------------------------------------------------
# Image copy helpers
# ---------------------------------------------------------------------------

def copy_as_png(src: str, dst: str) -> bool:
    """
    Copy an image file to dst, converting to PNG if needed.
    Returns True on success.
    """
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.lower().endswith(".png"):
        shutil.copy2(src, dst)
    else:
        # Convert via OpenCV for non-PNG sources (e.g. DSLR JPEGs)
        import cv2
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        cv2.imwrite(dst, img)
    return True


def image_id_str(image_id: int) -> str:
    """BOP image filename stem: zero-padded 6-digit integer."""
    return f"{image_id:06d}"


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_scene(args) -> None:
    """
    Convert one MOAD scene/pose to BOP format.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    pose_path    = os.path.join(args.data_root, args.object, args.pose)

    # BOP dataset root — the toolkit looks for <bop_root>/<dataset_name>/ as the dataset
    dataset_dir  = os.path.join(args.bop_root, args.dataset_name)

    # models/ at dataset root — where the BOP toolkit expects PLY files and models_info.json
    models_dir   = os.path.join(dataset_dir, "models_eval")

    # meta/ for MOAD-specific persistent indices (not consumed by BOP toolkit)
    meta_dir     = os.path.join(dataset_dir, "meta")

    # test/ is the BOP split folder name the toolkit looks for
    test_dir     = os.path.join(dataset_dir, "test")

    # Annotation paths
    dslr_annot_dir = os.path.join(pose_path, "scene_replica", "pose_dslr")
    rs_annot_dir   = os.path.join(pose_path, "scene_replica", "pose_realsense")
    # Source data paths
    dslr_img_dir   = os.path.join(pose_path, "images_4")
    # Optional generated depth aligned to the DSLR views (see DSLR DEPTH above)
    dslr_depth_dir = (os.path.join(pose_path, args.dslr_depth_subdir)
                      if args.dslr_depth_subdir else None)
    rs_img_dir     = os.path.join(pose_path, "realsense")
    # Sensor calibration paths
    dslr_calib_path = os.path.join(args.calib_root, "cam_parameters.json")
    rs_calib_path   = os.path.join(args.calib_root, "realsense_cam_parameters.json")

    for d in [dataset_dir, models_dir, meta_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Model frame correction + persistent settings check ───────────────────
    R_fix = parse_frame_rotation(args.model_frame_rot)
    rot_label = args.model_frame_rot or "none"
    print(f"\n  {'=' * 74}")
    print(f"  MODEL FRAME: source meshes rotated into the annotation frame by "
          f"R_fix = {rot_label}")
    print(f"  (poses are written exactly as annotated; models_eval/ meshes carry "
          f"the rotation)")
    if np.allclose(R_fix, np.eye(3)):
        print(f"  (identity: meshes keep their source orientation)")
    else:
        for row in R_fix:
            print(f"      [{row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}]")
    print(f"  {'=' * 74}")
    check_conversion_settings(meta_dir, {
        "model_units":     args.model_units,
        "model_frame_rot": rot_label,
        "R_fix":           [round(float(v), 9) for v in R_fix.ravel()],
        # What R_fix is applied to. An earlier version rotated the POSES with
        # the same flag value; recording this keeps the two from being mixed.
        "frame_rot_applies_to": "mesh",
    }, args.force_settings)

    # ── Load / update persistent indices ─────────────────────────────────────
    obj_index_path   = os.path.join(meta_dir, "object_index.json")
    scene_index_path = os.path.join(meta_dir, "scene_index.json")
    models_info_path = os.path.join(models_dir, "models_info.json")

    object_index = load_json(obj_index_path, {})
    scene_index  = load_json(scene_index_path, {})
    models_info  = load_json(models_info_path, {})

    scene_key = f"{args.object}/{args.pose}"
    scene_id  = get_or_assign_scene_id(scene_index, scene_key)
    scene_dir = os.path.join(test_dir, f"{scene_id:06d}")

    print(f"\n  Scene key : {scene_key}")
    print(f"  Scene ID  : {scene_id:06d}")
    print(f"  Scene dir : {scene_dir}\n")

    # ── Determine which modalities are available ───────────────────────────────
    has_dslr = (os.path.isdir(dslr_annot_dir) and
                os.path.isfile(dslr_calib_path) and
                os.path.isdir(dslr_img_dir))

    has_rs   = (os.path.isdir(rs_annot_dir) and
                os.path.isfile(rs_calib_path) and
                os.path.isdir(rs_img_dir))

    if not has_dslr and not has_rs:
        print("[ERROR] No annotation data found for either modality.")
        print(f"        DSLR annot: {dslr_annot_dir}")
        print(f"        RS annot  : {rs_annot_dir}")
        sys.exit(1)

    # DSLR depth is optional: absent folder means this scene is converted
    # colour-only, exactly as before this feature existed.
    has_dslr_depth = False
    if has_dslr and dslr_depth_dir:
        if os.path.isdir(dslr_depth_dir):
            has_dslr_depth = True
        else:
            print(f"[ERROR] --dslr-depth-subdir given but not found: {dslr_depth_dir}")
            sys.exit(1)

    print(f"  Modalities: "
          f"{'DSLR ' if has_dslr else ''}{'RealSense' if has_rs else ''}")
    if has_dslr:
        print(f"  DSLR depth: "
              f"{dslr_depth_dir if has_dslr_depth else 'none (colour only)'}")

    # ── Load calibration files ────────────────────────────────────────────────
    dslr_cal = None
    rs_cal   = None

    if has_dslr:
        with open(dslr_calib_path) as f:
            dslr_cal = json.load(f)

    if has_rs:
        with open(rs_calib_path) as f:
            rs_cal = json.load(f)

    # ── Load all annotation files ─────────────────────────────────────────────
    # Each sensor is loaded independently. GT poses and targets are built
    # separately from each since cam_R_m2c / cam_t_m2c are in each sensor's
    # camera frame, and visibility fractions differ between sensors.
    if has_dslr:
        dslr_annotations = load_annotation_files(dslr_annot_dir)
        print(f"  DSLR  annotations: {len(dslr_annotations)}")
    else:
        dslr_annotations = {}

    if has_rs:
        rs_annotations = load_annotation_files(rs_annot_dir)
        print(f"  RS    annotations: {len(rs_annotations)}")
    else:
        rs_annotations = {}

    # ── Collect all object names and assign / update IDs ─────────────────────
    all_obj_names: set[str] = set()
    for ann in {**dslr_annotations, **rs_annotations}.values():
        for obj in ann.get("objects", []):
            name = obj.get("object_name", "")
            if name:
                all_obj_names.add(name)

    for name in sorted(all_obj_names):
        get_or_assign_object_id(object_index, name)

    # ── Copy model PLY files into models_eval/, converting units to mm ───────
    # BOP is millimetres throughout; MOAD source meshes are metres. See
    # MODEL UNITS in the module docstring for why this fails silently if wrong.
    model_scale = {"meters": 1000.0, "mm": 1.0}[args.model_units]
    print(f"\n  {'=' * 74}")
    print(f"  MODEL UNITS: source meshes declared as '{args.model_units}' "
          f"→ scaling vertices by x{model_scale:g} into millimetres (BOP)")
    if model_scale == 1.0:
        print(f"  (no rescale: meshes are copied byte-for-byte)")
    print(f"  {'=' * 74}")

    if args.model_dir:
        for name in sorted(all_obj_names):
            obj_id   = object_index[name]
            dst_ply  = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
            if os.path.isfile(dst_ply):
                # Existing models are left alone so IDs stay stable across runs.
                # Their extent is still checked: models converted before the
                # unit fix will be flagged here.
                diag = describe_extent(read_ply_vertices(dst_ply))[1]
                print(f"  [MODEL KEEP ] {name} → obj_{obj_id:06d}.ply "
                      f"(existing, bbox diag {diag:.1f} mm)")
                check_extent_units(diag, name)
            if not os.path.isfile(dst_ply):
                # Search for a PLY named after the object in model_dir
                candidates = (
                    glob.glob(os.path.join(args.model_dir, name, "**", "*.ply"),
                              recursive=True) +
                    glob.glob(os.path.join(args.model_dir, "**", f"{name}.ply"),
                              recursive=True)
                )
                if candidates:
                    copy_and_scale_ply(candidates[0], dst_ply,
                                       model_scale, name, R_fix)
                else:
                    print(f"  [WARN] No PLY found for '{name}' in {args.model_dir}")

    # ── Update models_info.json with any new objects ──────────────────────────
    for name in sorted(all_obj_names):
        obj_id  = object_index[name]
        id_str  = str(obj_id)
        if id_str not in models_info:
            ply_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
            entry    = build_models_info_entry(obj_id, name, ply_path)
            models_info[id_str] = entry
            if not os.path.isfile(ply_path):
                print(f"  [WARN] PLY not found: {ply_path}")
                print(f"         Place pre-converted PLY there for the BOP toolkit.")
            else:
                print(f"  [MODEL] {name} → obj_{obj_id:06d}.ply  "
                      f"(diam={entry['diameter']:.1f} mm)")

    
    # ── Build and write scene-level JSON structures ───────────────────────────
    # Each sensor gets its own GT dict keyed by image_id. Poses are expressed
    # relative to that sensor's camera frame, so they differ between sensors
    # even for the same physical object instance.
    scene_camera_dslr  = {}   # image_id → camera entry
    scene_camera_rs    = {}
    scene_gt_dslr      = {}   # image_id → list of GT pose entries (DSLR frame)
    scene_gt_rs        = {}   # image_id → list of GT pose entries (RS frame)
    scene_gt_info_dslr = {}
    scene_gt_info_rs   = {}

    # -- DSLR ------------------------------------------------------------------
    # cam_K must describe the images that are actually written (images_4),
    # not the full-resolution calibration; probe the first frame rather than
    # assuming a downscale factor.
    dslr_size = None
    if has_dslr:
        dslr_src_paths = sorted(glob.glob(os.path.join(dslr_img_dir, "frame_*.jpg")))
        if not dslr_src_paths:
            print(f"[ERROR] no frame_*.jpg images in {dslr_img_dir}")
            sys.exit(1)
        dslr_size = image_size(dslr_src_paths[0])
        if dslr_size is None:
            print(f"[ERROR] cannot read {dslr_src_paths[0]}")
            sys.exit(1)

        # Depth must sit on the same pixel grid as the colour it accompanies,
        # since a single cam_K describes both.
        if has_dslr_depth:
            probe_stem  = Path(dslr_src_paths[0]).stem
            probe_dpath = os.path.join(dslr_depth_dir, f"{probe_stem}.png")
            d_size = image_size(probe_dpath)
            if d_size is None:
                print(f"[ERROR] no depth frame for {probe_stem} in {dslr_depth_dir}")
                sys.exit(1)
            if d_size != dslr_size:
                print(f"[ERROR] depth is {d_size[0]}x{d_size[1]} but DSLR colour is "
                      f"{dslr_size[0]}x{dslr_size[1]}; cam_K describes the colour "
                      f"grid, so they must match.")
                print(f"        Regenerate depth against "
                      f"{os.path.basename(dslr_img_dir)}.")
                sys.exit(1)
            import cv2 as _cv2
            probe_depth = _cv2.imread(probe_dpath, _cv2.IMREAD_UNCHANGED)
            if probe_depth is not None and probe_depth.dtype != np.uint16:
                print(f"  [WARN] depth dtype is {probe_depth.dtype}, expected uint16 "
                      f"(BOP: pixel * depth_scale = mm)")

        dslr_intr = scale_intrinsics(dslr_cal["intrinsics"], *dslr_size)
        print(f"  DSLR cam_K scaled to {dslr_size[0]}x{dslr_size[1]}: "
              f"fx={dslr_intr['fx']:.2f} cx={dslr_intr['cx']:.2f}")
        cam_entry = build_scene_camera_entry(
            dslr_intr,
            depth_scale=args.dslr_depth_scale,
            include_depth_scale=has_dslr_depth)
        for image_id, ann in dslr_annotations.items():
            scene_camera_dslr[image_id]  = cam_entry
            scene_gt_dslr[image_id]      = build_scene_gt_entries(ann, object_index)
            scene_gt_info_dslr[image_id] = build_scene_gt_info_entries(ann)

    # -- RealSense -------------------------------------------------------------
    if has_rs:
        rs_intr   = rs_cal["intrinsics"]
        cam_entry = build_scene_camera_entry(rs_intr,
                                             depth_scale=args.rs_depth_scale,
                                             include_depth_scale=True)
        for image_id, ann in rs_annotations.items():
            scene_camera_rs[image_id]   = cam_entry
            scene_gt_rs[image_id]       = build_scene_gt_entries(ann, object_index)
            scene_gt_info_rs[image_id]  = build_scene_gt_info_entries(ann)

    # ── Write JSON files ──────────────────────────────────────────────────────
    print(f"\n  Writing JSON files to {scene_dir}/")

    os.makedirs(scene_dir, exist_ok=True)

    # scene_index entry for human reference
    with open(os.path.join(scene_dir, "scene_index.json"), "w") as f:
        json.dump({
            "scene_id":   scene_id,
            "scene_key":  scene_key,
            "object":     args.object,
            "pose":       args.pose,
            "data_root":  args.data_root,
        }, f, indent=2)

    def _write_bop_json(data: dict, path: str) -> None:
        """Write a BOP-style JSON with integer keys as string keys."""
        out = {str(k): v for k, v in data.items()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    if has_dslr:
        _write_bop_json(scene_camera_dslr,
                        os.path.join(scene_dir, "scene_camera_dslr.json"))
        _write_bop_json(scene_gt_dslr,
                        os.path.join(scene_dir, "scene_gt_dslr.json"))
        _write_bop_json(scene_gt_info_dslr,
                        os.path.join(scene_dir, "scene_gt_info_dslr.json"))
        print(f"  scene_camera_dslr.json    ({len(scene_camera_dslr)} images)")
        print(f"  scene_gt_dslr.json        ({len(scene_gt_dslr)} images)")
        print(f"  scene_gt_info_dslr.json   ({len(scene_gt_info_dslr)} images)")

    if has_rs:
        _write_bop_json(scene_camera_rs,
                        os.path.join(scene_dir, "scene_camera_realsense.json"))
        _write_bop_json(scene_gt_rs,
                        os.path.join(scene_dir, "scene_gt_realsense.json"))
        _write_bop_json(scene_gt_info_rs,
                        os.path.join(scene_dir, "scene_gt_info_realsense.json"))
        print(f"  scene_camera_realsense.json  ({len(scene_camera_rs)} images)")
        print(f"  scene_gt_realsense.json      ({len(scene_gt_rs)} images)")
        print(f"  scene_gt_info_realsense.json ({len(scene_gt_info_rs)} images)")

    # ── Write camera.json and dataset_info.json at dataset root (once) ───────
    camera_json_path = os.path.join(dataset_dir, "camera.json")
    if not os.path.isfile(camera_json_path):
        # Use DSLR intrinsics as the reference camera (for rendering simulation).
        # These are the images_4 scaled values, matching p["im_size"] in dataset_params.
        # Scaled to the resolution actually written, matching p["im_size"] in
        # the toolkit's dataset_params for this sensor.
        if has_dslr:
            ref_intr = scale_intrinsics(dslr_cal["intrinsics"], *dslr_size)
        else:
            ref_intr = rs_cal["intrinsics"]
        camera_entry = {
            "cx":          round(ref_intr["cx"], 4),
            "cy":          round(ref_intr["cy"], 4),
            "depth_scale": 1.0,
            "fx":          round(ref_intr["fx"], 4),
            "fy":          round(ref_intr["fy"], 4),
            "height":      int(ref_intr["height"]),
            "width":       int(ref_intr["width"]),
        }
        with open(camera_json_path, "w") as f:
            json.dump(camera_entry, f, indent=2)
        print(f"  camera.json written ({camera_entry['width']}×{camera_entry['height']})")

    dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.isfile(dataset_info_path):
        with open(dataset_info_path, "w") as f:
            json.dump({
                "description": "MOAD multi-camera 6D object pose estimation dataset",
                "url":         "",
                "ref":         "",
            }, f, indent=2)
        print(f"  dataset_info.json written")

    # ── Build per-sensor test targets files at dataset root ──────────────────
    # Each sensor gets its own targets file so evaluation is always driven by
    # the correct GT — DSLR and RS have independent image ID spaces and can
    # have different visib_fract values for the same physical object instance.
    #
    #   test_targets_moad_dslr.json       → used with --sensor=dslr
    #   test_targets_moad_realsense.json  → used with --sensor=realsense
    #
    for sensor_name, sensor_annotations, targets_filename in [
        ("dslr",       dslr_annotations, "test_targets_moad_dslr.json"),
        ("realsense",  rs_annotations,   "test_targets_moad_realsense.json"),
    ]:
        if not sensor_annotations:
            continue

        targets_path     = os.path.join(dataset_dir, targets_filename)
        existing_targets = load_json(targets_path, [])

        # Re-run safety: remove existing entries for this scene before re-adding
        existing_targets = [t for t in existing_targets
                            if t.get("scene_id") != scene_id]

        new_targets = build_test_targets(
            scene_id     = scene_id,
            annotations  = sensor_annotations,
            object_index = object_index,
            visib_thresh = args.visib_thresh,
        )
        existing_targets.extend(new_targets)

        with open(targets_path, "w") as f:
            json.dump(existing_targets, f, indent=2)
        print(f"  {targets_filename} "
              f"(+{len(new_targets)} targets, {len(existing_targets)} total)")

    # ── Copy images ───────────────────────────────────────────────────────────
    print(f"\n  Copying images...")

    # -- DSLR colour (+ optional depth) ----------------------------------------
    if has_dslr:
        rgb_dir = os.path.join(scene_dir, "rgb_dslr")
        os.makedirs(rgb_dir, exist_ok=True)

        depth_dst_dir = None
        if has_dslr_depth:
            depth_dst_dir = os.path.join(scene_dir, "depth_dslr")
            os.makedirs(depth_dst_dir, exist_ok=True)

        # dslr_src_paths was resolved (and sorted) when building scene_camera.
        copied = copied_d = missing_d = 0
        for image_id, src in enumerate(dslr_src_paths, start=1):
            dst = os.path.join(rgb_dir, f"{image_id_str(image_id)}.png")
            if copy_as_png(src, dst):
                copied += 1

            if depth_dst_dir:
                # Match depth by STEM, never by a second sorted listing: one
                # missing depth frame would otherwise shift every later
                # image_id and pair depth with the wrong pose.
                stem      = Path(src).stem                  # frame_00001
                depth_src = os.path.join(dslr_depth_dir, f"{stem}.png")
                dst_depth = os.path.join(depth_dst_dir,
                                         f"{image_id_str(image_id)}.png")
                if copy_as_png(depth_src, dst_depth):       # PNG -> byte copy
                    copied_d += 1
                else:
                    missing_d += 1

        print(f"  rgb_dslr/                   {copied} DSLR images")
        if depth_dst_dir:
            note = f"  [{missing_d} MISSING]" if missing_d else ""
            print(f"  depth_dslr/                 {copied_d} DSLR depth images{note}")
            if missing_d:
                print(f"  [WARN] {missing_d} colour frames have no matching depth; "
                      f"those image_ids will have colour but no depth.")

    # -- RealSense colour + depth ----------------------------------------------
    if has_rs:
        rgb_rs_dir   = os.path.join(scene_dir, "rgb_realsense")
        depth_rs_dir = os.path.join(scene_dir, "depth_realsense")
        os.makedirs(rgb_rs_dir,   exist_ok=True)
        os.makedirs(depth_rs_dir, exist_ok=True)

        rs_color_paths = sorted(glob.glob(os.path.join(rs_img_dir, "rs*_color.png")))
        copied_c = copied_d = 0
        for image_id, color_src in enumerate(rs_color_paths, start=1):
            # Colour
            dst_color = os.path.join(rgb_rs_dir, f"{image_id_str(image_id)}.png")
            if copy_as_png(color_src, dst_color):
                copied_c += 1

            # Depth — derive path by replacing _color with _depth
            depth_src = color_src.replace("_color.png", "_depth.png")
            dst_depth = os.path.join(depth_rs_dir, f"{image_id_str(image_id)}.png")
            if copy_as_png(depth_src, dst_depth):
                copied_d += 1

        print(f"  rgb_realsense/         {copied_c} RS color images")
        print(f"  depth_realsense/       {copied_d} RS depth images")

    # ── Record DSLR depth provenance ─────────────────────────────────────────
    # BOP has no field for "where did this depth come from", and the three
    # possible generators (RealSense reprojection, COLMAP stereo, DA3) have
    # very different error characteristics. The generator's own meta.json
    # carries backend/model/geometry/config hash, so copy it in (minus the
    # per-frame stats, which would bloat the scene folder).
    if has_dslr and has_dslr_depth:
        depth_info = {
            "source_dir":   dslr_depth_dir,
            "depth_scale":  args.dslr_depth_scale,
            "units":        "millimetres (pixel * depth_scale = mm)",
            "geometry":     "raw (distorted) DSLR, pixel-aligned with rgb_dslr/",
            "image_count":  copied_d,
            "missing_count": missing_d,
        }
        gen_meta_path = os.path.join(dslr_depth_dir, "meta.json")
        if os.path.isfile(gen_meta_path):
            gen_meta = load_json(gen_meta_path, {})
            gen_meta.pop("frames", None)          # per-frame stats: not needed here
            depth_info["generator_meta"] = gen_meta
        else:
            depth_info["generator_meta"] = None
            print(f"  [WARN] no meta.json in {dslr_depth_dir} — depth provenance "
                  f"will only record the source path")
        save_json(os.path.join(scene_dir, "depth_dslr_info.json"), depth_info)
        print(f"  depth_dslr_info.json        (provenance for depth_dslr/)")

    # ── Save updated persistent indices ──────────────────────────────────────
    save_json(obj_index_path,   object_index)
    save_json(scene_index_path, scene_index)
    save_json(models_info_path, models_info)

    print(f"\n  ✓  Conversion complete.")
    print(f"     Dataset dir : {dataset_dir}")
    print(f"     Scene dir   : {scene_dir}")
    print(f"     Objects     : {len(object_index)} in index")
    print(f"     Scenes      : {len(scene_index)} in index\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a MOAD scene/pose to BOP-scenewise format."
    )

    # ── Source data ───────────────────────────────────────────────────────────
    parser.add_argument("--data-root",  default="/home/csrobot/MOAD_DATA",
                        help="MOAD data root directory")
    parser.add_argument("--object",     default="ex2_006",
                        help="Object subfolder name")
    parser.add_argument("--pose",       default="pose-b",
                        help="Pose subfolder name")
    parser.add_argument("--calib-root", default="/home/csrobot/moad_control/moad_cui/calibration/55mm_joint",
                        help="Calibration folder (must contain cam_parameters.json "
                             "and optionally realsense_cam_parameters.json)")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--bop-root",   default="/home/csrobot/BOP_MOAD_DATA",
                        help="Root directory for BOP data. A 'moad/' subfolder "
                             "is created here containing the dataset in BOP format. "
                             "Set BOP_PATH to this directory when running the toolkit.")
    parser.add_argument("--dataset-name", default="moad",
                        help="The name of the overall dataset to add the target scene to. "
                            "Dataset must be defined in bop_toolkit_lib/dataset_params.py")
    # ── Model assets ──────────────────────────────────────────────────────────
    parser.add_argument("--model-dir",  default="/home/csrobot/MOADv2/data/ply_for_BOP",
                        help="Directory containing per-object subfolders with "
                             "pre-converted .ply files. If provided, PLY files "
                             "are copied to meta/models/obj_XXXXXX.ply. "
                             "If omitted, place PLY files there manually.")

    # ── Evaluation settings ───────────────────────────────────────────────────
    parser.add_argument("--model-units", choices=["meters", "mm"], default="meters",
                        help="Units of the SOURCE meshes in --model-dir. BOP "
                             "requires millimetres, so 'meters' (the MOAD "
                             "default) scales vertices by 1000 on copy. Getting "
                             "this wrong does not fail loudly: MSSD/MSPD still "
                             "score 1.0 on a GT-vs-GT run and only VSD collapses.")
    parser.add_argument("--model-frame-rot", default="x:-90",
                        help="Rotation applied to source meshes as they are "
                             "copied into models_eval/, taking them into the "
                             "annotation frame, as 'axis:deg' terms, e.g. "
                             "'x:-90' for Z-up source meshes annotated against "
                             "Y-up replication meshes. Poses stay exactly as "
                             "annotated. Invalidates models_info symmetries if "
                             "changed. Default: none.")
    parser.add_argument("--force-settings", action="store_true",
                        help="Allow this run's model units / frame rotation to "
                             "differ from meta/conversion_settings.json (only "
                             "when reconverting the whole dataset).")
    parser.add_argument("--visib-thresh", type=float, default=0.1,
                        help="Minimum visib_fract for an instance to be included "
                             "in test_targets_bop19.json (default: 0.1)")
    parser.add_argument("--dslr-depth-subdir", default="DSLR_depth",
                        help="Subfolder of the scan/pose directory holding depth "
                             "aligned to the DSLR frames, e.g. DSLR_depth or "
                             "DSLR_depth_da3. Copied into depth_dslr/ with the "
                             "source recorded in depth_dslr_info.json. Omit to "
                             "convert DSLR colour only.")
    parser.add_argument("--dslr-depth-scale", type=float, default=1.0,
                        help="DSLR depth scale: multiply uint16 pixel value by "
                             "this to get mm. MOAD depth PNGs are already "
                             "millimetres, so 1.0.")
    parser.add_argument("--rs-depth-scale", type=float, default=1.0,
                        help="RealSense depth scale: multiply uint16 pixel value "
                             "by this to get depth in mm. Default 0.001 converts RS native "
                             "uint16 in mm to meters (model scales)).")

    args = parser.parse_args()
    convert_scene(args)


if __name__ == "__main__":
    main()