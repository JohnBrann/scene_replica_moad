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
            ├── rgb/                  # DSLR colour images (000001.png …)
            ├── scene_camera.json     # DSLR intrinsics per image
            ├── scene_gt_dslr.json    # GT poses relative to DSLR cameras
            ├── scene_gt_realsense.json  # GT poses relative to RS cameras
            ├── scene_gt_info.json    # visibility info for DSLR images
            ├── rgb_realsense/        # RS colour images
            ├── depth_realsense/      # RS depth images (uint16, mm)
            ├── scene_camera_realsense.json
            └── scene_gt_info_realsense.json

MODALITY CONVENTIONS
    DSLR        : image IDs 1…360, written to rgb/ and scene_camera.json
    RealSense   : image IDs 1…360, written to rgb_realsense/, depth_realsense/
                  and scene_camera_realsense.json  (independent ID space)
    Each sensor's GT poses are expressed in that sensor's camera frame —
    scene_gt_dslr.json and scene_gt_realsense.json contain different R/t
    values for the same physical object because each camera has different
    extrinsics. Targets files are also per-sensor (test_targets_moad_dslr.json
    and test_targets_moad_realsense.json) since visibility fractions differ.

UNITS
    All translation vectors are in MILLIMETRES (BOP convention).
    Source annotations are in metres — the script multiplies t by 1000.
    Depth images are copied as-is (RealSense uint16 in mm, depth_scale=1.0).

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

def object_diameter_from_ply(ply_path: str) -> float:
    """
    Approximate the BOP diameter (largest pairwise vertex distance) for a PLY
    file by reading vertex positions and computing the max distance between
    any two vertices. For large meshes we subsample to keep it fast.

    Returns diameter in millimetres (BOP convention). PLY vertices are assumed
    to already be in millimetres (convert from metres before calling if needed).
    """
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(ply_path)
        verts = np.asarray(mesh.vertices)
    except ImportError:
        # Fallback: parse ASCII PLY manually for vertex lines
        verts = []
        in_vertex = False
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
                            verts.append([float(p) for p in parts[:3]])
                        except ValueError:
                            pass
        verts = np.array(verts)

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
    print(f"PLY path: {ply_path}")
    diam = object_diameter_from_ply(ply_path) if os.path.isfile(ply_path) else 0.0

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


def build_scene_gt_entries(
    annotation:    dict,
    object_index:  dict,
) -> list[dict]:
    """
    Build the list of GT entries for one image in scene_gt.json.
    Translation is converted from metres to millimetres.
    R is stored row-wise as a flat 9-element list.
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
    rs_img_dir     = os.path.join(pose_path, "realsense")
    # Sensor calibration paths
    dslr_calib_path = os.path.join(args.calib_root, "cam_parameters.json")
    rs_calib_path   = os.path.join(args.calib_root, "realsense_cam_parameters.json")

    for d in [dataset_dir, models_dir, meta_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

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

    print(f"  Modalities: "
          f"{'DSLR ' if has_dslr else ''}{'RealSense' if has_rs else ''}")

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

    # ── Copy model PLY files into meta/models/ if source dir provided ─────────
    if args.model_dir:
        for name in sorted(all_obj_names):
            obj_id   = object_index[name]
            dst_ply  = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
            if not os.path.isfile(dst_ply):
                # Search for a PLY named after the object in model_dir
                candidates = (
                    glob.glob(os.path.join(args.model_dir, name, "**", "*.ply"),
                              recursive=True) +
                    glob.glob(os.path.join(args.model_dir, "**", f"{name}.ply"),
                              recursive=True)
                )
                if candidates:
                    shutil.copy2(candidates[0], dst_ply)
                    print(f"  [COPY PLY] {candidates[0]} → {dst_ply}")
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
    if has_dslr:
        dslr_intr = dslr_cal["intrinsics"]
        cam_entry = build_scene_camera_entry(dslr_intr,
                                             include_depth_scale=False)
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
        ref_intr = (dslr_cal or rs_cal)["intrinsics"]
        # Scale to images_4 resolution (÷4) if using full-res DSLR calibration
        scale = 1500 / ref_intr["width"] if has_dslr else 1.0
        camera_entry = {
            "cx":          round(ref_intr["cx"] * scale, 4),
            "cy":          round(ref_intr["cy"] * scale, 4),
            "depth_scale": 1.0,
            "fx":          round(ref_intr["fx"] * scale, 4),
            "fy":          round(ref_intr["fy"] * scale, 4),
            "height":      int(round(ref_intr["height"] * scale)),
            "width":       int(round(ref_intr["width"]  * scale)),
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

    # -- DSLR colour -----------------------------------------------------------
    if has_dslr:
        rgb_dir = os.path.join(scene_dir, "rgb_dslr")
        os.makedirs(rgb_dir, exist_ok=True)

        dslr_paths = sorted(glob.glob(os.path.join(dslr_img_dir, "frame_*.jpg")))
        copied = 0
        for image_id, src in enumerate(dslr_paths, start=1):
            dst = os.path.join(rgb_dir, f"{image_id_str(image_id)}.png")
            if copy_as_png(src, dst):
                copied += 1
        print(f"  rgb_dslr/                   {copied} DSLR images")

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
    parser.add_argument("--object",     default="batch1_007",
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
    parser.add_argument("--visib-thresh", type=float, default=0.1,
                        help="Minimum visib_fract for an instance to be included "
                             "in test_targets_bop19.json (default: 0.1)")
    parser.add_argument("--rs-depth-scale", type=float, default=1.0,
                        help="RealSense depth scale: multiply uint16 pixel value "
                             "by this to get depth in mm. Default 0.001 converts RS native "
                             "uint16 in mm to meters (model scales)).")

    args = parser.parse_args()
    convert_scene(args)


if __name__ == "__main__":
    main()