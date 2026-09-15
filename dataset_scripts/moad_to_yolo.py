#!/usr/bin/env python3
"""
moad_to_yolo.py
---------------
Convert MOAD annotated replicated scenes into YOLO-format detection datasets
for benchmarking trained detection models.

Each scene (object/pose pair) is converted to its own self-contained YOLO
dataset directory, suitable for direct use with Ultralytics model.val().

OUTPUT STRUCTURE (one per scene)
    <output_root>/<scene_name>/
    ├── data.yaml              # Ultralytics dataset descriptor
    ├── images/
    │   └── test/
    │       ├── frame_00001.jpg
    │       └── ...
    └── labels/
        └── test/
            ├── frame_00001.txt   # YOLO format: class cx cy w h (normalised)
            └── ...

YOLO LABEL FORMAT
    One line per object instance:
        <class_id> <cx> <cy> <width> <height>
    All values normalised to [0, 1] by image width/height.
    bbox_visib is used (visible region only — matches what the detector sees).

CLASS REMAPPING
    Object names from annotations are remapped via CLASS_REMAP before being
    assigned integer class IDs. Instances whose remapped name is not in the
    final class list are skipped with a warning.

USAGE
    # Single scene
    python3 moad_to_yolo.py \\
        --data-root  /home/csrobot/MOAD_DATA \\
        --object     batch1_007 \\
        --pose       pose-a \\
        --output-root /home/csrobot/yolo_datasets

    # Multiple scenes in one call
    python3 moad_to_yolo.py \\
        --data-root  /home/csrobot/MOAD_DATA \\
        --scenes     batch1_007/pose-a batch1_007/pose-b batch2_003/pose-a \\
        --output-root /home/csrobot/yolo_datasets

    # Then benchmark with Ultralytics:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    results = model.val(data="/home/csrobot/yolo_datasets/batch1_007_pose-a/data.yaml",
                        split="test")
"""

import os
import json
import glob
import shutil
import argparse
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Class remapping
# ---------------------------------------------------------------------------

# Map annotation object names → YOLO class names.
# Keys are what appears in the annotation JSON "object_name" field.
# Values are the class names used in your trained YOLO models.
# Instances whose remapped name is not in CLASSES are skipped.
CLASS_REMAP: dict[str, str] = {
    "kiri-wp":        "conn_wp",
    "kiri-gear":      "gear_large",
    "kiri-nut":       "nut_m16",
    "kiri-sprocket":  "sprocket_large",
}

# Final ordered class list — determines integer class IDs (index = ID).
# Edit this to match the exact class order your YOLO models were trained with.
CLASSES: list[str] = [
    "conn_wp",
    "gear_large",
    "nut_m16",
    "sprocket_large",
]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VISIB_THRESH  = 0.25   # minimum visib_fract to include an instance
DEFAULT_IMAGES_SUBDIR = "images_4"   # DSLR images subfolder within pose folder
DEFAULT_ANNOT_SUBDIR  = "scene_replica/pose_dslr"   # annotation JSON subfolder
DEFAULT_SPLIT         = "test"   # YOLO split name (test / val / train)


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_scene(
    pose_path:      str,
    output_dir:     str,
    visib_thresh:   float = DEFAULT_VISIB_THRESH,
    images_subdir:  str   = DEFAULT_IMAGES_SUBDIR,
    annot_subdir:   str   = DEFAULT_ANNOT_SUBDIR,
    split:          str   = DEFAULT_SPLIT,
    class_remap:    dict  = CLASS_REMAP,
    classes:        list  = CLASSES,
    copy_images:    bool  = True,
) -> dict:
    """
    Convert one MOAD annotated scene to YOLO format.

    Args:
        pose_path      : path to the pose folder
                         (e.g. /MOAD_DATA/batch1_007/pose-a)
        output_dir     : output dataset root for this scene
        visib_thresh   : minimum visib_fract — instances below this are skipped
        images_subdir  : subfolder within pose_path containing DSLR images
        annot_subdir   : subfolder within pose_path containing annotation JSONs
        split          : YOLO split name written to data.yaml and used as
                         the subfolder under images/ and labels/
        class_remap    : dict mapping annotation names → YOLO class names
        classes        : ordered list of final class names (index = class ID)
        copy_images    : if True, copy images to output_dir/images/<split>/
                         if False, data.yaml will reference images in-place

    Returns:
        Summary dict with counts of frames, instances, skipped instances.
    """
    images_dir = os.path.join(pose_path, images_subdir)
    annot_dir  = os.path.join(pose_path, annot_subdir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Images folder not found: {images_dir}\n"
            f"Check --images-subdir (default: '{DEFAULT_IMAGES_SUBDIR}')"
        )
    if not os.path.isdir(annot_dir):
        raise FileNotFoundError(
            f"Annotation folder not found: {annot_dir}\n"
            f"Check --annot-subdir (default: '{DEFAULT_ANNOT_SUBDIR}')"
        )

    # Build class ID lookup from the final class list
    class_to_id = {name: i for i, name in enumerate(classes)}

    # Output directories
    out_images = os.path.join(output_dir, "images", split)
    out_labels = os.path.join(output_dir, "labels", split)
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # Find all annotation JSONs, sorted for deterministic ordering
    annot_paths = sorted(glob.glob(os.path.join(annot_dir, "*.json")))
    if not annot_paths:
        raise FileNotFoundError(f"No annotation JSON files in: {annot_dir}")

    # Counters for summary
    n_frames           = 0
    n_instances        = 0
    n_skipped_visib    = 0
    n_skipped_noclass  = 0
    n_skipped_badbox   = 0
    n_missing_images   = 0

    for annot_path in annot_paths:
        with open(annot_path) as f:
            ann = json.load(f)

        frame_name    = ann.get("frame", "")
        img_w         = ann.get("width",  0)
        img_h         = ann.get("height", 0)

        if img_w <= 0 or img_h <= 0:
            print(f"  [WARN] {os.path.basename(annot_path)}: "
                  f"missing/invalid image dimensions — skipping")
            continue

        # Locate source image
        img_src = os.path.join(images_dir, frame_name)
        if not os.path.isfile(img_src):
            # Try matching by stem in case extension differs
            stem      = os.path.splitext(frame_name)[0]
            candidates = glob.glob(os.path.join(images_dir, f"{stem}.*"))
            if candidates:
                img_src = candidates[0]
                frame_name = os.path.basename(img_src)
            else:
                print(f"  [WARN] Image not found: {img_src} — skipping frame")
                n_missing_images += 1
                continue

        # Build YOLO label lines for this frame
        label_lines = []
        for obj in ann.get("objects", []):
            obj_name    = obj.get("object_name", "")
            visib_fract = float(obj.get("visib_fract", 1.0))
            bbox_visib  = obj.get("bbox_visib")   # [x, y, w, h] in pixels

            # Visibility filter
            if visib_fract < visib_thresh:
                n_skipped_visib += 1
                continue

            # Class remapping
            yolo_name = class_remap.get(obj_name, obj_name)
            if yolo_name not in class_to_id:
                n_skipped_noclass += 1
                continue
            class_id = class_to_id[yolo_name]

            # Validate and normalise bbox
            if not bbox_visib or len(bbox_visib) < 4:
                n_skipped_badbox += 1
                continue

            x, y, bw, bh = [float(v) for v in bbox_visib]
            if bw <= 0 or bh <= 0:
                n_skipped_badbox += 1
                continue

            # YOLO format: normalised centre x, centre y, width, height
            cx = (x + bw / 2.0) / img_w
            cy = (y + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Clamp to [0, 1] to handle any boundary rounding
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_instances += 1

        # Skip frames with no valid instances
        # (keeps images/ and labels/ in sync — YOLO expects one label per image)
        if not label_lines:
            continue

        # Write label file
        stem       = os.path.splitext(frame_name)[0]
        label_path = os.path.join(out_labels, f"{stem}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        # Copy or symlink image
        img_dst = os.path.join(out_images, frame_name)
        if copy_images:
            shutil.copy2(img_src, img_dst)
        else:
            if not os.path.exists(img_dst):
                os.symlink(os.path.abspath(img_src), img_dst)

        n_frames += 1

    return {
        "frames":          n_frames,
        "instances":       n_instances,
        "skipped_visib":   n_skipped_visib,
        "skipped_noclass": n_skipped_noclass,
        "skipped_badbox":  n_skipped_badbox,
        "missing_images":  n_missing_images,
    }


def write_data_yaml(
    output_dir: str,
    scene_name: str,
    classes:    list[str],
    split:      str,
) -> str:
    """
    Write the Ultralytics data.yaml descriptor for this dataset.

    Returns the path to the written file.
    """
    # Ultralytics resolves image paths relative to data.yaml location
    yaml_data = {
        "path":  os.path.abspath(output_dir),
        "train": None,
        "val":   None,
        "test":  None,
        "nc":    len(classes),
        "names": classes,
    }

    # Set the active split key
    if split == "test":
        yaml_data["test"] = f"images/test"
    elif split == "val":
        yaml_data["val"] = f"images/val"
    else:
        yaml_data["train"] = f"images/train"

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    return yaml_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    # Build the list of (object, pose) pairs to process
    scenes: list[tuple[str, str]] = []

    if args.scenes:
        # --scenes batch1_007/pose-a batch1_007/pose-b ...
        for s in args.scenes:
            parts = s.strip("/").split("/")
            if len(parts) != 2:
                print(f"  [ERROR] --scenes entries must be 'object/pose', got: {s}")
                continue
            scenes.append((parts[0], parts[1]))
    elif args.object and args.pose:
        scenes.append((args.object, args.pose))
    else:
        print("[ERROR] Provide either --scenes or both --object and --pose.")
        return

    print(f"\n  MOAD → YOLO conversion")
    print(f"  Data root    : {args.data_root}")
    print(f"  Output root  : {args.output_root}")
    print(f"  Split        : {args.split}")
    print(f"  Visib thresh : {args.visib_thresh}")
    print(f"  Classes      : {CLASSES}")
    print(f"  Remap        : {CLASS_REMAP}")
    print(f"  Scenes       : {len(scenes)}\n")

    for obj_name, pose_name in scenes:
        scene_name = f"{obj_name}_{pose_name}"
        pose_path  = os.path.join(args.data_root, obj_name, pose_name)
        output_dir = os.path.join(args.output_root, scene_name)

        print(f"  {'─'*56}")
        print(f"  Scene : {scene_name}")
        print(f"  Input : {pose_path}")
        print(f"  Output: {output_dir}")

        if not os.path.isdir(pose_path):
            print(f"  [SKIP] Pose folder not found: {pose_path}")
            continue

        try:
            stats = convert_scene(
                pose_path     = pose_path,
                output_dir    = output_dir,
                visib_thresh  = args.visib_thresh,
                images_subdir = args.images_subdir,
                annot_subdir  = args.annot_subdir,
                split         = args.split,
                class_remap   = CLASS_REMAP,
                classes       = CLASSES,
                copy_images   = not args.symlink_images,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        yaml_path = write_data_yaml(
            output_dir = output_dir,
            scene_name = scene_name,
            classes    = CLASSES,
            split      = args.split,
        )

        print(f"\n  Frames written      : {stats['frames']}")
        print(f"  Instances written   : {stats['instances']}")
        print(f"  Skipped (visib)     : {stats['skipped_visib']}")
        print(f"  Skipped (no class)  : {stats['skipped_noclass']}")
        print(f"  Skipped (bad bbox)  : {stats['skipped_badbox']}")
        print(f"  Missing images      : {stats['missing_images']}")
        print(f"  data.yaml           : {yaml_path}")
        print(f"\n  To benchmark:")
        print(f"    from ultralytics import YOLO")
        print(f"    results = YOLO('best.pt').val(")
        print(f"        data='{yaml_path}',")
        print(f"        split='{args.split}')")

    print(f"\n  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MOAD annotated scenes to YOLO-format detection datasets."
    )

    # ── Scene selection ───────────────────────────────────────────────────────
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--scenes", nargs="+",
                     metavar="OBJECT/POSE",
                     help="One or more object/pose pairs to convert, e.g. "
                          "batch1_007/pose-a batch1_007/pose-b")
    grp.add_argument("--object", metavar="OBJECT",
                     help="Object subfolder name (use with --pose)")

    parser.add_argument("--pose", metavar="POSE",
                        help="Pose subfolder name (use with --object)")

    # ── Paths ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data-root",
                        default="/home/csrobot/MOAD_DATA",
                        help="MOAD data root directory")
    parser.add_argument("--output-root",
                        default="/home/csrobot/yolo_datasets",
                        help="Root directory for output YOLO datasets "
                             "(one subdirectory per scene)")
    parser.add_argument("--images-subdir",
                        default=DEFAULT_IMAGES_SUBDIR,
                        help=f"Images subfolder within pose folder "
                             f"(default: '{DEFAULT_IMAGES_SUBDIR}')")
    parser.add_argument("--annot-subdir",
                        default=DEFAULT_ANNOT_SUBDIR,
                        help=f"Annotation subfolder within pose folder "
                             f"(default: '{DEFAULT_ANNOT_SUBDIR}')")

    # ── Dataset options ───────────────────────────────────────────────────────
    parser.add_argument("--split",
                        default=DEFAULT_SPLIT,
                        choices=["test", "val", "train"],
                        help=f"YOLO split name (default: '{DEFAULT_SPLIT}'). "
                             f"Use 'test' for benchmarking, 'val' for validation "
                             f"during training.")
    parser.add_argument("--visib-thresh",
                        type=float, default=DEFAULT_VISIB_THRESH,
                        help=f"Minimum visib_fract to include an instance "
                             f"(default: {DEFAULT_VISIB_THRESH})")
    parser.add_argument("--symlink-images",
                        action="store_true", default=False,
                        help="Symlink images instead of copying them. "
                             "Faster and saves disk space, but the dataset "
                             "is not portable to other machines.")

    args = parser.parse_args()

    # Validate --object requires --pose
    if args.object and not args.pose:
        parser.error("--object requires --pose")

    main(args)
