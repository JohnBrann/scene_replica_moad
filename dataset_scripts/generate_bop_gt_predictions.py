#!/usr/bin/env python3
"""
generate_bop_gt_predictions.py
-------------------------------
Generates BOP-format predictions CSVs from ground truth annotations,
one per available sensor modality.

Used to verify the full evaluation pipeline end-to-end:
    1. Convert MOAD scene to BOP format  (moad_to_bop.py)
    2. Generate GT predictions CSVs      ← this script
    3. Run MOAD evaluation               (eval_moad_pose.py --sensor dslr/realsense)
    4. Confirm score ≈ 1.0

A perfect score validates that:
  - scene_gt_[sensor].json is correctly formatted
  - models_info.json has correct diameters
  - test_targets_bop19.json lists the right instances
  - cam_R_m2c / cam_t_m2c units and conventions are correct

OUTPUT FILENAMES
    {method}-{sensor}_moad-test.csv

    The BOP toolkit parses filenames by splitting on the FIRST underscore:
        result_name.split('_') → [method_token, 'moad-test']
        'moad-test'.split('-') → ['moad', 'test']

    Using a hyphen between method and sensor name keeps exactly one underscore
    in the filename, so dataset is always parsed as 'moad':

        gt-dslr_moad-test.csv       → method='gt-dslr',  dataset='moad', split='test' ✓
        gt-realsense_moad-test.csv  → method='gt-realsense', dataset='moad', split='test' ✓

USAGE
    # Both sensors, perfect GT:
    python3 generate_bop_gt_predictions.py \\
        --bop-root   /home/csrobot/BOP_MOAD_DATA/moad \\
        --output-dir /home/csrobot/BOP_MOAD_DATA/predictions \\
        --method-name gt

    # Specific scenes only:
    python3 generate_bop_gt_predictions.py \\
        --bop-root   /home/csrobot/BOP_MOAD_DATA/moad \\
        --output-dir /home/csrobot/BOP_MOAD_DATA/predictions \\
        --method-name gt \\
        --scene-ids 1 2

    # With noise for degradation testing:
    python3 generate_bop_gt_predictions.py \\
        --bop-root   /home/csrobot/BOP_MOAD_DATA/moad \\
        --output-dir /home/csrobot/BOP_MOAD_DATA/predictions \\
        --method-name gt_noise \\
        --noise-t 5.0 --noise-r 5.0

OUTPUT FORMAT (BOP CSV)
    scene_id,im_id,obj_id,score,R,t,time
    R: 9 space-separated floats (row-wise 3x3 rotation matrix)
    t: 3 space-separated floats (translation in mm)
"""

import os
import sys
import json
import csv
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Sensor configuration
# ---------------------------------------------------------------------------

# Maps sensor name → GT filename within each scene folder
SENSOR_GT_FILES = {
    "dslr":      "scene_gt_dslr.json",
    "realsense": "scene_gt_realsense.json",
}


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def random_rotation_perturbation(deg_std: float) -> np.ndarray:
    """
    Generate a small random rotation matrix with angle drawn from
    N(0, deg_std) degrees around a uniformly random axis.
    """
    angle_rad = np.random.normal(0, np.radians(deg_std))
    axis      = np.random.randn(3)
    axis     /= np.linalg.norm(axis) + 1e-12

    K = np.array([
        [      0, -axis[2],  axis[1]],
        [ axis[2],       0, -axis[0]],
        [-axis[1],  axis[0],       0],
    ])
    R = (np.eye(3)
         + np.sin(angle_rad) * K
         + (1 - np.cos(angle_rad)) * K @ K)
    return R


def apply_noise(
    R: np.ndarray,
    t: np.ndarray,
    noise_t_mm:  float,
    noise_r_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply independent Gaussian noise to a pose.

    Args:
        R           : (3,3) rotation matrix
        t           : (3,) translation in mm
        noise_t_mm  : std dev of translation noise in mm  (0 = no noise)
        noise_r_deg : std dev of rotation noise in degrees (0 = no noise)

    Returns:
        R_noisy, t_noisy
    """
    t_noisy = t.copy()
    R_noisy = R.copy()
    if noise_t_mm  > 0:
        t_noisy = t + np.random.normal(0, noise_t_mm, size=3)
    if noise_r_deg > 0:
        R_noisy = random_rotation_perturbation(noise_r_deg) @ R
    return R_noisy, t_noisy


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_predictions_csv(rows: list[dict], output_path: str) -> None:
    """Write a BOP-format predictions CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for row in rows:
            R_flat = " ".join(f"{v:.8f}" for v in row["R"].flatten())
            t_flat = " ".join(f"{v:.4f}"  for v in row["t"])
            writer.writerow([
                row["scene_id"],
                row["im_id"],
                row["obj_id"],
                row["score"],
                R_flat,
                t_flat,
                row["time"],
            ])
    print(f"  Written {len(rows)} rows → {output_path}")


# ---------------------------------------------------------------------------
# Per-sensor GT extraction
# ---------------------------------------------------------------------------

def generate_for_sensor(
    sensor:      str,
    gt_filename: str,
    scenes_dir:  str,
    scene_ids:   list[int],
    noise_t:     float,
    noise_r:     float,
) -> list[dict]:
    """
    Load scene_gt_[sensor].json for each scene and build prediction rows.

    Returns a list of row dicts, or an empty list if no GT files were found.
    """
    rows = []

    for scene_id in scene_ids:
        gt_path = os.path.join(scenes_dir, f"{scene_id:06d}", gt_filename)
        if not os.path.isfile(gt_path):
            print(f"  [{sensor.upper()}] [SKIP] scene {scene_id:06d}: "
                  f"{gt_filename} not found")
            continue

        with open(gt_path) as f:
            scene_gt = json.load(f)

        n_instances = sum(len(v) for v in scene_gt.values())
        print(f"  [{sensor.upper()}] scene {scene_id:06d}: "
              f"{len(scene_gt)} images, {n_instances} instances")

        for im_id_str, gt_list in scene_gt.items():
            im_id = int(im_id_str)
            for gt in gt_list:
                R = np.array(gt["cam_R_m2c"]).reshape(3, 3)
                t = np.array(gt["cam_t_m2c"])       # already in mm

                if noise_t > 0 or noise_r > 0:
                    R, t = apply_noise(R, t, noise_t, noise_r)

                rows.append({
                    "scene_id": scene_id,
                    "im_id":    im_id,
                    "obj_id":   gt["obj_id"],
                    "score":    1.0,   # perfect confidence for GT baseline
                    "R":        R,
                    "t":        t,
                    "time":     -1,    # no inference time for GT
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # bop_root is expected to be the dataset dir (containing test/ and meta/)
    scenes_dir = os.path.join(args.bop_root, "test")

    if not os.path.isdir(scenes_dir):
        print(f"[ERROR] test/ directory not found: {scenes_dir}")
        print(f"        Run moad_to_bop.py first, and point --bop-root at the "
              f"moad dataset directory (containing test/, meta/, models_eval/).")
        sys.exit(1)

    # Discover available scene IDs
    available_scenes = sorted([
        int(d) for d in os.listdir(scenes_dir)
        if os.path.isdir(os.path.join(scenes_dir, d)) and d.isdigit()
    ])
    if not available_scenes:
        print(f"[ERROR] No scene folders found in {scenes_dir}")
        sys.exit(1)

    # Filter to requested scene IDs
    if args.scene_ids:
        scene_ids = args.scene_ids
        missing   = [s for s in scene_ids if s not in available_scenes]
        if missing:
            print(f"[ERROR] Requested scene IDs not found: {missing}")
            print(f"        Available: {available_scenes}")
            sys.exit(1)
    else:
        scene_ids = available_scenes
        print(f"  No --scene-ids specified, using all: {scene_ids}")

    print(f"\n  BOP dataset dir : {args.bop_root}")
    print(f"  Scene IDs       : {scene_ids}")
    print(f"  Method name     : {args.method_name}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Noise t         : {args.noise_t} mm std")
    print(f"  Noise r         : {args.noise_r} deg std\n")

    os.makedirs(args.output_dir, exist_ok=True)
    generated = []

    for sensor, gt_filename in SENSOR_GT_FILES.items():
        print(f"  ── {sensor.upper()} ──")

        rows = generate_for_sensor(
            sensor      = sensor,
            gt_filename = gt_filename,
            scenes_dir  = scenes_dir,
            scene_ids   = scene_ids,
            noise_t     = args.noise_t,
            noise_r     = args.noise_r,
        )

        if not rows:
            print(f"  [{sensor.upper()}] No GT data found — skipping.\n")
            continue

        # Filename: {method}-{sensor}_moad-test.csv
        # The BOP toolkit parses: result_name.split('_') → [method_token, 'moad-test']
        # then 'moad-test'.split('-') → ['moad', 'test'] for dataset + split.
        # Using a hyphen between method and sensor keeps the single underscore
        # that separates the method token from the dataset-split token, so
        # dataset is always parsed as 'moad' regardless of the sensor name.
        #
        # Example: gt-dslr_moad-test.csv
        #   split('_') → ['gt-dslr', 'moad-test']
        #   'moad-test'.split('-') → ['moad', 'test']  ← correct
        csv_name    = f"{args.method_name}-{sensor}_moad-test.csv"
        output_path = os.path.join(args.output_dir, csv_name)
        write_predictions_csv(rows, output_path)
        generated.append((sensor, csv_name))
        print()

    if not generated:
        print("[ERROR] No GT data found for any sensor — nothing written.")
        sys.exit(1)

    # ── Evaluation commands ───────────────────────────────────────────────────
    print(f"\n  ✓  Done. Generated {len(generated)} CSV file(s):\n")
    for sensor, csv_name in generated:
        print(f"    {csv_name}")

    print(f"\n  To evaluate with eval_moad_pose.py:")
    for sensor, csv_name in generated:
        print(f"\n    # {sensor.upper()}")
        print(f"    python scripts/eval_moad_pose.py \\")
        print(f"        --sensor={sensor} \\")
        print(f"        --result_filenames={csv_name} ")
        # print(f"        --results_path={args.output_dir} \\")
        # print(f"        --eval_path=/home/csrobot/BOP_MOAD_DATA/eval_output")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate BOP-format GT predictions CSVs for pipeline verification. "
                    "Automatically generates one CSV per available sensor modality."
    )

    parser.add_argument(
        "--bop-root", default="/home/csrobot/BOP_MOAD_DATA/moad",
        help="MOAD BOP dataset directory — the folder containing test/, meta/, "
             "models_eval/ (i.e. /home/csrobot/BOP_MOAD_DATA/moad)."
    )
    parser.add_argument(
        "--output-dir", default="/home/csrobot/BOP_MOAD_DATA/predictions",
        help="Output directory for the generated CSV files "
             "(i.e. your BOP_RESULTS_PATH)."
    )
    parser.add_argument(
        "--method-name", default="gt",
        help="Method name prefix for the output filenames. "
             "Output will be: {method-name}_{sensor}_moad-test.csv. "
             "Use 'gt' for perfect GT baseline, 'gt_noise' for noisy GT. "
             "(default: gt)"
    )
    parser.add_argument(
        "--scene-ids", nargs="*", type=int, default=None,
        help="Scene IDs to include (default: all available)."
    )
    parser.add_argument(
        "--noise-t", type=float, default=0.0,
        help="Std dev of Gaussian translation noise in mm (default: 0 = perfect GT)."
    )
    parser.add_argument(
        "--noise-r", type=float, default=0.0,
        help="Std dev of Gaussian rotation noise in degrees (default: 0 = perfect GT)."
    )

    args = parser.parse_args()
    main(args)
