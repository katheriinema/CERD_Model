#!/usr/bin/env python
"""
compare_grasp_state.py

This script loads 3D hand keypoints (saved as .npy files from HaMeR) and object point clouds
(saved as .ply files from Grounded-SAM2) and then computes a simple grasp state, but only for
matching timestamps. Finally, it sorts the results in ascending timestamp order.

File naming assumptions:
- Hand file names look like: "pour_water_02_1740349023.000_0_hand_3d.npy"
  --> We'll parse out the 4th underscore-delimited part as the timestamp (index=3).
- Object file names look like: "1740349023.000_1740349023.000_mask_0.ply"
  --> We'll parse out the 1st underscore-delimited part as the timestamp (index=0).

For each hand–object pair whose timestamps match, we:
  1. Convert hand keypoints from meters -> millimeters (if needed).
  2. Compute the centroid of the 21 keypoints.
  3. Find the min distance from centroid to object point cloud (in mm).
  4. Decide if it's a "grasp" if min distance < DIST_THRESH.
  5. Accumulate these rows and sort them by ascending timestamp.

This ensures we only compare matching frames, and output is sequential by timestamp.
"""

import os
import csv
import glob
import numpy as np
import open3d as o3d

# --- Configuration Parameters ---
HAND_KEYPOINTS_FOLDER = "/home/kma/hamer/demo_out/joints_npy"
OBJECT_PLY_FOLDER = "/home/kma/CERD_Model/dataset/pour_water_02/point_clouds"
OUTPUT_CSV = "/home/kma/CERD_Model/dataset/pour_water_02/grasp.csv"

# Distance threshold (in millimeters) for deciding a "grasp."
DIST_THRESH = 350.0

def parse_hand_timestamp(hand_id: str) -> str:
    """
    E.g. "pour_water_02_1740349023.000_0_hand_3d"
    We'll split on underscores and assume the 4th element is the timestamp (index=3).
    """
    parts = hand_id.split("_")
    # e.g. ["pour","water","02","1740349023.000","0","hand","3d"]
    if len(parts) < 4:
        return ""
    return parts[3]  # "1740349023.000"

def parse_object_timestamp(obj_id: str) -> str:
    """
    E.g. "1740349023.000_1740349023.000_mask_0"
    We'll split on underscores and assume the 1st element is the timestamp (index=0).
    """
    parts = obj_id.split("_")
    # e.g. ["1740349023.000","1740349023.000","mask","0"]
    if len(parts) < 1:
        return ""
    return parts[0]

def parse_timestamp_to_float(ts_str: str) -> float:
    """
    Convert a timestamp string like '1740349023.000' into a float 1740349023.000.
    """
    try:
        return float(ts_str)
    except ValueError:
        return 0.0

def load_hand_keypoints(folder):
    """
    Loads .npy files => {hand_id -> 21x3 array (in millimeters)}.
    """
    hand_files = glob.glob(os.path.join(folder, "*.npy"))
    hand_data = {}
    for f in hand_files:
        base = os.path.basename(f)
        hand_id = os.path.splitext(base)[0]  # e.g. "pour_water_02_1740349023.000_0_hand_3d"
        keypoints_m = np.load(f)  # shape (21, 3) in meters
        # Convert from meters to millimeters:
        keypoints_mm = keypoints_m * 1000.0
        hand_data[hand_id] = keypoints_mm
    return hand_data

def load_object_point_clouds(folder):
    """
    Loads .ply files => {obj_id -> open3d.geometry.PointCloud} (points presumably in mm).
    """
    ply_files = glob.glob(os.path.join(folder, "*.ply"))
    obj_data = {}
    for f in ply_files:
        base = os.path.basename(f)
        obj_id = os.path.splitext(base)[0]
        pcd = o3d.io.read_point_cloud(f)
        obj_data[obj_id] = pcd
    return obj_data

def compute_grasp_state(hand_keypoints_mm, obj_pcd, dist_thresh):
    """
    1) Compute centroid of (21, 3) array (in mm).
    2) For each point in obj_pcd (in mm), compute distance to centroid.
    3) is_grasp = (min_distance < dist_thresh).
    Returns (is_grasp, min_distance).
    """
    centroid = np.mean(hand_keypoints_mm, axis=0)  # shape (3,)
    obj_points_mm = np.asarray(obj_pcd.points)
    if obj_points_mm.size == 0:
        return False, None

    dists = np.linalg.norm(obj_points_mm - centroid, axis=1)
    min_dist = float(np.min(dists))
    is_grasp = (min_dist < dist_thresh)
    return is_grasp, min_dist

def main():
    # 1. Load data
    hand_data = load_hand_keypoints(HAND_KEYPOINTS_FOLDER)  # {hand_id -> (21, 3) in mm}
    obj_data = load_object_point_clouds(OBJECT_PLY_FOLDER)  # {obj_id -> pointcloud in mm}

    # We'll accumulate results, then sort them by ascending timestamp.
    results = []  # each element will be (timestamp_float, hand_id, obj_id, is_grasp, min_dist)

    # 2. Compare only matching timestamps
    for hand_id, hand_kps_mm in hand_data.items():
        hand_ts_str = parse_hand_timestamp(hand_id)  # e.g. '1740349023.000'
        hand_ts_val = parse_timestamp_to_float(hand_ts_str)

        matched_any_object = False
        for obj_id, obj_pcd in obj_data.items():
            obj_ts_str = parse_object_timestamp(obj_id)
            obj_ts_val = parse_timestamp_to_float(obj_ts_str)

            if hand_ts_str and (hand_ts_str == obj_ts_str):
                # Timestamps match as strings
                is_grasp, min_dist = compute_grasp_state(hand_kps_mm, obj_pcd, DIST_THRESH)
                results.append((hand_ts_val, hand_id, obj_id, is_grasp, min_dist))
                matched_any_object = True

        if not matched_any_object:
            print(f"No object matched for hand {hand_id} (timestamp={hand_ts_str})")

    # 3. Sort results by ascending numeric timestamp
    results.sort(key=lambda row: row[0])

    # 4. Write them to CSV (and optionally print)
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["hand_id", "object_id", "grasp", "min_distance_mm"])

        for row in results:
            # row = (timestamp_val, hand_id, obj_id, is_grasp, min_dist)
            _, hand_id, obj_id, grasp, min_dist = row
            writer.writerow([hand_id, obj_id, grasp, f"{min_dist:.4f}" if min_dist else "None"])
            print(f"Time {row[0]} | Hand {hand_id} vs Object {obj_id}: "
                  f"grasp={grasp}, dist={min_dist:.2f} mm")

    print(f"\nGrasp state results saved (sorted) to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
