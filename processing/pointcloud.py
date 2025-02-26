import os
import cv2
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt  # For colormap

# --- Load camera intrinsics automatically from calibration JSON ---
calib_file = "/home/kma/CERD_Model/dataset/pour_water_02/calibration.json"
with open(calib_file, "r") as f:
    calib = json.load(f)
intrinsics = calib["intrinsic_left"]
fx = intrinsics["fx"]
fy = intrinsics["fy"]
cx = intrinsics["cx"]
cy = intrinsics["cy"]
print(f"Loaded intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

def generate_point_cloud_from_mask(depth_map, mask, fx, fy, cx, cy):
    # Get indices where mask is nonzero (assumes mask is a uint8 image with 0 and 255)
    ys, xs = np.where(mask > 0)
    depths = depth_map[ys, xs]
    
    # Filter out invalid depth values (e.g., zeros)
    valid = depths > 0
    xs = xs[valid]
    ys = ys[valid]
    depths = depths[valid]
    
    # Project pixels (u, v) to 3D points using the pinhole camera model
    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths
    points = np.vstack((X, Y, Z)).T  # Shape: (N, 3)
    return points

def add_depth_colors_to_pcd(pcd):
    """
    Assign colors to the point cloud based on the Z (depth) values.
    This function uses the 'viridis' colormap from Matplotlib.
    """
    points = np.asarray(pcd.points)
    if points.size == 0:
        return pcd
    depths = points[:, 2]
    # Normalize depths to [0, 1]
    depth_min = depths.min()
    depth_max = depths.max()
    norm_depths = (depths - depth_min) / (depth_max - depth_min + 1e-8)
    
    # Get colormap from Matplotlib
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm_depths)[:, :3]  # Use only RGB, ignore alpha.
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_point_cloud(points, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optional: Downsample and remove outliers.
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Assign colors based on depth.
    pcd = add_depth_colors_to_pcd(pcd)
    
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

def process_key_frame(timestamp_str, rgb_left_folder, depth_folder, segmentation_folder, fx, fy, cx, cy):
    # Construct filenames for the key frame (adjust naming as needed)
    left_filename = os.path.join(rgb_left_folder, f"pour_water_02_{timestamp_str}.png")
    depth_filename = os.path.join(depth_folder, f"pour_water_02_{timestamp_str}.npy")
    
    # Load the depth map
    depth_map = np.load(depth_filename)
    
    # List segmentation mask files in the folder (using all mask files)
    segmentation_files = [f for f in os.listdir(segmentation_folder) if f.startswith("mask_") and f.endswith(".png")]
    
    if not segmentation_files:
        print("No segmentation files found in", segmentation_folder)
        return
    
    for seg_file in segmentation_files:
        mask_path = os.path.join(segmentation_folder, seg_file)
        # Load the binary mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask {mask_path}")
            continue
        # Ensure mask is binary: threshold at 127
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply dilation to improve mask coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Generate raw 3D points from the (improved) mask and depth map
        points = generate_point_cloud_from_mask(depth_map, mask, fx, fy, cx, cy)
        
        # Save the processed point cloud to a PLY file (with depth-based colors)
        ply_filename = os.path.join(segmentation_folder, seg_file.replace(".png", ".ply"))
        save_point_cloud(points, ply_filename)

# --- Example Usage ---
# Define your directories using Linux-style paths
rgb_left_folder = "/home/kma/CERD_Model/dataset/pour_water_02/rgb_left"
depth_folder = "/home/kma/CERD_Model/dataset/pour_water_02/depth"
segmentation_folder = "/home/kma/CERD_Model/Grounded-SAM-2/outputs/grounded_sam2_local_demo/1740349023.000_mask_1.png"  # Folder where your SAM2 masks are saved

# Specify the timestamp string corresponding to the key frame (e.g., "1739654380.000")
timestamp_str = "1740349023.000"  # Replace with the actual timestamp

# Process the key frame to generate processed point clouds for each segmented object
process_key_frame(timestamp_str, rgb_left_folder, depth_folder, segmentation_folder, fx, fy, cx, cy)
