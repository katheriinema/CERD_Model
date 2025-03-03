#!/usr/bin/env python
import open3d as o3d
import numpy as np
import os

def print_ply_scale(ply_file):
    # Load the point cloud from the PLY file.
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    
    if points.size == 0:
        print(f"Error: No points found in {ply_file}")
        return

    # Compute the minimum and maximum coordinates along each axis.
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    extents = max_vals - min_vals

    # Print the results.
    print(f"PLY file: {ply_file}")
    print("Minimum coordinates (x, y, z):", min_vals)
    print("Maximum coordinates (x, y, z):", max_vals)
    print("Extents (width, height, depth):", extents)
    print("Approximate scale (max extent):", np.max(extents))

if __name__ == '__main__':
    # Update the path below to the PLY file you want to inspect.
    ply_file_path = "/home/kma/CERD_Model/dataset/pour_water_02/point_clouds/1740349023.000_1740349023.000_mask_1.ply"
    
    if os.path.exists(ply_file_path):
        print_ply_scale(ply_file_path)
    else:
        print(f"File not found: {ply_file_path}")
