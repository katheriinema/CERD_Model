import os
import re
import open3d as o3d

# Path to the folder where your PLY files are saved.
PLY_FOLDER = "/home/kma/CERD_Model/dataset/pour_water_02/point_clouds"

# List all PLY files.
ply_files = [os.path.join(PLY_FOLDER, f) for f in os.listdir(PLY_FOLDER) if f.endswith(".ply")]

def parse_filename(filename):
    """
    Parse a filename of the form: 
    <timestamp>_mask_<mask_index>.ply
    For example: "1740349022.000_mask_0.ply"
    Returns a tuple: (timestamp as float, mask_index as int)
    """
    base = os.path.basename(filename)
    m = re.match(r"([0-9\.]+)_mask_(\d+)\.ply", base)
    if m:
        timestamp = float(m.group(1))
        mask_index = int(m.group(2))
        return (timestamp, mask_index)
    else:
        return (float('inf'), 0)

# Sort the PLY files by timestamp, then by mask index.
sorted_ply_files = sorted(ply_files, key=lambda f: parse_filename(f))

# Load the point clouds.
pcd_list = [o3d.io.read_point_cloud(p) for p in sorted_ply_files]

# Visualize each point cloud individually (optional).
for i, pcd in enumerate(pcd_list):
    print(f"Visualizing point cloud {i+1}/{len(pcd_list)}: {sorted_ply_files[i]}")
    o3d.visualization.draw_geometries([pcd],
                                      window_name=f"Point Cloud {i+1}",
                                      width=800,
                                      height=600)

