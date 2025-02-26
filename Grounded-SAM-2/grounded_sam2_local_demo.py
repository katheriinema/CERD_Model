import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import matplotlib.pyplot as plt  # For colormap

# --- Configuration Parameters ---
TEXT_PROMPT = "bottle. cup."  # Adjust as needed.
RGB_FOLDER = "/home/kma/CERD_Model/dataset/pour_water_02/rgb_left"   # Folder containing RGB left images.
DEPTH_FOLDER = "/home/kma/CERD_Model/dataset/pour_water_02/depth"     # Folder containing depth maps (npy files).
OUTPUT_MASK_FOLDER = Path("outputs/grounded_sam2_local_demo")          # Where segmentation masks are saved.
OUTPUT_MASK_FOLDER.mkdir(parents=True, exist_ok=True)
POINT_CLOUD_OUTPUT_FOLDER = Path("/home/kma/CERD_Model/dataset/pour_water_02/point_clouds")
POINT_CLOUD_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.70
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUMP_JSON_RESULTS = True

# --- Load Camera Intrinsics ---
calib_file = "/home/kma/CERD_Model/dataset/pour_water_02/calibration.json"
with open(calib_file, "r") as f:
    calib = json.load(f)
intrinsics = calib["intrinsic_left"]
fx = intrinsics["fx"]
fy = intrinsics["fy"]
cx = intrinsics["cx"]
cy = intrinsics["cy"]
print(f"Loaded intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# --- Initialize Models ---
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# --- Helper Functions ---
def debug_overlay_mask_on_depth(mask, depth_map, timestamp):
    """Overlay mask on depth map to check alignment."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_map, cmap='gray')
    plt.title(f"Depth Map {timestamp}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask
    plt.title(f"Overlayed Mask {timestamp}")
    plt.colorbar()
    plt.show()


def extract_timestamp(filename):
    # Assumes filename format: "pour_water_02_<timestamp>.png"
    base = os.path.basename(filename)
    timestamp = base.replace("pour_water_02_", "").replace(".png", "")
    return timestamp

def process_grounded_sam2_for_image(img_path):
    """Process one RGB image with Grounded SAM2 and save segmentation masks."""
    timestamp = extract_timestamp(img_path)
    print(f"Processing image for timestamp: {timestamp}")
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)
    
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    h, w, _ = image_source.shape
    print(f"Original boxes (normalized): {boxes}")
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    print(f"Converted boxes: {input_boxes}")
    
    # Get segmentation masks using SAM2.
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,  # Change to True if desired.
    )
    if masks.ndim == 4:
        if masks.shape[1] == 3:
            masks = masks[:, 0, :, :]
        elif masks.shape[1] == 1:
            masks = masks.squeeze(1)
    
    # Save each segmentation mask with timestamp in filename.
    for i, mask in enumerate(masks):
        mask_uint8 = (mask.astype(np.uint8)) * 255
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        out_filename = os.path.join(OUTPUT_MASK_FOLDER, f"{timestamp}_mask_{i}.png")
        cv2.imwrite(out_filename, mask_uint8)
        print(f"Saved segmentation mask: {out_filename}")
    
    return timestamp

def generate_point_cloud_from_mask(depth_map, mask, fx, fy, cx, cy):
    # Resize mask to match depth map resolution.
    mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
    print("DEBUG: Mask shape after resize:", mask.shape)
    # (Optional) Debug: Visualize overlay of mask and depth map.

    ys, xs = np.where(mask > 0)
    print("DEBUG: Number of non-zero pixels in mask:", len(ys))
    if len(ys) == 0:
        print("WARNING: Mask has no non-zero pixels!")
        return np.empty((0, 3))
    
    depths = depth_map[ys, xs]
    valid = depths > 0
    num_valid = np.count_nonzero(valid)
    print("DEBUG: Number of valid depth pixels (depth > 0):", num_valid)
    if num_valid == 0:
        print("WARNING: No valid depth values found for masked region!")
        return np.empty((0, 3))
    
    xs = xs[valid]
    ys = ys[valid]
    depths = depths[valid]
    
    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths
    points = np.vstack((X, Y, Z)).T
    print("DEBUG: Number of points generated:", points.shape[0])
    return points

def add_depth_colors_to_pcd(pcd):
    points = np.asarray(pcd.points)
    if points.size == 0:
        return pcd
    depths = points[:, 2]
    depth_min = depths.min()
    depth_max = depths.max()
    norm_depths = (depths - depth_min) / (depth_max - depth_min + 1e-8)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm_depths)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_point_cloud(points, filename):
    if points.shape[0] == 0:
        print(f"WARNING: No points to save for {filename}. Skipping point cloud generation.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print("DEBUG: Point cloud before downsampling has", np.asarray(pcd.points).shape[0], "points")
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print("DEBUG: Point cloud after voxel downsampling has", np.asarray(pcd.points).shape[0], "points")
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    pcd = add_depth_colors_to_pcd(pcd)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename} with {np.asarray(pcd.points).shape[0]} points")

def process_point_cloud_for_timestamp(timestamp):
    depth_filename = os.path.join(DEPTH_FOLDER, f"pour_water_02_{timestamp}.npy")
    print(f"DEBUG: Loading depth map from {depth_filename}")
    try:
        depth_map = np.load(depth_filename)
        print("DEBUG: Loaded depth map with shape:", depth_map.shape)
    except Exception as e:
        print(f"ERROR: Could not load depth map {depth_filename}: {e}")
        return
    
    # Find segmentation masks for this timestamp.
    segmentation_files = [f for f in os.listdir(OUTPUT_MASK_FOLDER) if f.startswith(timestamp) and f.endswith(".png")]
    if not segmentation_files:
        print(f"No segmentation masks found for timestamp {timestamp}")
        return
    for seg_file in segmentation_files:
        mask_path = os.path.join(OUTPUT_MASK_FOLDER, seg_file)
        print(f"DEBUG: Processing segmentation mask {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask {mask_path}")
            continue
        
        # Resize mask to match depth map dimensions.
        mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
        print("DEBUG: Mask shape after resize:", mask.shape)
        
        print("DEBUG: Mask unique values before thresholding:", np.unique(mask))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        debug_overlay_mask_on_depth(mask, depth_map, timestamp)

        # Increase kernel size and use morphological closing.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        print("DEBUG: Mask unique values after processing:", np.unique(mask))
        
        points = generate_point_cloud_from_mask(depth_map, mask, fx, fy, cx, cy)
        if points.shape[0] == 0:
            print(f"WARNING: No points generated for mask {mask_path}")
        ply_filename = os.path.join(POINT_CLOUD_OUTPUT_FOLDER, f"{timestamp}_{seg_file.replace('.png','.ply')}")
        save_point_cloud(points, ply_filename)

# --- Batch Processing Pipeline ---

# Step 1: Process all RGB left images with Grounded SAM2 to generate segmentation masks.
rgb_files = [os.path.join(RGB_FOLDER, f) for f in os.listdir(RGB_FOLDER) if f.endswith(".png")]
rgb_files.sort()
timestamps = []
for rgb_file in rgb_files:
    print(f"Processing image: {rgb_file}")
    ts = process_grounded_sam2_for_image(rgb_file)
    if ts is not None:
        timestamps.append(ts)
timestamps = sorted(set(timestamps))
print("Timestamps processed:", timestamps)

# Step 2: For each timestamp, generate 3D point clouds using corresponding depth maps and segmentation masks.
for ts in timestamps:
    print(f"Processing point cloud for timestamp: {ts}")
    process_point_cloud_for_timestamp(ts)
