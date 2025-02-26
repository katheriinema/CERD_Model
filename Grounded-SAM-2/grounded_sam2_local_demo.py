import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Hyper parameters
"""
TEXT_PROMPT = "bottle. cup."
IMG_PATH = "/home/kma/CERD_Model/dataset/pour_water_02/rgb_left/pour_water_02_1740349023.000.png"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Setup the input image and text prompt for SAM2 and Grounding DINO.
# VERY important: text queries need to be lowercased and end with a dot.
text = TEXT_PROMPT
img_path = IMG_PATH

image_source, image = load_image(img_path)
sam2_predictor.set_image(image_source)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# Process the box prompt for SAM2.
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

# Enable autocast for bfloat16 (if applicable)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Get segmentation masks from SAM2.
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# Print shape for debugging.
print("Masks shape before processing:", masks.shape)
# If masks have shape (n, 3, H, W), convert to (n, H, W) by taking the first channel.
if masks.ndim == 4 and masks.shape[1] == 3:
    masks = masks[:, 0, :, :]
    print("Converted masks shape to:", masks.shape)
elif masks.ndim == 4 and masks.shape[1] == 1:
    masks = masks.squeeze(1)
    print("Squeezed masks shape to:", masks.shape)
else:
    print("Masks shape remains:", masks.shape)

# ----- Save Each Binary Mask and its Contours as PNG Images for Inspection -----
for i, mask in enumerate(masks):
    # Convert boolean mask to 0-255 image.
    mask_uint8 = (mask.astype(np.uint8)) * 255
    # Ensure mask is binary: threshold at 127
    _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    
    # Save the binary mask.
    mask_filename = os.path.join(OUTPUT_DIR, f"mask_{i}.png")
    cv2.imwrite(mask_filename, mask_uint8)
    print(f"Saved binary mask for object {i} to {mask_filename}")
    
# ----------------------------------------------------------------

confidences = confidences.numpy().tolist()
class_names = labels
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(class_names, confidences)
]

# Visualize image with supervision's useful API.
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, H, W)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

"""
Dump the results in standard format and save as JSON files.
"""
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    input_boxes_list = input_boxes.tolist()
    scores_list = scores.tolist()
    results_dict = {
        "image_path": img_path,
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
        ],
        "box_format": "xyxy",
        "img_width": image_source.shape[1],
        "img_height": image_source.shape[0],
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
