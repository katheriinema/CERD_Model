from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py',
            trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Keypoint detector
    cpm = ViTPoseModel(device)

    # Renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Create the main output folder if it doesn't exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Create subfolders for CSV outputs
    joints_folder = os.path.join(args.out_folder, "joints_csv")
    orientation_folder = os.path.join(args.out_folder, "orientation_csv")
    os.makedirs(joints_folder, exist_ok=True)
    os.makedirs(orientation_folder, exist_ok=True)

    # Get all demo images
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Process each image
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Use ViTPose to detect keypoints for each bounding box
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            # Left hand
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                        keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            # Right hand
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                        keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Build a dataset for HaMeR
        dataset = ViTDetDataset(
            model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2 * batch['right'] - 1)
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            batch_size_ = batch['img'].shape[0]
            for n in range(batch_size_):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])

                white_img = (
                    torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255
                ) / (DEFAULT_STD[:, None, None] / 255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                    DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                regression_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )

                if args.side_view:
                    side_img = renderer(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        out['pred_cam_t'][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True
                    )
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                out_img_name = f'{img_fn}_{person_id}.png'
                cv2.imwrite(os.path.join(args.out_folder, out_img_name), 255 * final_img[:, :, ::-1])

                # Extract 778 MANO mesh vertices
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_flag = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_flag - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_flag)

                # =======================
                # Save 3D joints as CSV and .npy
                # =======================
                joints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # Expected shape: (21, 3)
                import csv
                joints_csv_path = os.path.join(joints_folder, f'{img_fn}_{person_id}_joints.csv')
                with open(joints_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['joint_index', 'x', 'y', 'z'])
                    for idx_j, joint in enumerate(joints_3d):
                        writer.writerow([idx_j] + joint.tolist())
                print(f"Saved 3D hand joints to {joints_csv_path}")

                # Save joints as .npy file
                joints_npy_folder = os.path.join(args.out_folder, "joints_npy")
                os.makedirs(joints_npy_folder, exist_ok=True)
                joints_npy_path = os.path.join(joints_npy_folder, f'{img_fn}_{person_id}_hand_3d.npy')
                np.save(joints_npy_path, joints_3d)
                print(f"Saved 3D hand joints as .npy to {joints_npy_path}")

                # =======================
                # Save orientation as CSV and .npy
                # =======================
                wrist = joints_3d[0]
                middle_tip = joints_3d[9]  # Adjust if necessary for your joint ordering.
                orientation_vector = middle_tip - wrist

                orientation_csv_path = os.path.join(orientation_folder, f'{img_fn}_{person_id}_orientation.csv')
                with open(orientation_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['ox', 'oy', 'oz'])
                    writer.writerow(orientation_vector.tolist())
                print(f"Saved hand orientation to {orientation_csv_path}")

                orientation_npy_folder = os.path.join(args.out_folder, "orientation_npy")
                os.makedirs(orientation_npy_folder, exist_ok=True)
                orientation_npy_path = os.path.join(orientation_npy_folder, f'{img_fn}_{person_id}_orientation.npy')
                np.save(orientation_npy_path, orientation_vector)
                print(f"Saved hand orientation as .npy to {orientation_npy_path}")

                # Optionally save MANO meshes
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right_flag)
                    tmesh_name = f'{img_fn}_{person_id}.obj'
                    tmesh.export(os.path.join(args.out_folder, tmesh_name))

        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            alpha_ch = np.ones_like(input_img[:, :, 0:1])
            input_img = np.concatenate([input_img, alpha_ch], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            front_view_name = f'{img_fn}_all.jpg'
            cv2.imwrite(os.path.join(args.out_folder, front_view_name), 255 * input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()