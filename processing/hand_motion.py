import pyzed.sl as sl
import cv2
import mediapipe as mp
import numpy as np
import csv

# --- Object Segmentation Imports ---
# Here we use the Segment Anything Model (SAM) with its automatic mask generator.
# You can update this to SAM2 if available.
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -----------------------------
# Utility functions for object segmentation
# -----------------------------

def filter_manipulated_objects(masks, prompt="manipulated object"):
    """
    Placeholder to filter segmentation masks using a vision–language model such as Florence‑2.
    In a real implementation, you would pass each mask (or its image crop) along with a text prompt
    to Florence‑2 and retain only those masks that score highly.
    """
    # For now, simply return all masks.
    return masks

def compute_3d_object_boundaries(mask, point_cloud):
    """
    Given a binary mask (numpy array) and the ZED point cloud, compute the 3D boundary
    (min and max coordinates) for the segmented object.
    """
    indices = np.where(mask)
    if indices[0].size == 0:
        return None
    pts = []
    # Iterate over pixels in the mask. (This may be slow; consider downsampling in practice.)
    for y, x in zip(indices[0], indices[1]):
        err, point3D = point_cloud.get_value(x, y)
        if err == sl.ERROR_CODE.SUCCESS:
            pts.append(point3D)
    pts = np.array(pts)
    if pts.shape[0] == 0:
        return None
    min_coords = np.min(pts, axis=0)
    max_coords = np.max(pts, axis=0)
    return (min_coords, max_coords)

def compute_hand_interaction_status(hand_landmarks, frame_bgr, object_masks):
    """
    Check whether any of the hand’s fingertips (landmarks 4, 8, 12, 16, 20)
    fall inside any of the segmented object masks.
    If yes, return "Closed" (i.e. interacting), otherwise "Open".
    """
    fingertip_ids = [4, 8, 12, 16, 20]
    for fid in fingertip_ids:
        x_px = int(hand_landmarks.landmark[fid].x * frame_bgr.shape[1])
        y_px = int(hand_landmarks.landmark[fid].y * frame_bgr.shape[0])
        for mask_dict in object_masks:
            mask = mask_dict['segmentation']  # a boolean (or 0/1) mask of shape (H,W)
            # Ensure coordinates are within mask bounds
            if y_px >= mask.shape[0] or x_px >= mask.shape[1]:
                continue
            if mask[y_px, x_px]:
                return "Closed"
    return "Open"

# -----------------------------
# Hand Analysis Utility Functions
# -----------------------------

def hand_open_closed_status(landmarks_3d):
    """
    Original heuristic based on the distances between fingertips.
    """
    fingertip_ids = [4, 8, 12, 16, 20]
    valid_points = []
    for fid in fingertip_ids:
        if fid < len(landmarks_3d) and landmarks_3d[fid] is not None:
            valid_points.append(np.array(landmarks_3d[fid]))
    if len(valid_points) < 2:
        return "Unknown"
    distances = []
    for i in range(len(valid_points) - 1):
        dist = np.linalg.norm(valid_points[i] - valid_points[i + 1])
        distances.append(dist)
    if len(distances) == 0:
        return "Unknown"
    avg_dist = np.mean(distances)
    return "Closed" if avg_dist < 0.05 else "Open"

def compute_hand_orientation(landmarks_3d):
    """
    Compute the angle in the XY plane from the wrist (landmark 0) to the middle finger tip (landmark 12).
    """
    if len(landmarks_3d) < 21:
        return "Unknown"
    if landmarks_3d[0] is None or landmarks_3d[12] is None:
        return "Unknown"
    wrist = np.array(landmarks_3d[0])
    middle_tip = np.array(landmarks_3d[12])
    vec = middle_tip - wrist
    angle_degs = np.degrees(np.arctan2(vec[1], vec[0]))
    return f"{angle_degs:.2f} deg (XY-plane)"

# -----------------------------
# Main Script
# -----------------------------

def main():
    # A) Initialize ZED Camera from SVO
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(r"C:\Users\arche\OneDrive\Desktop\cerd_videos\Data\pour_water_1.svo2")
    init_params.svo_real_time_mode = False  # Process as fast as possible
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # High-accuracy depth
    init_params.coordinate_units = sl.UNIT.METER  # Depth in meters
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED SVO file:", err)
        exit(1)
    runtime_params = sl.RuntimeParameters()

    # B) Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # C) Initialize SAM for Object Segmentation
    # You must specify the correct checkpoint path.
    sam_checkpoint = r"C:\Users\arche\code\sam_vit_h_4b8939.pth"  # <--- UPDATE THIS PATH
    model_type = "vit_h"  # You can change this depending on your model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # If using GPU, move the model accordingly (e.g., sam.to("cuda"))
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Create an OpenCV window
    cv2.namedWindow("ZED + MediaPipe (SVO)", cv2.WINDOW_NORMAL)

    # Prepare Mats for image and point cloud retrieval
    image_zed = sl.Mat()
    point_cloud = sl.Mat()

    # Set up CSV for saving hand landmark & trajectory data.
    # CSV header: FrameID, HandIndex, LandmarkIndex, X, Y, Z, Status, Orientation
    csv_file = open("hand_data_svo.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["FrameID", "HandIndex", "LandmarkIndex", "X", "Y", "Z", "Status", "Orientation"])

    frame_id = 0
    # Dictionary to hold trajectory visualization points (using average fingertip 2D positions)
    trajectories = {}  # key: hand_idx, value: list of (x, y) tuples

    while True:
        grab_state = zed.grab(runtime_params)
        if grab_state != sl.ERROR_CODE.SUCCESS:
            print("Stopping because of error or end of SVO. Code:", grab_state)
            break

        # Retrieve image and point cloud from the ZED
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

        # Convert BGRA to BGR then to RGB for MediaPipe
        frame_bgra = image_zed.get_data()
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Object Segmentation via SAM ---
        masks = mask_generator.generate(frame_bgr)
        # Optionally, filter the masks using a vision–language model (e.g., Florence‑2)
        object_masks = filter_manipulated_objects(masks, prompt="manipulated object")

        # Visualize object segmentation:
        for mask_dict in object_masks:
            seg_mask = mask_dict['segmentation']  # binary mask (numpy array)
            ys, xs = np.where(seg_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            # Draw 2D bounding box on the frame
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            # Compute (and optionally display) the 3D boundaries of the object
            boundaries_3d = compute_3d_object_boundaries(seg_mask, point_cloud)
            if boundaries_3d is not None:
                min_coords, max_coords = boundaries_3d
                cv2.putText(frame_bgr, f"3D: {min_coords}-{max_coords}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # --- Hand Detection with MediaPipe ---
        results = hands_detector.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw 2D landmarks on the frame
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get 3D positions for 21 landmarks using the ZED point cloud
                hand_3d_positions = [None] * 21
                for i in range(21):
                    x_n = hand_landmarks.landmark[i].x
                    y_n = hand_landmarks.landmark[i].y
                    # Convert normalized coordinates to pixel coordinates
                    x_px = int(x_n * frame_bgr.shape[1])
                    y_px = int(y_n * frame_bgr.shape[0])
                    x_px = max(0, min(frame_bgr.shape[1] - 1, x_px))
                    y_px = max(0, min(frame_bgr.shape[0] - 1, y_px))
                    err_code, point3D = point_cloud.get_value(x_px, y_px)
                    if err_code == sl.ERROR_CODE.SUCCESS:
                        X, Y, Z = point3D[0], point3D[1], point3D[2]
                        if not np.isnan(X) and not np.isnan(Y) and not np.isnan(Z):
                            hand_3d_positions[i] = (X, Y, Z)
                        else:
                            hand_3d_positions[i] = None
                    else:
                        hand_3d_positions[i] = None

                # --- Compute Hand Status ---
                # 1. Original heuristic based on finger separation
                status_distance = hand_open_closed_status(hand_3d_positions)
                # 2. Interaction check based on whether fingertips fall inside an object mask
                status_interaction = compute_hand_interaction_status(hand_landmarks, frame_bgr, object_masks)
                # Combine the two: if the interaction test indicates contact, mark as "Closed"
                status = "Closed" if status_interaction == "Closed" else status_distance
                orientation = compute_hand_orientation(hand_3d_positions)
                print(f"Frame: {frame_id}, Hand: {hand_idx}, Status: {status}, Orientation: {orientation}")

                # Write 21 landmark rows for the hand to CSV
                for lm_idx, coords in enumerate(hand_3d_positions):
                    if coords is not None:
                        X_val, Y_val, Z_val = coords
                    else:
                        X_val, Y_val, Z_val = (None, None, None)
                    csv_writer.writerow([frame_id, hand_idx, lm_idx, X_val, Y_val, Z_val, status, orientation])

                # --- Trajectory Logging ---
                # Compute the average fingertip location (using landmarks 4, 8, 12, 16, 20)
                fingertip_ids = [4, 8, 12, 16, 20]
                sum_x_3d, sum_y_3d, sum_z_3d, count3d = 0, 0, 0, 0
                sum_x_px, sum_y_px, count2d = 0, 0, 0
                for fid in fingertip_ids:
                    # 3D average for CSV logging
                    if hand_3d_positions[fid] is not None:
                        X3, Y3, Z3 = hand_3d_positions[fid]
                        sum_x_3d += X3
                        sum_y_3d += Y3
                        sum_z_3d += Z3
                        count3d += 1
                    # 2D average for visualization
                    x_n = hand_landmarks.landmark[fid].x
                    y_n = hand_landmarks.landmark[fid].y
                    x_px = int(x_n * frame_bgr.shape[1])
                    y_px = int(y_n * frame_bgr.shape[0])
                    sum_x_px += x_px
                    sum_y_px += y_px
                    count2d += 1
                avg_3d_x = sum_x_3d / count3d if count3d > 0 else None
                avg_3d_y = sum_y_3d / count3d if count3d > 0 else None
                avg_3d_z = sum_z_3d / count3d if count3d > 0 else None
                avg_px_x = int(sum_x_px / count2d) if count2d > 0 else None
                avg_px_y = int(sum_y_px / count2d) if count2d > 0 else None

                # Write a trajectory row (using "Trajectory" as LandmarkIndex)
                csv_writer.writerow([frame_id, hand_idx, "Trajectory", avg_3d_x, avg_3d_y, avg_3d_z, status, orientation])

                # For on-screen visualization, update trajectory points every 25 frames
                if frame_id % 25 == 0 and avg_px_x is not None and avg_px_y is not None:
                    if hand_idx not in trajectories:
                        trajectories[hand_idx] = []
                    trajectories[hand_idx].append((avg_px_x, avg_px_y))
                    cv2.circle(frame_bgr, (avg_px_x, avg_px_y), 5, (0, 255, 0), -1)

        # Draw continuous trajectories by connecting the dots (for each hand)
        for hand_idx, points in trajectories.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame_bgr, points[i - 1], points[i], (255, 0, 0), 2)
            # Re-draw the dots to ensure visibility
            for point in points:
                cv2.circle(frame_bgr, point, 5, (0, 255, 0), -1)

        # Display the annotated frame
        cv2.imshow("ZED + MediaPipe (SVO)", frame_bgr)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    zed.close()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
