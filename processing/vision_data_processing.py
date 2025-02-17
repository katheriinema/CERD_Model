# Saving PNG RGB images from the left and right camera, PNG depth map of the ZED stereo camera for each key frame

import os
import cv2
import pyzed.sl as sl
import numpy as np

# Input SVO file path and output dataset folder
svo_file_path = r"C:\Users\arche\code\CERD_Model\data_test\SVO\pour_water_1.svo2"
output_folder = r"C:\Users\arche\code\CERD_Model\dataset\pour_water_02"
sequence_id = "pour_water_02"  # Modify as needed for your session naming

def create_output_folders(base_folder):
    """
    Creates output folders for left/right RGB images, depth maps, and video.
    """
    rgb_left_folder = os.path.join(base_folder, "rgb_left")
    rgb_right_folder = os.path.join(base_folder, "rgb_right")
    depth_folder = os.path.join(base_folder, "depth")
    video_folder = os.path.join(base_folder, "video")

    os.makedirs(rgb_left_folder, exist_ok=True)
    os.makedirs(rgb_right_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    return rgb_left_folder, rgb_right_folder, depth_folder, video_folder

def main():
    # Create output folders and define video output path
    rgb_left_folder, rgb_right_folder, depth_folder, video_folder = create_output_folders(output_folder)
    output_video_path = os.path.join(video_folder, f"{sequence_id}.mp4")

    # Initialize the ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    # Depending on your SDK version, you might need:
    # init_params.input.set_from_svo_file(svo_file_path)
    # or the following (if your SDK supports it):
    init_params.set_from_svo_file(svo_file_path)
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 1280x720 resolution
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA         # Ultra depth mode

    # Open the SVO file
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening SVO file: {err}")
        exit(1)

    # Setup video writer for MP4 output (using left camera frames)
    width, height = 1280, 720
    fps = 30  # Adjust FPS as needed or retrieve via zed.get_camera_information().camera_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create sl.Mat objects to hold images and depth
    image_left = sl.Mat()
    image_right = sl.Mat()
    depth = sl.Mat()

    key_frame_interval = 25  # Save every 25th frame as key frame
    frame_id = 0

    print("Starting to grab frames...")
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        frame_id += 1

        # Always retrieve the left image for video output
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        left_img = image_left.get_data()
        left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_BGRA2BGR)
        video_writer.write(left_img_bgr)

        # Process key frames for saving PNG images
        if frame_id % key_frame_interval == 0:
            # Get the timestamp (in seconds) for naming
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()
            timestamp_str = f"{timestamp:.3f}"  # Format to 3 decimal places

            # Retrieve right image and depth map for key frames
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            right_img = image_right.get_data()
            depth_img = depth.get_data()

            right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_BGRA2BGR)

            # Construct file names using the naming template: {sequence_id}_{timestamp}.png
            left_filename = os.path.join(rgb_left_folder, f"{sequence_id}_{timestamp_str}.png")
            right_filename = os.path.join(rgb_right_folder, f"{sequence_id}_{timestamp_str}.png")
            depth_filename = os.path.join(depth_folder, f"{sequence_id}_{timestamp_str}.npy")

            # Save key frame images
            cv2.imwrite(left_filename, left_img_bgr)
            cv2.imwrite(right_filename, right_img_bgr)
            np.save(depth_filename, depth_img)

            print(f"Saved key frame {frame_id} at timestamp {timestamp_str}")

    # Release resources
    video_writer.release()
    zed.close()
    print("Finished processing SVO file.")

if __name__ == "__main__":
    main()
