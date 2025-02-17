import pyzed.sl as sl
import cv2
import os

def extract_key_frames(svo_path, output_folder, frame_interval=25):
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the ZED camera with the SVO file.
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False  # Process as fast as possible.
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # High-accuracy depth.
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open SVO file:", err)
        return

    runtime_parameters = sl.RuntimeParameters()
    image_zed = sl.Mat()

    frame_id = 0
    key_frame_count = 0

    while True:
        grab_state = zed.grab(runtime_parameters)
        if grab_state != sl.ERROR_CODE.SUCCESS:
            print("Finished processing SVO or encountered an error.")
            break

        # Retrieve the left image (BGRA format)
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame = image_zed.get_data()  # Numpy array in BGRA format

        # Save every 'frame_interval'-th frame.
        if frame_id % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{frame_id:06d}.png")
            cv2.imwrite(filename, frame)
            key_frame_count += 1
            print(f"Saved key frame: {filename}")

        frame_id += 1

    print(f"Extraction complete. Total key frames saved: {key_frame_count}")
    zed.close()

if __name__ == "__main__":
    svo_file_path = r"C:\Users\arche\OneDrive\Desktop\cerd_videos\Data\pour_water_1.svo2"  # Update path if needed.
    output_dir = r"extracted_frames"  # Folder to store key frames.
    extract_key_frames(svo_file_path, output_dir, frame_interval=25)
