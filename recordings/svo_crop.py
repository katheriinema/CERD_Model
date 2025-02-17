#!/usr/bin/env python3
import sys
import pyzed.sl as sl


# ======== USER CONFIGURATION =========
# Input SVO file path (change to your input file path)
input_svo_file = r"C:\Users\arche\OneDrive\Desktop\cerd_videos\pour_water_2.svo2"

# Output SVO file path (must end with .svo or .svo2)
output_svo_file = r"C:\Users\arche\OneDrive\Desktop\cerd_videos\pour_water_cropped.svo2"

# Number of frames to skip at the beginning of the SVO file
crop_beginning = 200

# Number of frames to skip at the end of the SVO file
crop_end = 300
# ========================================

def main():
    # Create and configure the ZED camera instance.
    cam = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(input_svo_file)
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.async_image_retrieval = False  # Use synchronous retrieval
    init_params.camera_resolution = sl.RESOLUTION.HD720

    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Error opening SVO file:", input_svo_file)
        sys.exit(1)

    runtime = sl.RuntimeParameters()

    # Create a Mat for forcing image retrieval.
    image = sl.Mat()

    # --- Determine Frame Increment (SVO frames per image frame) ---
    initial_pos = cam.get_svo_position()
    if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        print("Error grabbing frame for measurement.")
        cam.close()
        sys.exit(1)
    new_pos = cam.get_svo_position()
    frame_increment = new_pos - initial_pos
    if frame_increment <= 0:
        frame_increment = 1
    print("Frame increment per grab:", frame_increment)

    # Total SVO frames and estimated total image frames.
    total_svo_frames = cam.get_svo_number_of_frames()
    total_image_frames = total_svo_frames // frame_increment
    print("Total SVO frames:", total_svo_frames)
    print("Estimated total image frames:", total_image_frames)

    # Validate crop parameters.
    if crop_beginning < 0 or crop_end < 0 or (crop_beginning + crop_end) >= total_image_frames:
        print("Invalid cropping parameters. Total image frames:", total_image_frames,
              "crop_beginning:", crop_beginning, "crop_end:", crop_end)
        cam.close()
        sys.exit(1)

    # --- Skip the first crop_beginning image frames ---
    # (Note: We already grabbed one frame for measurement; count that as image frame 1)
    skipped = 1
    print("\nSkipping image frames until reaching:", crop_beginning, "...")
    while skipped < crop_beginning:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            skipped += 1
            if skipped % 50 == 0:
                print("Skipped {} image frames...".format(skipped))
        else:
            print("Error grabbing frame while skipping.")
            cam.close()
            sys.exit(1)
    print("Finished skipping {} image frames.".format(skipped))

    # --- Enable Recording ---
    if not (output_svo_file.endswith(".svo") or output_svo_file.endswith(".svo2")):
        print("Output file must have .svo or .svo2 extension.")
        cam.close()
        sys.exit(1)
    rec_params = sl.RecordingParameters(output_svo_file, sl.SVO_COMPRESSION_MODE.H264)
    err = cam.enable_recording(rec_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error enabling recording:", err)
        cam.close()
        sys.exit(1)
    print("Recording enabled to:", output_svo_file)

    # --- Record the desired image frames ---
    # Calculate how many image frames to record:
    frames_to_record = total_image_frames - crop_beginning - crop_end
    print("Will record {} image frames.".format(frames_to_record))

    recorded = 0
    while recorded < frames_to_record:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            recorded += 1
            if recorded % 10 == 0:
                print("Recorded {} / {} image frames".format(recorded, frames_to_record), end="\r")
        else:
            print("Error grabbing frame during recording.")
            break
    print("\nFinished recording.")

    cam.disable_recording()
    cam.close()

if __name__ == "__main__":
    main()