# CERD_Model
Collect real world first-person perspective dataset designed for learning-based robotics manipulation at a low cost through computer vision model

## Overview
The **Human POV Manipulation Dataset** is a first-person perspective dataset designed for learning-based robotics manipulation at a low cost of collecting real-world data. It is captured using a **head-mounted ZED stereo camera**, providing **RGB, depth, and motion data** that are synchronized for **robotics learning, imitation learning, and vision-based control**.

## Dataset Information

```json
{
    "dataset_name": "Human POV Manipulation Dataset",
    "version": "1.0.0",
    "description": "This dataset contains first-person perspective videos and sensor data of human hand manipulation tasks. Data is captured using a head-mounted ZED stereo camera that provides synchronized RGB, depth, and motion information. It is ideal for research in learning-based robotics manipulation, imitation learning, and vision-based control.",
    "contact": {
      "name": "Archer Lin",
      "email": "lin1524@purdue.edu",
      "institution": "Purdue University"
    },
    "citation": "If you use this dataset in your work, please cite: [ Citation Placeholder ].",
    "license": "CC BY 4.0"
}
```

## Data Format and Structure
Each recorded session is stored in a dedicated subfolder containing separate folders and files for **RGB images, depth maps, videos, and synchronized sensor data**.

### File Structure
```
📂 dataset/
 ├── 📂 Task_Name_SessionID/
 │   ├── 📂 rgb_left/         # RGB images from the left camera (1280x720)
 │   ├── 📂 rgb_right/        # RGB images from the right camera (1280x720)
 │   ├── 📂 depth/            # Depth maps corresponding to RGB frames
 │   ├── 📂 video/            # MP4 video of the manipulation task
 │   ├── 📄 info.json         # Metadata describing the task and objects
 │   ├── 📄 result.csv        # Time-synchronized motion data (hand, object, camera, world coordinates)
 │   ├── 📄 calibration.json  # Camera calibration parameters
```

### Metadata Fields
- **info.json**
  - `task_name`, `task_description`, `object_used`, `duration`, `environment`, `annotations`
- **result.csv**
  - `sequence_id`, `timestamp`, `hand_position_x/y/z`, `hand_rotation_x/y/z`, `hand_grasp_state`
  - `object_position_x/y/z`, `object_rotation_x/y/z`
  - `camera_position_x/y/z`, `camera_orientation_x/y/z/w`
  - `world_position_x/y/z`, `world_orientation_x/y/z/w`
- **calibration.json**
  - `intrinsic_left`, `intrinsic_right`, `distortion_coefficients_left`, `distortion_coefficients_right`, `extrinsic_matrix`

## Acquisition Details
- **Camera**: ZED Stereo Camera
- **Mounting**: Head-mounted
- **Capture Period**: Start Date: 2025-02-15 | End Date: TBD
- **Environment**: Indoor & Outdoor

## ZED Stereo Camera Video Processing
Since we use the **ZED Stereo Camera**, the **ZED Python API** is required to process **SVO** (Stereo Video Object) files.

### Installation
1. Download and install **ZED SDK**: [ZED SDK Download](https://www.stereolabs.com/developers/release)
2. Install **ZED Python API**: [Python API Installation](https://www.stereolabs.com/docs/app-development/python/install)

### Recording Video
```bash
python svo_recording.py --output_svo_file "C:\Users\arche\OneDrive\Desktop\cerd_videos\Raw_SVO\[name].svo2"
```

### Playback Video
```bash
python svo_playback.py --input_svo_file "C:\Users\arche\OneDrive\Desktop\cerd_videos\Data\[name].svo2"
```

### Cutting Video
Ensure the file is correct before trimming:
```bash
& "C:\Program Files (x86)\ZED SDK\tools\ZED SVOEditor.exe" -inf "[file_path]"
```
To cut a segment:
```bash
& "C:\Program Files (x86)\ZED SDK\tools\ZED SVOEditor.exe" -cut \
"C:\Users\arche\OneDrive\Desktop\cerd_videos\Raw_SVO\raw_pour_water_2.svo2" \
-s [new start frame number] -e [new end frame number] \
"C:\Users\arche\OneDrive\Desktop\cerd_videos\Data\pour_water_1.svo2"
```

## License
This dataset is released under the **CC BY 4.0 License**. You are free to use, share, and adapt it as long as proper credit is given.

## Citation
If you use this dataset, please cite:
```
[ Citation Placeholder ]
```

