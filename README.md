# ⚽ Football Player Tracking and Spatial Analysis System

## 1. Project Overview
This repository contains an end-to-end computer vision pipeline designed for the automated detection, tracking, and spatial analysis of football players from broadcast footage. Developed as an academic project, the system extracts high-level tactical data from raw video feeds, rendering annotated outputs and analytical heatmaps to quantify player movement and density distribution.

## 2. Methodology
The system architecture leverages a dual-stage neural network approach:
- **Spatial Detection (YOLOv8):** We employ the Ultralytics YOLOv8 network pretrained on the COCO dataset to isolate specific target classes (`Person: 0`, `Sports Ball: 32`). YOLO provides robust frame-by-frame spatial bounding box coordinates across dynamic camera pans.
- **Temporal Tracking (ByteTrack):** To establish persistent identities globally without computationally expensive Re-ID mechanisms, we integrate ByteTrack. By utilizing a Kalman Filter alongside a Hungarian matching algorithm natively factoring in low-confidence unoccluded detections, ByteTrack drastically mitigates ID flickering when athletes cross paths or become temporarily occluded.
- **Data Densification (Seaborn KDE):** Extracted `[X, Y]` coordinates are bridged into Kernel Density Estimates (KDE) to generate topographical heatmaps mapping spatial density over the temporal duration of the match.

## 3. Pipeline Architecture
The system is built sequentially to enforce a strictly modular, single-directional data flow:
1. **Video Ingestion:** OpenCV decodes the video into isolated RGB frames.
2. **Inference:** YOLO extracts a raw array: `[x1, y1, x2, y2, confidence, class_id]`.
3. **Identity Mathematics:** ByteTrack evaluates the timeline vectors and injects a stable `track_id`.
4. **Data Translation:** Formatted coordinates calculate Cartesian centers (`center_x`, `center_y`) and are logged persistently to Pandas memory.
5. **Visualization Rendering:** Track bounds, IDs, and confidence overlays are drawn atop the frame.
6. **Data Output:** 
   - A fully annotated `.avi` stream is saved contextually.
   - Matrices are dumped automatically into strict, duplicate-free CSV formatting (`tracking_data.csv`).
   - Distance aggregations and spatial rendering grids are plotted.

## 4. Folder Structure
The codebase strictly adheres to modular Python conventions:
```text
Football Tracking Project/
│
├── .gitignore
├── input.mp4                  <-- Raw video feed
├── output.avi                 <-- Annotated processing output
├── yolov8n.pt                 <-- Local neural weights
├── requirements.txt           <-- Dependency list
│
├── app/                       <-- Modular Core Components
│   ├── config/
│   │   └── default.yaml       <-- Pipeline parameters
│   ├── core/
│   │   ├── pipeline.py        <-- Central data flow loop
│   │   └── video_io.py        <-- OpenCV multiplexers
│   ├── detection/
│   │   └── yolo_detector.py   
│   ├── tracking/
│   │   └── byte_tracker.py    
│   ├── analytics/
│   │   ├── distance.py        <-- Euclidean calculations
│   │   ├── heatmap.py         <-- Matplotlib renderers
│   │   └── trajectory.py
│   ├── visualization/
│   │   └── annotator.py
│   └── output/
│       └── writer_csv.py      
│
└── scripts/
    └── run.py                 <-- Central Execution Script
```

## 5. Installation
Ensure that `Python 3.8+` is installed on your local environment.
```bash
# 1. Clone the repository
git clone https://github.com/your-username/football-tracking.git
cd football-tracking

# 2. Create a virtual environment (Recommended)
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On MacOS/Linux:
source .venv/bin/activate

# 3. Install required geometric and vision dependencies
pip install -r requirements.txt
```
*(Dependencies securely rely upon `opencv-python`, `ultralytics`, `supervision`, `pandas`, `seaborn`, and `matplotlib`.)*

## 6. Execution Instructions
To execute the pipeline smoothly, trigger the modular entry point from the root directory:

**Standard Tracking Execution:**
```bash
python scripts/run.py
```

**Generate Topographical Heatmaps Post-Tracking:**
```bash
python scripts/run.py --heatmaps
```

**Override Configuration via YAML:**
```bash
python scripts/run.py --config "app/config/custom_setup.yaml"
```

## 7. Sample Outputs
Upon conclusion, the model autonomously exports three forms of data into the `/data/output/` and `/data/analysis/` channels:
- **`annotated_sample.avi`**: A fully rendered broadcast copy displaying locked mathematical bounding bounds and identifier tags natively matching athletes frame-by-frame.
- **`trajectory_data.csv`**: Statistical traces detailing topological metrics: `[frame_id, track_id, class_name, x1, y1, x2, y2, center_x, center_y, confidence]`.
- **`heatmap_global.png`**: High-contrast graphic heatmaps showcasing absolute pitch distribution mapping.

## 8. Limitations & Future Work
- **Perspective Distortion (Pixel vs. Meter Mapping):** Currently, Euclidean tracking distances are computed mathematically via pure flat 2D camera pixel metrics. A pixel traveling horizontally in the background covers significantly more true physical ground than a foreground pixel. Future implementation demands a robust mathematically accurate Homography Matrix (Perspective Transform) to flatten the tactical cam down to true $m^2$ metrics.
- **Camera Panning Dynamics:** The baseline ByteTrack algorithm operates independently of camera motion mapping. Extreme rapid broadcast camera panning natively inflates absolute Euclidean pixel distance because the background structure actively shifts independently of the player's physical sprint layout.
- **Extreme Occlusion Gaps:** Despite high `track_buffer` capacities, tight goal-box scrambles where bounding boxes disappear completely for numerous seconds may accidentally wipe entity matrix arrays, generating arbitrary new IDs upon reemergence.
