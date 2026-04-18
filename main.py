import cv2
import os

# Import modular pipeline scripts specifically matching your defined folder structure.
# Modify the classes if your detect.py / track.py expose standard functions instead!
from app.detection.detect import FootballDetector
from app.tracking.track import FootballTracker

def draw_annotations(frame, tracked_objects):
    """
    Renders cleanly formatted bounding boxes and tracking IDs over the OpenCV processed frame.
    Assumed data mapping: [track_id, x1, y1, x2, y2, confidence]
    """
    annotated = frame.copy()
    
    for obj in tracked_objects:
        # Unpack the specific payload formatting
        track_id, x1, y1, x2, y2, conf = obj
        
        # Set explicitly to Blue for standard Player Tracking
        color = (255, 0, 0)
        
        # Draw physical boundaries
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Dynamically draw aesthetic UI Label Background to make text readable
        label = f"ID: {track_id} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # UI Base Rectangle
        cv2.rectangle(annotated, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        # UI Foreground String Data
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated


def main():
    # === 1. Structural Configuration & Hard Checks ===
    input_video = "data/input/sample.mp4"
    output_video = "data/output/final_tracked.mp4"
    
    # Ensure nested OS directory hierarchies exist before attempting deployment
    os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
    if not os.path.exists(input_video):
        print(f"Error: Target video missing at {input_video}")
        return

    # === 2. Module Initialization ===
    detector = FootballDetector() 
    tracker = FootballTracker()

    # === 3. OpenCV Hardware Decoding Setup ===
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: OpenCV failed to open video source. File may be corrupted.")
        return
        
    # Extract fundamental parameters for perfect identical recreation
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 'mp4v' operates cleanly and natively across MacOS/Windows/Linux sequentially
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Started Processing: {total_frames} total frames deploying to {output_video}")

    # === 4. Explicit Processing Loop ===
    frame_count = 0
    while True:
        # Load temporal frame sequentially
        ret, frame = cap.read()
        if not ret:
            break
            
        # -------------------------------------------------------------
        # STEP A: DETECT. Find objects spatially.
        detections = detector.detect(frame)
        
        # STEP B: TRACK. Map timelines and inject persistent trace IDs.
        tracked_objects = tracker.update(detections)
        # -------------------------------------------------------------
        
        # RENDER: Paint generated analytics atop of the pixels natively
        annotated_frame = draw_annotations(frame, tracked_objects)
        
        # Export logic physically immediately pushes result out to video
        writer.write(annotated_frame)
        
        # Feedback logic purely for developer CLI tracking
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed: {frame_count} / {total_frames} frames...")

    # === 5. Graceful Resource Cleanup ===
    cap.release()
    writer.release()
    print("Football tracking pipeline successfully completed!")

if __name__ == "__main__":
    main()