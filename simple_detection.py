import cv2
from ultralytics import YOLO
import os

def main():
    # === Configuration ===
    input_video_path = "data/input/sample.mp4"
    output_video_path = "data/output/simple_annotated.mp4"
    model_name = "yolov8n.pt"  # Ultralytics will download this automatically
    target_classes = [0, 32]   # COCO dataset IDs -> 0: person, 32: sports ball
    confidence_thresh = 0.3

    # Ensure our output destination folder exists
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    # === 1. Load the YOLOv8 model ===
    print(f"Loading YOLOv8 model '{model_name}'...")
    model = YOLO(model_name)
    print("Model loaded successfully.")

    # === 2. Read the football video using OpenCV ===
    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found!")
        print("Please place a sample '.mp4' file inside 'data/input/' to run this standalone script.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source at {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video. Output will run loop for {total_frames} frames.")
    frame_count = 0

    # === 3. Process video frame-by-frame ===
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run bare YOLO inference on frame targeting our specific classes
        results = model.predict(frame, conf=confidence_thresh, classes=target_classes, verbose=False)[0]

        # === 4. Iterate detections and draw annotations ===
        for box in results.boxes:
            # Extract coordinates and metrics
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            name = model.names[cls_id]

            # Distinguish UI colors based on class
            if cls_id == 0:
                color = (255, 0, 0)   # Blue for person
            elif cls_id == 32:
                color = (0, 165, 255) # Orange for sports ball
            else:
                color = (0, 255, 0)   # Green fallback

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Define label string and calculate its dimensions to draw aesthetic background
            label = f"{name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            
            # Draw label inline text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Write completely annotated frame to the output video sequence
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    # Gracefully garbage collect OS memory limits
    cap.release()
    out.release()
    print(f"Processing complete! Check output at: {output_video_path}")

if __name__ == "__main__":
    main()
