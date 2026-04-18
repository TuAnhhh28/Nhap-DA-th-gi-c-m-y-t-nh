import cv2
import os
from ultralytics import YOLO
from track import FootballTracker  # IMPORTANT INTEGRATION IMPORT

class FootballDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.3):
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.target_classes = [0, 32] # 0: person, 32: sports ball
        
    def detect_frame(self, frame):
        results = self.model.predict(
            frame, 
            conf=self.conf_thresh, 
            classes=self.target_classes, 
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            detections.append([x1, y1, x2, y2, conf, class_id])
            
        return detections
        
    def draw_tracked_frame(self, frame, tracked_players, raw_detections):
        """
        Draws ID-assigned boxes for tracked players and raw boxes for the untracked ball.
        """
        annotated_frame = frame.copy()
        
        # 1. Draw securely tracked players (Blue)
        for tp in tracked_players:
            track_id, x1, y1, x2, y2, conf = tp
            color = (255, 0, 0) # Blue
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {conf:.2f}"
            
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 2. Draw standard untracked objects bypassing the tracker, e.g. the football (Orange)
        balls = [det for det in raw_detections if det[5] == 32] # 32 is the class ID for the ball
        for ball in balls:
            x1, y1, x2, y2, conf, _ = ball
            color = (0, 165, 255) # Orange
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Ball {conf:.2f}"
            
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return annotated_frame


def main():
    input_video_path = "data/input/sample.mp4"
    output_video_path = "data/output/tracked_output.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    
    if not os.path.exists(input_video_path):
        print(f"Error: video {input_video_path} not found.")
        return

    # INITIALIZATION INTEGRATION: Bring online the Tracker and the Detector
    detector = FootballDetector()
    tracker = FootballTracker(fps=30)
    
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # STEP 1: DETECT. Find bodies spatially.
        raw_detections = detector.detect_frame(frame)
        
        # STEP 2: TRACK. Find timeline trace history and inject unique IDs.
        tracked_players = tracker.update(raw_detections)
        
        # STEP 3: RENDER. Draw visual UI overlays explicitly merging both processes.
        annotated_frame = detector.draw_tracked_frame(frame, tracked_players, raw_detections)
        
        writer.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed Frame Progress: {frame_count} / {total_frames}...")
            
    cap.release()
    writer.release()
    print("Tracking output successfully completed!")

if __name__ == "__main__":
    main()
