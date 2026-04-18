from app.core.video_io import VideoHandler
from app.detection.yolo_detector import YoloDetector
from app.tracking.byte_tracker import ByteTracker
from app.visualization.annotator import Annotator
from app.output.writer_csv import CsvWriter
import cv2 

class TrackerPipeline:
    def __init__(self, config: dict):
        self.config = config
        
        self.video_handler = VideoHandler(
            config['video']['input_path'], 
            config['video']['output_path']
        )
        self.detector = YoloDetector(
            config['detection']['model_name'],
            config['detection']['confidence_threshold'],
            config['detection']['classes']
        )
        self.tracker = ByteTracker(config['tracking'])
        self.annotator = Annotator()
        
        csv_path = config.get('output', {}).get('csv_metrics_path', 'data/output/tracking_data.csv')
        self.csv_writer = CsvWriter(csv_path)
        
        # === OPTIONAL MODULE INITIALIZATION ===
        # We hook into a specific boolean parameter logic to prevent pipeline bloat
        self.enable_emotion = config.get('optional_features', {}).get('enable_emotion_detection', False)
        
        if self.enable_emotion:
            from app.optional.emotion.face_detector import FaceDetector
            from app.optional.emotion.emotion_classifier import EmotionClassifier
            
            # Minimum face size blocks artifacting if camera focuses on distant spectators
            self.face_detector = FaceDetector(min_size=(40, 40))
            self.emotion_classifier = EmotionClassifier()
            print("Analytics Module [Face/Emotion] is ENABLED.")
        
    def run(self):
        print(f"Starting pipeline processing. Output will loop over {self.video_handler.total_frames} frames...")
        frame_idx = 0
        
        for frame in self.video_handler.read_frames():
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            self.csv_writer.log_frame_data(frame_idx, tracks)
            
            annotated_frame = self.annotator.draw(frame, tracks)
            
            # === OPTIONAL ANALYTICS RENDER BLOCK ===
            if self.enable_emotion:
                # 1. Detect faces independently bypassing tracked bodies (uncoupled process)
                face_boxes = self.face_detector.detect(frame)
                
                # 2. Extract emotion metrics mapping out the generated crop regions
                emotion_results = self.emotion_classifier.analyze(frame, face_boxes)
                
                # 3. Draw directly over frame via custom pink CV2 layout elements
                for result in emotion_results:
                    ex1, ey1, ex2, ey2 = result["face_bbox"]
                    emotion = result["emotion"]
                    cv2.rectangle(annotated_frame, (ex1, ey1), (ex2, ey2), (255, 0, 255), 2)  # Pink Boxes
                    cv2.putText(annotated_frame, f"{emotion}", (ex1, ey1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            # =========================================
            
            self.video_handler.write_frame(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{self.video_handler.total_frames} frames...")
                
        self.video_handler.release()
        self.csv_writer.save()
        print(f"Processing complete! Check output video at: {self.config['video']['output_path']}")
