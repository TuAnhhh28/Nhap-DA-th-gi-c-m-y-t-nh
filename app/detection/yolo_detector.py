from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_name: str, conf_thresh: float, classes: list):
        self.conf_thresh = conf_thresh
        self.classes = classes
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        print(f"YOLO model loaded successfully. Target classes: {self.classes}")
        
    def detect(self, frame):
        # Run inference using Ultralytics
        results = self.model.predict(
            frame, 
            conf=self.conf_thresh, 
            classes=self.classes,
            verbose=False
        )[0]
        
        # Parse results into a standard dictionary format
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            name = self.model.names[cls]
            
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class_id": cls,
                "class_name": name
            })
            
        return detections
