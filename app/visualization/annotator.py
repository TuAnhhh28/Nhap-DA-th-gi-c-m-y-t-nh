import cv2
import numpy as np

class Annotator:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        # Colors: BGR format
        self.colors = {
            0: (255, 0, 0),     # Person -> Blue
            32: (0, 165, 255)   # Sports ball -> Orange
        }
        
    def draw(self, frame, tracks):
        annotated_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            cls = track['class_id']
            conf = track['confidence']
            name = track['class_name']
            
            track_id = track.get('track_id', None)
            color = self.colors.get(cls, (0, 255, 0)) # Default green
            
            # 1. Trail Logic
            trail = track.get('trail', [])
            if len(trail) > 1:
                # Must format coordinate memory strictly into N,1,2 geometry structure for cv2 paths
                pts = np.array(trail, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=self.thickness)
            
            # 2. Box Logic
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # 3. Text Labeling Logic
            dist = track.get('distance_pixels', 0)
            if track_id is not None:
                # Provide full analytic readouts inline
                label = f"#{track_id} {name} (Dist: {int(dist)}px)"
            else:
                label = f"{name} {conf:.2f}"
                
            (text_width, text_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.thickness)
            
        return annotated_frame
