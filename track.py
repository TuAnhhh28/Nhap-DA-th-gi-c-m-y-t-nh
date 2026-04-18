import supervision as sv
import numpy as np

class FootballTracker:
    """Wraps ByteTrack logic to uniquely identify and track football players across frames."""
    
    def __init__(self, track_activation_threshold=0.25, lost_track_buffer=30, match_thresh=0.8, fps=30):
        # Instantiate the underlying ByteTrack model from the supervision library
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=fps
        )

    def update(self, detections: list) -> list:
        """
        Ingests the raw bounding boxes from YOLO, matches them against historical tracks,
        and generates unique track_id integers.
        
        Input Format:  [ [x1, y1, x2, y2, confidence, class_id], ... ]
        Output Format: [ [track_id, x1, y1, x2, y2, confidence], ... ]
        """
        # Isolation Feature: We strictly track players (COCO class ID 0)
        player_detections = [det for det in detections if int(det[5]) == 0]
        
        if len(player_detections) == 0:
            # Continues stepping the tracking filter to increment memory logic even when no bodies are visible
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
            
        # Supervision Detections strictly require NumPy arrays
        xyxy = np.array([det[0:4] for det in player_detections])
        confidence = np.array([det[4] for det in player_detections])
        class_id = np.array([det[5] for det in player_detections])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Pass mathematical detections directly into the Kalman Filter tracking algorithm
        tracked_sv = self.tracker.update_with_detections(sv_detections)
        
        # Unpack the generated tracked objects back into the exact requested output sequence
        tracked_objects = []
        for i in range(len(tracked_sv.xyxy)):
            x1, y1, x2, y2 = [int(v) for v in tracked_sv.xyxy[i]]
            conf = float(tracked_sv.confidence[i])
            track_id = int(tracked_sv.tracker_id[i])
            
            # Format Request: [track_id, x1, y1, x2, y2, confidence]
            tracked_objects.append([track_id, x1, y1, x2, y2, conf])
            
        return tracked_objects
