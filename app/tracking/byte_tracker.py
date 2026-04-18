import supervision as sv
import numpy as np

class ByteTracker:
    """Wrapper for the ByteTrack multi-object tracking algorithm."""
    
    def __init__(self, config: dict):
        # We increase track_buffer to 60 (2 seconds) to heavily reduce ID flickering 
        # when players rapidly cross each other or get occluded.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.get('track_activation_threshold', 0.25),
            lost_track_buffer=60, 
            minimum_matching_threshold=config.get('match_thresh', 0.8),
            frame_rate=config.get('fps', 30)
        )

    def update(self, detections: list) -> list:
        """
        Updates the tracker securely.
        Input Format:  [ [x1, y1, x2, y2, confidence, class_id], ... ]
        Output Format: [ [x1, y1, x2, y2, confidence, class_id, track_id], ... ] (Objects with IDs)
        """
        
        # Requirement: Track only players (COCO class_id == 0)
        player_detections = [det for det in detections if int(det[5]) == 0]
        other_detections = [det for det in detections if int(det[5]) != 0] # Example: sports ball

        if len(player_detections) == 0:
            # We must step the tracker to keep frame logic synchronized even if no one is visible
            self.tracker.update_with_detections(sv.Detections.empty())
            return other_detections

        # 1. Parse Standard Array Values
        xyxy = np.array([det[0:4] for det in player_detections])
        confidence = np.array([det[4] for det in player_detections])
        class_id = np.array([det[5] for det in player_detections])

        # 2. Package into Supervision object
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # 3. Update Kalman Filter & Hungarian Matching Algorithm
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # 4. Standardize output back to the required Array Format
        tracked_objects = []
        for i in range(len(tracked_detections.xyxy)):
            x1, y1, x2, y2 = [int(v) for v in tracked_detections.xyxy[i]]
            conf = float(tracked_detections.confidence[i])
            cls_id = int(tracked_detections.class_id[i])
            track_id = int(tracked_detections.tracker_id[i])
            
            # Format: [x1, y1, x2, y2, confidence, class_id, track_id]
            tracked_objects.append([x1, y1, x2, y2, conf, cls_id, track_id])

        # Safely recombine the correctly tracked players + untracked objects
        return tracked_objects + other_detections
