import supervision as sv
import numpy as np

class ByteTracker:
    """Wrapper for the ByteTrack multi-object tracking algorithm using Supervision."""
    
    def __init__(self, config: dict):
        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.get('track_activation_threshold', 0.25),
            lost_track_buffer=config.get('track_buffer', 30),
            minimum_matching_threshold=config.get('match_thresh', 0.8),
            frame_rate=config.get('fps', 30)
        )
        print(f"Initialized ByteTrack with params: {self.config}")

    def update(self, detections: list) -> list:
        """
        Filters detections to 'track only players', converts to tracker format,
        updates the tracker, and returns current tracked players alongside untracked balls.
        """
        # Requirement: Track only players (class_id == 0)
        player_detections = [det for det in detections if det['class_id'] == 0]
        other_detections = [det for det in detections if det['class_id'] != 0] # Keeps balls untracked

        if len(player_detections) == 0:
            # We still step the tracker by passing empty detections to increment memory
            self.tracker.update_with_detections(sv.Detections.empty())
            return other_detections

        # 1. Convert player detections into numpy arrays
        xyxy = np.array([det['bbox'] for det in player_detections])
        confidence = np.array([det['confidence'] for det in player_detections])
        class_id = np.array([det['class_id'] for det in player_detections])
        class_names = np.array([det['class_name'] for det in player_detections])

        # 2. Package into a Supervision Detections object for ByteTrack
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={"class_name": class_names}
        )

        # 3. Update the tracker (handles ID assignment and missing frames)
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # 4. Parse the outputs back into our modular pipeline format
        tracked_objects = []
        for i in range(len(tracked_detections.xyxy)):
            tracked_objects.append({
                "bbox": [int(v) for v in tracked_detections.xyxy[i]],
                "confidence": float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0,
                "class_id": int(tracked_detections.class_id[i]),
                "class_name": str(tracked_detections.data["class_name"][i]) if "class_name" in tracked_detections.data else "person",
                "track_id": int(tracked_detections.tracker_id[i]) 
            })

        # Re-combine tracked players and untracked (raw) other objects like the ball
        return tracked_objects + other_detections
