from collections import defaultdict

class TrajectoryHistory:
    """Manages the historical coordinates for visualization and analytics."""
    def __init__(self, max_length: int = 30):
        self.max_length = max_length
        # Maps track_id to a list of (x, y) tuples
        self.history = defaultdict(list)
        
    def update(self, tracks: list) -> list:
        """Appends the newest center coordinate to the trace for each tracked object."""
        for track in tracks:
            track_id = track.get('track_id')
            if track_id is None:
                continue
                
            x1, y1, x2, y2 = track['bbox']
            center = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
            
            self.history[track_id].append(center)
            
            # Cap the memory to strictly `max_length` to prevent infinite trails
            if len(self.history[track_id]) > self.max_length:
                self.history[track_id].pop(0)
                
            # Inject 'trail' list into the track dict purely for the Annotator downstream
            track['trail'] = list(self.history[track_id])
            
        return tracks
