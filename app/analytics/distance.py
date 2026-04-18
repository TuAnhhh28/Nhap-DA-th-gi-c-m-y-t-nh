import pandas as pd
import numpy as np

class DistanceCalculator:
    """Post-processing analytics module to calculate player distances from flat CSVs."""
    
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath

    def compute_distances(self) -> list:
        """
        Reads CSV tracking traces, sorts temporally, aggregates vector pixel distances, 
        and computes total outputs grouped logically per tracking ID.
        """
        try:
            df = pd.read_csv(self.csv_filepath)
        except Exception as e:
            print(f"Error reading tracking CSV: {e}")
            return []

        if df.empty:
            return []

        # Critical step: Sort temporally so Euclidean differences match frame-to-frame reality
        df = df.sort_values(by=['track_id', 'frame_id'])
        summary = []
        
        # Group by individual players
        for track_id, group in df.groupby('track_id'):
            # Pandas .diff() maps the delta compared against the immediate previous row
            diff_x = group['center_x'].diff()
            diff_y = group['center_y'].diff()
            
            # Pythagorean theorem (a² + b² = c²) applied over vectorized pandas columns 
            # Note: diff() leaves a NaN on the first row (since there's no preceding row), 
            # numpy functions natively ignore it when computing sums.
            pixel_distances = np.sqrt(diff_x**2 + diff_y**2)
            
            total_dist = pixel_distances.sum()
            total_frames = len(group)
            
            summary.append({
                "track_id": track_id,
                "total_distance_pixels": round(total_dist, 2),
                "total_frames_tracked": total_frames
            })
            
        return summary
