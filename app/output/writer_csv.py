import pandas as pd
import os

class CsvWriter:
    """Handles logging of tracking metrics and exporting them to a CSV file."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.data_store = []
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    def log_frame_data(self, frame_idx: int, tracks: list):
        """Calculates center points mapping and caches the frame data metrics inside memory."""
        for track in tracks:
            track_id = track.get('track_id')
            if track_id is None:
                continue
                
            x1, y1, x2, y2 = track['bbox']
            center_x = int((x1 + x2) / 2.0)
            center_y = int((y1 + y2) / 2.0)
            
            self.data_store.append({
                "frame_id": frame_idx,
                "track_id": track_id,
                "class_name": track['class_name'],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "confidence": round(track['confidence'], 4)
            })
            
    def save(self):
        """Transforms nested memory dictionaries into a Pandas DataFrame and saves to CSV."""
        if not self.data_store:
            print("No tracking data was found to save. (Video empty or no detections)")
            return
            
        df = pd.DataFrame(self.data_store)
        
        # Explicit Safety Requirement: Avoid Duplicate Rows
        df = df.drop_duplicates(subset=['frame_id', 'track_id'], keep='first')
        
        # Sort chronologically by timeline then entity ID securely
        df = df.sort_values(by=['frame_id', 'track_id'])
        
        df.to_csv(self.output_path, index=False)
        print(f"Tracking data metrics exported successfully to {self.output_path}")

    @staticmethod
    def save_summary(summary_data: list, output_path: str):
        """Static method saving aggregated analytics dictionaries flatly without memory footprints."""
        if not summary_data:
            print("No summary data to save.")
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Summary metrics exported successfully to {output_path}")
