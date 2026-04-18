import pandas as pd
import numpy as np
import os

def compute_distances(csv_file_path):
    """
    Reads the tracking CSV, calculates Euclidean pixel distances frame-by-frame 
    per player, and returns the aggregated distances in pixels.
    """
    df = pd.read_csv(csv_file_path)
    
    # Requirement: Ensure data is sorted securely so temporal differences correlate correctly
    df = df.sort_values(by=['track_id', 'frame_id'])
    
    results = {}
    
    # Group iteratively by unique player track_ids
    for track_id, group in df.groupby('track_id'):
        
        # Calculate mathematical X and Y pixel deltas between consecutive frames
        dx = group['center_x'].diff()
        dy = group['center_y'].diff()
        
        # Pythagorean theorem (a² + b² = c²). diff() places NaNs on the 0th frame natively.
        pixel_distance_deltas = np.sqrt(dx**2 + dy**2)
        
        # Aggregate delta distance; pandas natively drops the initial NaN during sum()
        total_pixels_traveled = pixel_distance_deltas.sum()
        
        results[track_id] = round(total_pixels_traveled, 2)
        
    return results

if __name__ == "__main__":
    # Standard test execution wrapper
    file_path = "data/output/tracking_data.csv"
    
    if not os.path.exists(file_path):
        print(f"Could not locate {file_path}. Please run the main tracking pipeline first!")
    else:
        distances = compute_distances(file_path)
        print("Total distances traveled (Pixels):")
        print("-" * 35)
        for track_id, dist in distances.items():
            print(f"Player #{track_id}:\t{dist} px")
