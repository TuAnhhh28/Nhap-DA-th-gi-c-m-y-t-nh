import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_player_heatmaps(csv_file_path, output_dir, max_players=5):
    """
    Reads tracking CSV specifically designed in formatting constraints
    and uses matplotlib explicitly to generate spatial 2D histogram heatmaps 
    targeting discreet players.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: Could not locate CSV trace array at {csv_file_path}. Run tracking pipeline first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file_path)

    # Assumed standard 1080p geometry for tactical scaling boundaries.
    # Update these if your input video uses different resolution limits!
    frame_width = 1920
    frame_height = 1080

    # We filter explicitly by the most frequently tracked IDs to grab the main active players
    active_player_ids = df['track_id'].value_counts().head(max_players).index.tolist()

    print(f"Generating isolated spatial heatmaps for the top {len(active_player_ids)} tracked players...")

    for track_id in active_player_ids:
        player_data = df[df['track_id'] == track_id]
        
        # Ensure sufficient tracking coordinate clusters to build a mathematical field
        if len(player_data) < 10:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create an aesthetically pleasing tactical green background
        ax.set_facecolor('#2e8b57') # 'SeaGreen' simulates pitch turf

        # Mathematically lock spatial boundaries to maintain strict video contextual representations
        ax.set_xlim(0, frame_width)
        # We invert the Y coordinate strictly because OpenCV UI space maps (0,0) at Top-Left, 
        # while matplotlib maps (0,0) natively to Bottom-Left.
        ax.set_ylim(frame_height, 0) 

        x = player_data['center_x']
        y = player_data['center_y']

        # Hexbin inherently processes geometric scatter maps into high density clustered 'Heat Map' grids.
        # mincnt=1 ensures we only draw colors where a player ACTUALLY registered data.
        hb = ax.hexbin(x, y, gridsize=35, cmap='magma', mincnt=1, alpha=0.85)

        # Draw a legend mapping referencing the color intensity logic
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Player Spatial Frame Residency (Density)')
        
        ax.set_title(f"Movement Heatmap - Track ID #{track_id}")
        ax.set_xlabel("Field X Coordinate (Pixels)")
        ax.set_ylabel("Field Y Coordinate (Pixels)")
        
        # Render mathematical plot output cleanly into an image buffer
        output_path = os.path.join(output_dir, f"heatmap_player_{track_id}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"-> Exported Heatmap Graphic: {output_path}")

if __name__ == "__main__":
    # Setup Pipeline Paths natively assuming generic schema
    CSV_INPUT = "data/output/tracking_data.csv"
    OUTPUT_DIR = "data/analysis/"
    
    generate_player_heatmaps(CSV_INPUT, OUTPUT_DIR)
