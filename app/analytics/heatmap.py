import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class HeatmapGenerator:
    """Generates 2D spatial distribution heatmaps directly from a tracking CSV file."""
    
    def __init__(self, csv_filepath: str, bg_image_width: int = 1920, bg_image_height: int = 1080):
        # We ingest the CSV directly from the post-processing pipeline
        self.csv_filepath = csv_filepath
        self.width = bg_image_width
        self.height = bg_image_height

    def generate(self, output_dir: str):
        """
        Reads CSV tracking traces, calculates Kernel Density Estimates (KDE), 
        and plots smooth heat patterns globally and individually.
        """
        if not os.path.exists(self.csv_filepath):
            print(f"Error: Tracking data not found at {self.csv_filepath}.")
            return

        df = pd.read_csv(self.csv_filepath)
        if df.empty:
            print("No tracking data available for heatmaps.")
            return

        os.makedirs(output_dir, exist_ok=True)
        print("Generating Heatmap Renderings...")

        # 1. Generate Global Output (All tracking points merged into one field spread)
        self._plot_kde(df, "Match Global Heatmap (All Entities)", os.path.join(output_dir, "heatmap_global.png"))

        # 2. Iterate dynamically over individual Players 
        # (Filtering selectively to the Top 5 most active players to avoid rendering 50+ noise artifacts)
        top_players = df['track_id'].value_counts().head(5).index.tolist()
        
        for pid in top_players:
            df_player = df[df['track_id'] == pid]
            # Safety parameter: Require at least 15 spatial frames of data to draw a mathematical geometric curve
            if len(df_player) >= 15:
                # Group by Track ID implicitly
                self._plot_kde(
                    df_player, 
                    f"Individual Heatmap - Player #{int(pid)}", 
                    os.path.join(output_dir, f"heatmap_player_{int(pid)}.png")
                )

    def _plot_kde(self, df: pd.DataFrame, title: str, save_path: str):
        """Internal plotting function utilizing Matplotlib and Seaborn"""
        plt.figure(figsize=(10, 6))
        
        # Enforce graph dimensions to accurately match standard video bounds
        plt.xlim(0, self.width)
        
        # Invert Y-axis natively because OpenCV UI maps (0,0) to Top-Left, 
        # while analytical mathematics maps (0,0) to Bottom-Left.
        plt.ylim(self.height, 0) 
        
        # Seaborn KDE creates polished, beautifully colored gradient clouds from scattered coordinate frames
        sns.kdeplot(
            x=df['center_x'], 
            y=df['center_y'], 
            cmap="magma",       # High intensity heatmap styling
            fill=True, 
            thresh=0.05,        # Ignore deep outlier noise (1-frame glitches)
            levels=100,         # Granular color banding intensity
            alpha=0.85
        )
        
        # Decorate plot UI
        plt.title(title, pad=15)
        plt.xlabel("Pitch X Coordinate (Camera Space)")
        plt.ylabel("Pitch Y Coordinate (Camera Space)")
        
        # Apply Aesthetic 'Grass Turf' background for visualization clarity
        ax = plt.gca()
        ax.set_facecolor('#2e8b57') # Hex code for 'SeaGreen'
        
        # Export logic physically immediately pushes result out to a PNG image buffer
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # File name output tracking
        print(f"  -> Saved Render: {save_path}")
