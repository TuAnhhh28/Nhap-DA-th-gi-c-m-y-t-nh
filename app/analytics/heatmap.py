import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class HeatmapGenerator:
    """Generates 2D spatial distribution heatmaps from video tracking coordinates."""
    
    def __init__(self, bg_image_width: int, bg_image_height: int):
        self.width = bg_image_width
        self.height = bg_image_height

    def generate(self, tracking_data: list, output_dir: str, player_ids: list = None):
        """
        Calculates Kernel Density Estimate (KDE) and plots smooth heat patterns.
        
        Args:
            tracking_data: Array of dictionaries (straight from Memory CsvWriter)
            output_dir: Parent folder to save output images
            player_ids: Array of integers specifying individual tracks to map alone
        """
        if not tracking_data:
            print("No tracking data available for heatmaps.")
            return

        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(tracking_data)

        # 1. Central Global Heatmap (All players moving across the whole session)
        self._plot_kde(df, "Global Player Heatmap", os.path.join(output_dir, "heatmap_global.png"))

        # 2. Iterative Individual Heatmaps
        if player_ids:
            for pid in player_ids:
                df_player = df[df['track_id'] == pid]
                if not df_player.empty:
                    self._plot_kde(
                        df_player, 
                        f"Heatmap - Track #{pid}", 
                        os.path.join(output_dir, f"heatmap_player_{pid}.png")
                    )

    def _plot_kde(self, df: pd.DataFrame, title: str, save_path: str):
        plt.figure(figsize=(10, 6))
        
        # Enforce graph dimensions to accurately match the 16:9 pixel scale of the video
        plt.xlim(0, self.width)
        # We invert the Y axis because OpenCV treats (0,0) as TOP-left, 
        # but traditional matplotlib charts treat (0,0) as BOTTOM-left
        plt.ylim(self.height, 0) 
        
        # Seaborn KDE creates highly polished, smooth gradient clouds from scattered points
        sns.kdeplot(
            x=df['center_x'], 
            y=df['center_y'], 
            cmap="magma",       # High contrast heatmap styling
            fill=True, 
            thresh=0.05,        # Ignore deep outlier noise
            levels=100,         # Granular color bands
            alpha=0.8
        )
        
        plt.title(title)
        plt.xlabel("X Coordinate (Camera Pixels)")
        plt.ylabel("Y Coordinate (Camera Pixels)")
        
        # Aesthetic generic grass background
        ax = plt.gca()
        ax.set_facecolor('#2e8b57') # Hex for standard 'SeaGreen'
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Heatmap saved: {save_path}")
