import sys
import os
from pathlib import Path

# Add the project root strictly to Python's system path to allow absolute 'app' imports
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

import argparse
import yaml
from app.core.pipeline import TrackerPipeline
from app.analytics.distance import DistanceCalculator
from app.output.writer_csv import CsvWriter

# Optional analytics if needed
try:
    from app.analytics.heatmap import HeatmapGenerator
except ImportError:
    HeatmapGenerator = None

def load_config(config_path: str) -> dict:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Football Player Tracking System")
    parser.add_argument("--config", type=str, default="app/config/default.yaml", help="Path to configuration yaml")
    parser.add_argument("--heatmaps", action="store_true", help="Generate heatmaps after tracking")
    args = parser.parse_args()

    # Load Configuration
    config = load_config(args.config)
    
    # 1. Run the Core Detection & Tracking Pipeline
    print("=== Phase 1: Tracking Pipeline ===")
    pipeline = TrackerPipeline(config)
    pipeline.run()

    # 2. Post-processing Analytics: Compute distance
    print("\n=== Phase 2: Post-Processing Analytics ===")
    csv_metrics_path = config.get('output', {}).get('csv_metrics_path', 'data/output/tracking_data.csv')
    
    if Path(csv_metrics_path).exists():
        distance_calc = DistanceCalculator(csv_filepath=csv_metrics_path)
        summary_data = distance_calc.compute_distances()
        
        # Export Analytics Summary
        summary_output_path = config.get('output', {}).get('csv_summary_path', 'data/output/distance_summary.csv')
        CsvWriter.save_summary(summary_data, summary_output_path)
        print(f"Distance summary saved to {summary_output_path}")

        # 3. Generate Heatmaps if requested
        if args.heatmaps and HeatmapGenerator:
            print("Generating spatial heatmaps...")
            heatmap_gen = HeatmapGenerator(csv_filepath=csv_metrics_path)
            heatmap_gen.generate(output_dir="data/analysis/")
    else:
        print(f"Warning: Tracking CSV not found at {csv_metrics_path}. Skipping analytics.")

    print("\nSystem pipeline successfully completed!")

if __name__ == "__main__":
    main()
