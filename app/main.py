import argparse
import yaml
from pathlib import Path
from app.core.pipeline import TrackerPipeline
from app.analytics.distance import DistanceCalculator
from app.output.writer_csv import CsvWriter

def load_config(config_path: str) -> dict:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Football Player Tracking System")
    parser.add_argument("--config", type=str, default="app/config/default.yaml", help="Path to config")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. Run the existing Tracker Pipeline
    pipeline = TrackerPipeline(config)
    pipeline.run()

    # 2. Post-processing Analytics: Compute distance from the exported tracking CSV
    print("Beginning Post-Processing Distance Calculation Analytics...")
    csv_metrics_path = config.get('output', {}).get('csv_metrics_path', 'data/output/tracking_data.csv')
    
    distance_calc = DistanceCalculator(csv_filepath=csv_metrics_path)
    summary_data = distance_calc.compute_distances()
    
    # 3. Export Analytics Summary
    summary_output_path = config.get('output', {}).get('csv_summary_path', 'data/output/distance_summary.csv')
    CsvWriter.save_summary(summary_data, summary_output_path)

if __name__ == "__main__":
    main()
