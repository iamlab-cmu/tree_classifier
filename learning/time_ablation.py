import os
import sys
import subprocess
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import time

# Hardcoded configuration parameters
WANDB_PROJECT_NAME = "contact-sound-time-ablation-v7"
NUM_EPOCHS = "50"  # Training epochs for ablation study
VAL_SPLIT = "0.2"  # 20% of training data used for validation

def run_training_experiment(experiment_config, run_name):
    """Run training with specified configuration."""
    # Format the command
    command = ["python3", "learning/train.py"]
    
    # Add all configuration overrides
    for key, value in experiment_config.items():
        command.append(f"{key}={value}")
    
    # Print info
    print("\n" + "="*80)
    print(f"Running training experiment {run_name}:")
    print(" ".join(command))
    print("="*80 + "\n")
    
    # Before running the command
    print("Debug - Training on:", experiment_config["data.train_base_path"], experiment_config["data.train_csv_path"])
    print("Debug - Testing on:", experiment_config["data.test_base_path"], experiment_config["data.test_csv_path"])
    
    # Create absolute paths for checking if the test file exists
    test_path = os.path.join(
        "learning", 
        experiment_config["data.test_base_path"], 
        experiment_config["data.test_csv_path"]
    )
    print(f"Debug - Full test path: {test_path}")
    print(f"Debug - Test file exists: {os.path.exists(test_path)}")
    
    # Run the command
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return False
    else:
        print(f"Training completed successfully")
        return True

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create timestamp for run naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    study_dir = args.output_dir if args.output_dir else f"time_ablation_study_{timestamp}"
    os.makedirs(study_dir, exist_ok=True)
    
    # Define window durations to test (from 0.1 to 1.0 seconds in 0.1 second intervals)
    window_configs = []
    for i in range(1, 11):  # Starting from 1 (0.1s) to 10 (1.0s)
        min_duration = round(i * 0.1, 1)  # 0.1, 0.2, ..., 1.0
        stride = round(min_duration / 3, 3)  # 1/3 of the min duration
        
        window_configs.append({
            "name": f"window_{int(min_duration*10)}",
            "window_length_seconds": min_duration,
            "window_stride_seconds": stride,
            "description": f"Window duration: {min_duration}s, stride: {stride}s (1/3 of window)"
        })
    
    # If specific window lengths are specified, filter the configs
    if args.window_lengths:
        specified_lengths = [float(w) for w in args.window_lengths.split(',')]
        window_configs = [cfg for cfg in window_configs 
                         if cfg["window_length_seconds"] in specified_lengths]
    
    # Make sure we're using multi-class (not binary)
    use_binary = False  # Override to always use multi-class
    
    # Look for existing datasets in the specified path
    # Default to current directory if not specified
    base_dir = args.datasets_path or ""
    
    # Print experiment summary
    print(f"Running time window ablation experiments with:")
    print(f"- Window configurations: {len(window_configs)} (0.1s to 1.0s)")
    print(f"- Classification: Multi-class")
    print(f"- Base directory prefix: {base_dir}")
    print(f"- Using full dataset (no subsetting)")
    
    found_datasets = []
    # Check which window duration datasets exist
    for config in window_configs:
        config_name = config["name"]
        
        # Look for datasets under the learning directory
        probe_dir = os.path.join("learning", f"audio_visual_dataset_{config_name}")
        robot_dir = os.path.join("learning", f"audio_visual_dataset_robo_{config_name}")
        
        # Handle double nesting - check if dataset.csv exists directly or in the nested folder
        probe_csv_direct = os.path.join(probe_dir, "dataset.csv")
        robot_csv_direct = os.path.join(robot_dir, "dataset.csv")
        
        # Check for double nesting
        probe_nested_dir = os.path.join(probe_dir, f"audio_visual_dataset_{config_name}")
        robot_nested_dir = os.path.join(robot_dir, f"audio_visual_dataset_robo_{config_name}")
        probe_csv_nested = os.path.join(probe_nested_dir, "dataset.csv")
        robot_csv_nested = os.path.join(robot_nested_dir, "dataset.csv")
        
        probe_csv_path = "learning/audio_visual_dataset_window_1"
        robot_csv_path = "learning/audio_visual_dataset_robo_window_1"
        
        probe_csv = os.path.join(probe_csv_path, "dataset.csv")
        robot_csv = os.path.join(robot_csv_path, "dataset.csv")
        
        if os.path.exists(probe_csv) and os.path.exists(robot_csv):
            # For train.py, we need to adjust the paths differently
            # The CSV path should be just "dataset.csv" and base_path should be the full path to the directory
            
            if os.path.exists(probe_csv_direct):
                # Standard structure
                probe_dir_no_prefix = f"audio_visual_dataset_{config_name}"
                probe_csv_no_prefix = "dataset.csv"  # Just the filename
            else:
                # Double nested structure
                probe_dir_no_prefix = os.path.join(f"audio_visual_dataset_{config_name}", 
                                                 f"audio_visual_dataset_{config_name}")
                probe_csv_no_prefix = "dataset.csv"  # Just the filename
            
            if os.path.exists(robot_csv_direct):
                # Standard structure
                robot_dir_no_prefix = f"audio_visual_dataset_robo_{config_name}"
                robot_csv_no_prefix = "dataset.csv"  # Just the filename
            else:
                # Double nested structure
                robot_dir_no_prefix = os.path.join(f"audio_visual_dataset_robo_{config_name}", 
                                                 f"audio_visual_dataset_robo_{config_name}")
                robot_csv_no_prefix = "dataset.csv"  # Just the filename
            
            found_datasets.append({
                "config": config,
                "probe_dir": probe_dir,  # Use the existing variable probe_dir
                "robot_dir": robot_dir,  # Use the existing variable robot_dir
                "probe_csv": probe_csv,  # Full path to CSV for checking
                "robot_csv": robot_csv,  # Full path to CSV for checking
                "probe_dir_no_prefix": probe_dir_no_prefix,  # For passing to train.py
                "robot_dir_no_prefix": robot_dir_no_prefix,  # For passing to train.py
                "probe_csv_no_prefix": probe_csv_no_prefix,  # For passing to train.py
                "robot_csv_no_prefix": robot_csv_no_prefix   # For passing to train.py
            })
            print(f"Found dataset for {config_name} - {config['description']}")
            print(f"  Probe dataset: {probe_dir}")
            print(f"  Robot dataset: {robot_dir}")
        else:
            print(f"No dataset found for {config_name} - {config['description']}")
            print(f"  Looked for: {probe_csv_direct} or {probe_csv_nested}")
            print(f"  And: {robot_csv_direct} or {robot_csv_nested}")
    
    if not found_datasets:
        print("No datasets found matching the specified window configurations.")
        print("Make sure the datasets exist and contain dataset.csv files")
        return
    
    print(f"Found {len(found_datasets)} datasets. Starting training experiments...")
    
    # Run training for each found dataset
    for dataset in found_datasets:
        config = dataset["config"]
        
        # Use paths without "learning/" prefix for train.py parameters
        run_name = f"{config['name']}"
        
        # Create simplified experiment config with only dataset parameters
        experiment_config = {
            "data.train_csv_path": dataset["probe_csv_no_prefix"],
            "data.test_csv_path": dataset["robot_csv_no_prefix"],
            "data.train_base_path": dataset["probe_dir_no_prefix"],
            "data.test_base_path": dataset["robot_dir_no_prefix"],
            "data.use_robot_data": "true",
            "data.use_subset": "false",  # Explicitly set to false to use full dataset
            "data.category_balancing.enabled": "true",  # Enable category balancing
            "data.category_balancing.min_samples_per_category": "700",  # Use 700 samples per category from default.yaml
            "data.category_balancing.balance_test_set": "true",  # Ensure test set is also balanced
            "data.category_balancing.test_min_samples_per_category": "155",  # Test set should use 155 samples per category
            "data.val_split": VAL_SPLIT,
            "training.epochs": NUM_EPOCHS,
            "wandb.run_name": run_name,
            "wandb.project": WANDB_PROJECT_NAME,
            "hydra.run.dir": os.path.join(study_dir, run_name)
        }
        
        # Debug: Confirm if test CSV will be found
        test_csv_full_path = os.path.join("learning", dataset["robot_dir_no_prefix"], dataset["robot_csv_no_prefix"])
        print(f"DEBUG - Full test CSV path: {test_csv_full_path}")
        print(f"DEBUG - Test CSV exists: {os.path.exists(test_csv_full_path)}")
        
        # Skip if this run already has a directory and --skip-existing is set
        if args.skip_existing and os.path.exists(os.path.join(study_dir, run_name)):
            print(f"Skipping existing run: {run_name}")
            continue
            
        # Run the training experiment
        run_training_experiment(experiment_config, run_name)
        
        # Wait a bit between runs to avoid resource conflicts
        print("Waiting for 5 seconds before starting next experiment...")
        time.sleep(5)
    
    # Create a README.md with experiment descriptions
    readme_path = os.path.join(study_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Window Duration Ablation Study (0.1s to 1.0s)\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Experiment Settings\n\n")
        f.write(f"- **WandB Project**: {WANDB_PROJECT_NAME}\n")
        f.write(f"- **Epochs**: {NUM_EPOCHS}\n")
        f.write(f"- **Using Full Dataset**: Yes (with training balanced to 700 samples per category, test set to 155)\n")
        f.write(f"- **Validation Split**: {VAL_SPLIT}\n")
        f.write(f"- **Classification**: Multi-class\n\n")
        
        f.write("## Window Configurations\n\n")
        for dataset in found_datasets:
            config = dataset["config"]
            f.write(f"### {config['name']}\n")
            f.write(f"- **Description**: {config['description']}\n")
            f.write(f"- **Window Length**: {config['window_length_seconds']} seconds\n")
            f.write(f"- **Window Stride**: {config['window_stride_seconds']} seconds\n")
            f.write(f"- **Train Dataset**: `{dataset['probe_dir']}`\n")
            f.write(f"- **Test Dataset**: `{dataset['robot_dir']}`\n\n")
    
    print(f"\nTime window ablation study completed. Results saved to {study_dir}")
    print(f"Check WandB project '{WANDB_PROJECT_NAME}' for results visualization")
    print(f"A README.md file with experiment descriptions has been created at: {readme_path}")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Time window ablation study")
    parser.add_argument(
        "--output-dir", 
        default=None, 
        help="Directory to save results (default: timestamped directory)"
    )
    parser.add_argument(
        "--window-lengths",
        type=str,
        default=None,
        help="Comma-separated list of window lengths to test (e.g., '0.5,0.7,1.0')"
    )
    parser.add_argument(
        "--datasets-path",
        type=str,
        default=None,
        help="Base directory prefix for datasets (optional)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if output directory already exists"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
