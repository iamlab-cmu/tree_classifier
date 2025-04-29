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

WANDB_PROJECT_NAME = "contact-sound-time-ablation-test-set-resampling"
NUM_EPOCHS = "50"  # Training epochs for ablation study
VAL_SPLIT = "0.2"  # 20% of training data used for validation


def run_training_experiment(experiment_config, run_name):
    """Run training with specified configuration."""
    command = ["python3", "learning/train.py"]

    for key, value in experiment_config.items():
        command.append(f"{key}={value}")

    print("\n" + "=" * 80)
    print(f"Running training experiment {run_name}:")
    print(" ".join(command))
    print("=" * 80 + "\n")

    print(
        "Debug - Training on:",
        experiment_config["data.train_base_path"],
        experiment_config["data.train_csv_path"],
    )
    print(
        "Debug - Testing on:",
        experiment_config["data.test_base_path"],
        experiment_config["data.test_csv_path"],
    )

    test_path = os.path.join(
        "learning",
        experiment_config["data.test_base_path"],
        experiment_config["data.test_csv_path"],
    )
    print(f"Debug - Full test path: {test_path}")
    print(f"Debug - Test file exists: {os.path.exists(test_path)}")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return False
    else:
        print(f"Training completed successfully")
        return True


def main():
    args = parse_arguments()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    study_dir = (
        args.output_dir if args.output_dir else f"time_ablation_study_{timestamp}"
    )
    os.makedirs(study_dir, exist_ok=True)

    window_configs = []
    for i in range(1, 11):
        window_configs.append(
            {
                "name": f"window_{i}",
                "window_length_seconds": i * 0.1,  # 0.5s, 0.6s, ..., 1.0s
                "description": f"Window duration: {i * 0.1}s",
            }
        )

    if args.window_lengths:
        specified_lengths = [float(w) for w in args.window_lengths.split(",")]
        window_configs = [
            cfg
            for cfg in window_configs
            if cfg["window_length_seconds"] in specified_lengths
        ]

    use_binary = False  # Override to always use multi-class

    base_dir = args.datasets_path or ""

    print(f"Running time window ablation experiments with:")
    print(
        f"- Window configurations: {len(window_configs)} (window_5 through window_10)"
    )
    print(f"- Classification: Multi-class")
    print(f"- Base directory prefix: {base_dir}")
    print(f"- Using full dataset (no subsetting)")

    found_datasets = []
    for config in window_configs:
        config_name = config["name"]

        probe_dir = os.path.join("learning", f"audio_visual_dataset_{config_name}")
        robot_dir = os.path.join("learning", f"audio_visual_dataset_robo_{config_name}")

        probe_csv_direct = os.path.join(probe_dir, "dataset.csv")
        robot_csv_direct = os.path.join(robot_dir, "dataset.csv")

        probe_nested_dir = os.path.join(
            probe_dir, f"audio_visual_dataset_{config_name}"
        )
        robot_nested_dir = os.path.join(
            robot_dir, f"audio_visual_dataset_robo_{config_name}"
        )
        probe_csv_nested = os.path.join(probe_nested_dir, "dataset.csv")
        robot_csv_nested = os.path.join(robot_nested_dir, "dataset.csv")

        probe_csv_path = probe_dir
        robot_csv_path = robot_dir

        probe_csv = os.path.join(probe_csv_path, "dataset.csv")
        robot_csv = os.path.join(robot_csv_path, "dataset.csv")

        if os.path.exists(probe_csv) and os.path.exists(robot_csv):
            if os.path.exists(probe_csv_direct):
                probe_dir_no_prefix = f"audio_visual_dataset_{config_name}"
                probe_csv_no_prefix = "dataset.csv"  # Just the filename
            else:
                probe_dir_no_prefix = os.path.join(
                    f"audio_visual_dataset_{config_name}",
                    f"audio_visual_dataset_{config_name}",
                )
                probe_csv_no_prefix = "dataset.csv"  # Just the filename

            if os.path.exists(robot_csv_direct):
                robot_dir_no_prefix = f"audio_visual_dataset_robo_{config_name}"
                robot_csv_no_prefix = "dataset.csv"  # Just the filename
            else:
                robot_dir_no_prefix = os.path.join(
                    f"audio_visual_dataset_robo_{config_name}",
                    f"audio_visual_dataset_robo_{config_name}",
                )
                robot_csv_no_prefix = "dataset.csv"  # Just the filename

            found_datasets.append(
                {
                    "config": config,
                    "probe_dir": probe_dir,  # Use the existing variable probe_dir
                    "robot_dir": robot_dir,  # Use the existing variable robot_dir
                    "probe_csv": probe_csv,  # Full path to CSV for checking
                    "robot_csv": robot_csv,  # Full path to CSV for checking
                    "probe_dir_no_prefix": probe_dir_no_prefix,  # For passing to train.py
                    "robot_dir_no_prefix": robot_dir_no_prefix,  # For passing to train.py
                    "probe_csv_no_prefix": probe_csv_no_prefix,  # For passing to train.py
                    "robot_csv_no_prefix": robot_csv_no_prefix,  # For passing to train.py
                }
            )
            print(f"Found dataset for {config_name} - {config['description']}")
            print(f"  Probe dataset: {probe_dir}")
            print(f"  Robot dataset: {robot_dir}")
        else:
            print(f"No dataset found for {config_name} - {config['description']}")
            print(f"  Looked for: {probe_csv}")
            print(f"  And: {robot_csv}")

    if not found_datasets:
        print("No datasets found matching the specified window configurations.")
        print("Make sure the datasets exist and contain dataset.csv files")
        return

    print(f"Found {len(found_datasets)} datasets. Starting training experiments...")

    for dataset in found_datasets:
        config = dataset["config"]

        run_name = f"{config['name']}"

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
            "hydra.run.dir": os.path.join(study_dir, run_name),
        }

        test_csv_full_path = os.path.join(
            "learning", dataset["robot_dir_no_prefix"], dataset["robot_csv_no_prefix"]
        )
        print(f"DEBUG - Full test CSV path: {test_csv_full_path}")
        print(f"DEBUG - Test CSV exists: {os.path.exists(test_csv_full_path)}")

        if args.skip_existing and os.path.exists(os.path.join(study_dir, run_name)):
            print(f"Skipping existing run: {run_name}")
            continue

        run_training_experiment(experiment_config, run_name)

        print("Waiting for 5 seconds before starting next experiment...")
        time.sleep(5)

    readme_path = os.path.join(study_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Window Duration Ablation Study (Window 5 through Window 10)\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Experiment Settings\n\n")
        f.write(f"- **WandB Project**: {WANDB_PROJECT_NAME}\n")
        f.write(f"- **Epochs**: {NUM_EPOCHS}\n")
        f.write(
            f"- **Using Full Dataset**: Yes (with training balanced to 700 samples per category, test set to 155)\n"
        )
        f.write(f"- **Validation Split**: {VAL_SPLIT}\n")
        f.write(f"- **Classification**: Multi-class\n")
        f.write(f"- **Cross-validation**: 5-fold\n\n")

        f.write("## Window Configurations\n\n")
        for dataset in found_datasets:
            config = dataset["config"]
            f.write(f"### {config['name']}\n")
            f.write(f"- **Description**: {config['description']}\n")
            f.write(f"- **Window Length**: {config['window_length_seconds']} seconds\n")
            f.write(f"- **Train Dataset**: `{dataset['probe_dir']}`\n")
            f.write(f"- **Test Dataset**: `{dataset['robot_dir']}`\n\n")

    print(f"\nTime window ablation study completed. Results saved to {study_dir}")
    print(f"Check WandB project '{WANDB_PROJECT_NAME}' for results visualization")
    print(
        f"A README.md file with experiment descriptions has been created at: {readme_path}"
    )


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
        help="Directory to save results (default: timestamped directory)",
    )
    parser.add_argument(
        "--window-lengths",
        type=str,
        default=None,
        help="Comma-separated list of window lengths to test (e.g., '0.3,0.4')",
    )
    parser.add_argument(
        "--datasets-path",
        type=str,
        default=None,
        help="Base directory prefix for datasets (optional)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if output directory already exists",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
