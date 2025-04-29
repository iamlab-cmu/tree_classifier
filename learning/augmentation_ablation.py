import os
import sys
import subprocess
import yaml
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import time

# Hardcoded configuration parameters
WANDB_PROJECT_NAME = "contact-sound-ablation-v2"
NUM_EPOCHS = "50"  # Shorter training for ablation study
DATA_SUBSET_SIZE = "0.50"  # Use 50% of data instead of 20% to ensure all classes are represented

def run_training_experiment(experiment_config, run_name):
    """Run training with specified configuration."""
    # Format the command
    command = ["python3", "learning/main_train_2.py"]
    
    # Add all configuration overrides
    for key, value in experiment_config.items():
        command.append(f"{key}={value}")
    
    # Print info
    print("\n" + "="*80)
    print(f"Running training experiment {run_name}:")
    print(" ".join(command))
    print("="*80 + "\n")
    
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
    
    # Set the study directory to the provided output directory
    study_dir = args.output_dir
    os.makedirs(study_dir, exist_ok=True)
    
    # Determine denoising settings to test
    denoising_settings = [args.denoising]
    
    # Make sure we're explicitly using multi-class (not binary)
    use_binary = False  # Force this to false to ensure multi-class classification
    binary_mode_str = "false"  # Always use multi-class
    
    # Define augmentation combinations to test
    augmentation_combinations = [
        # No augmentations (baseline)
        {
            "data.augmentations.image_rotation": "false",
            "data.augmentations.audio_frequency_scaling": "false",
            "data.augmentations.audio_robot_humming": "false"
        },
        # Individual augmentations
        {
            "data.augmentations.image_rotation": "true",
            "data.augmentations.audio_frequency_scaling": "false",
            "data.augmentations.audio_robot_humming": "false"
        },
        {
            "data.augmentations.image_rotation": "false",
            "data.augmentations.audio_frequency_scaling": "true",
            "data.augmentations.audio_robot_humming": "false"
        },
        {
            "data.augmentations.image_rotation": "false",
            "data.augmentations.audio_frequency_scaling": "false",
            "data.augmentations.audio_robot_humming": "true"
        },
        # All augmentations
        {
            "data.augmentations.image_rotation": "true",
            "data.augmentations.audio_frequency_scaling": "true",
            "data.augmentations.audio_robot_humming": "true"
        }
    ]
    
    # Filter augmentation combinations based on start index if provided
    if args.start_index is not None:
        if args.start_index < 0 or args.start_index >= len(augmentation_combinations):
            print(f"Invalid start index: {args.start_index}. Must be between 0 and {len(augmentation_combinations)-1}")
            return
        augmentation_combinations = augmentation_combinations[args.start_index:]
    
    # Print experiment summary
    print(f"Running experiments with:")
    print(f"- Denoising setting: {'enabled' if args.denoising else 'disabled'}")
    print(f"- Binary classification: {use_binary}")
    print(f"- Starting with augmentation combination {args.start_index if args.start_index is not None else 0}")
    print(f"- Running {len(augmentation_combinations)} augmentation combinations")
    
    # Run training with different augmentation settings
    for combo_idx, aug_combo in enumerate(augmentation_combinations):
        current_idx = (args.start_index or 0) + combo_idx
        print(f"Running augmentation combination {current_idx} of {(args.start_index or 0) + len(augmentation_combinations) - 1}")
        
        # Create a descriptive name for this combination
        aug_desc = []
        if aug_combo["data.augmentations.image_rotation"] == "true":
            aug_desc.append("img_rot")
        if aug_combo["data.augmentations.audio_frequency_scaling"] == "true":
            aug_desc.append("freq_scale")
        if aug_combo["data.augmentations.audio_robot_humming"] == "true":
            aug_desc.append("robot_hum")
            
        if not aug_desc:
            aug_suffix = "no_aug"
        else:
            aug_suffix = "_".join(aug_desc)
        
        # Full run name
        binary_prefix = "binary_" if use_binary else ""
        run_name = f"{binary_prefix}{'denoising' if args.denoising else 'no_denoising'}_{aug_suffix}"
        
        # Create full experiment config
        experiment_config = aug_combo.copy()
        experiment_config.update({
            "data.use_subset": "true",
            "data.subset_size": DATA_SUBSET_SIZE,
            "data.use_binary_classification": binary_mode_str,
            "model.num_classes": "4",  # Explicitly set to 4 classes
            "data.val_split": "0.2",   # Use a larger validation split
            "training.epochs": NUM_EPOCHS,
            "wandb.run_name": run_name,
            "wandb.project": WANDB_PROJECT_NAME,
            "hydra.run.dir": os.path.join(study_dir, run_name)
        })
        
        # Skip if this run already has a directory and --skip-existing is set
        if args.skip_existing and os.path.exists(os.path.join(study_dir, run_name)):
            print(f"Skipping existing run: {run_name}")
            continue
        
        # Run the training experiment
        success = run_training_experiment(experiment_config, run_name)
        
        if not success and args.stop_on_failure:
            print("Stopping due to training failure")
            break
            
        # Wait a bit between runs to avoid resource conflicts
        print("Waiting for 5 seconds before starting next experiment...")
        time.sleep(5)
    
    print(f"\nAblation study completed. Results saved to {study_dir}")
    print(f"Check WandB project '{WANDB_PROJECT_NAME}' for results visualization")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Augmentation ablation study")
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Directory to save results (must be specified)"
    )
    parser.add_argument(
        "--denoising",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="Set to true or false to specify which dataset to use (default: false - no denoising)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Index of the augmentation combination to start from (0-4, default: 0)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have output directories"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the experiment chain if a training run fails"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 