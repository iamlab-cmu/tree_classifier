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
WANDB_PROJECT_NAME = "contact-sound-ablation-v2"
NUM_EPOCHS = "50"  # Shorter training for ablation study
DATA_SUBSET_SIZE = "0.50"  # Use 50% of data instead of 20% to ensure all classes are represented

def run_data_generation(denoising_enabled, output_dir):
    """Run data generation with specified denoising setting."""
    # Create descriptive name
    setting_name = "denoising" if denoising_enabled else "no_denoising"
    
    # Prepare the command
    command = [
        "python3",
        "scripts/bag_audio_visual_segmentation.py",
        f"preprocessing.enable_denoising={str(denoising_enabled).lower()}",
        f"hydra.run.dir={os.path.join(output_dir, setting_name)}"
    ]
    
    # Print info
    print("\n" + "="*80)
    print(f"Running data generation with {setting_name}:")
    print(" ".join(command))
    print("="*80 + "\n")
    
    # Run the command
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"Data generation failed with return code {result.returncode}")
        return False
    else:
        print(f"Data generation completed successfully")
        return True

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
    
    # Create timestamp for run naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    study_dir = args.output_dir if args.output_dir else f"ablation_study_{timestamp}"
    os.makedirs(study_dir, exist_ok=True)
    
    # Determine denoising settings to test
    denoising_settings = [True, False] if args.denoising is None else [args.denoising]
    
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
    
    # Print experiment summary
    print(f"Running experiments with:")
    print(f"- Denoising settings: {denoising_settings}")
    print(f"- Binary classification: {use_binary}")
    print(f"- {len(augmentation_combinations)} augmentation combinations")
    
    # First, generate datasets with different denoising settings
    for denoising in denoising_settings:
        success = run_data_generation(denoising, study_dir)
        
        if not success:
            print(f"Skipping training for {'denoising' if denoising else 'no_denoising'}")
            continue
        
        # Now run training with different augmentation settings
        for aug_combo in augmentation_combinations:
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
            run_name = f"{binary_prefix}{'denoising' if denoising else 'no_denoising'}_{aug_suffix}"
            
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
            
            # Run the training experiment
            run_training_experiment(experiment_config, run_name)
            
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
    parser = argparse.ArgumentParser(description="Preprocessing ablation study")
    parser.add_argument(
        "--output-dir", 
        default=None, 
        help="Directory to save results (default: timestamped directory)"
    )
    parser.add_argument(
        "--denoising",
        type=lambda x: x.lower() == 'true',
        choices=[True, False],
        default=None,
        help="Set to true or false to run only one denoising setting (default: run both)"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=False,  # Make sure this is False by default
        help="Use binary classification (contact vs no-contact) instead of multi-class"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
