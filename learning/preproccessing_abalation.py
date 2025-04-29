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

WANDB_PROJECT_NAME = "contact-sound-ablation-v2"
NUM_EPOCHS = "50"  # Shorter training for ablation study
DATA_SUBSET_SIZE = (
    "0.50"  # Use 50% of data instead of 20% to ensure all classes are represented
)


def run_data_generation(denoising_enabled, output_dir):
    """Run data generation with specified denoising setting."""
    setting_name = "denoising" if denoising_enabled else "no_denoising"

    command = [
        "python3",
        "scripts/bag_audio_visual_segmentation.py",
        f"preprocessing.enable_denoising={str(denoising_enabled).lower()}",
        f"hydra.run.dir={os.path.join(output_dir, setting_name)}",
    ]

    print("\n" + "=" * 80)
    print(f"Running data generation with {setting_name}:")
    print(" ".join(command))
    print("=" * 80 + "\n")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Data generation failed with return code {result.returncode}")
        return False
    else:
        print(f"Data generation completed successfully")
        return True


def run_training_experiment(experiment_config, run_name):
    """Run training with specified configuration."""
    command = ["python3", "learning/main_train_2.py"]

    for key, value in experiment_config.items():
        command.append(f"{key}={value}")

    print("\n" + "=" * 80)
    print(f"Running training experiment {run_name}:")
    print(" ".join(command))
    print("=" * 80 + "\n")

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
    study_dir = args.output_dir if args.output_dir else f"ablation_study_{timestamp}"
    os.makedirs(study_dir, exist_ok=True)

    denoising_settings = [True, False] if args.denoising is None else [args.denoising]

    use_binary = False  # Force this to false to ensure multi-class classification
    binary_mode_str = "false"  # Always use multi-class

    augmentation_combinations = [
        {
            "data.augmentations.image_rotation": "false",
            "data.augmentations.audio_frequency_scaling": "false",
            "data.augmentations.audio_robot_humming": "false",
        },
        {
            "data.augmentations.image_rotation": "true",
            "data.augmentations.audio_frequency_scaling": "false",
            "data.augmentations.audio_robot_humming": "false",
        },
        {
            "data.augmentations.image_rotation": "false",
            "data.augmentations.audio_frequency_scaling": "true",
            "data.augmentations.audio_robot_humming": "false",
        },
        {
            "data.augmentations.image_rotation": "false",
            "data.augmentations.audio_frequency_scaling": "false",
            "data.augmentations.audio_robot_humming": "true",
        },
        {
            "data.augmentations.image_rotation": "true",
            "data.augmentations.audio_frequency_scaling": "true",
            "data.augmentations.audio_robot_humming": "true",
        },
    ]

    print(f"Running experiments with:")
    print(f"- Denoising settings: {denoising_settings}")
    print(f"- Binary classification: {use_binary}")
    print(f"- {len(augmentation_combinations)} augmentation combinations")

    for denoising in denoising_settings:
        success = run_data_generation(denoising, study_dir)

        if not success:
            print(
                f"Skipping training for {'denoising' if denoising else 'no_denoising'}"
            )
            continue

        for aug_combo in augmentation_combinations:
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

            binary_prefix = "binary_" if use_binary else ""
            run_name = f"{binary_prefix}{'denoising' if denoising else 'no_denoising'}_{aug_suffix}"

            experiment_config = aug_combo.copy()
            experiment_config.update(
                {
                    "data.use_subset": "true",
                    "data.subset_size": DATA_SUBSET_SIZE,
                    "data.use_binary_classification": binary_mode_str,
                    "model.num_classes": "4",  # Explicitly set to 4 classes
                    "data.val_split": "0.2",  # Use a larger validation split
                    "training.epochs": NUM_EPOCHS,
                    "wandb.run_name": run_name,
                    "wandb.project": WANDB_PROJECT_NAME,
                    "hydra.run.dir": os.path.join(study_dir, run_name),
                }
            )

            run_training_experiment(experiment_config, run_name)

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
        help="Directory to save results (default: timestamped directory)",
    )
    parser.add_argument(
        "--denoising",
        type=lambda x: x.lower() == "true",
        choices=[True, False],
        default=None,
        help="Set to true or false to run only one denoising setting (default: run both)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=False,  # Make sure this is False by default
        help="Use binary classification (contact vs no-contact) instead of multi-class",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
