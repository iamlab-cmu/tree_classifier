import os
import subprocess
import time
import datetime
from typing import Dict, Any
import itertools

WANDB_PROJECT_NAME = "contact-sound-classification-augmentations-v3"
DATA_SUBSET_SIZE = "0.25"
NUM_EPOCHS = "25"  
EARLY_STOPPING_PATIENCE = "3"  

DATASET_VARIATIONS = [
    {
        "name": "default",
        "train_path": "audio_visual_dataset_default",
        "test_path": "audio_visual_dataset_robo_default",
        "do_norm": "true",  
        "description": "Default (normalized & denoised)"
    },
    {
        "name": "no_norm",
        "train_path": "audio_visual_dataset_no_norm",
        "test_path": "audio_visual_dataset_robo_no_norm",
        "do_norm": "false",  
        "description": "No normalization (only denoised)"
    },
    {
        "name": "no_denoise",
        "train_path": "audio_visual_dataset_no_denoise",
        "test_path": "audio_visual_dataset_robo_no_denoise",
        "do_norm": "true",  
        "description": "No denoising (only normalized)"
    },
    {
        "name": "raw",
        "train_path": "audio_visual_dataset_raw",
        "test_path": "audio_visual_dataset_robo_raw",
        "do_norm": "false",  
        "description": "Raw audio (no preprocessing)"
    }
]

def run_experiment(experiment_config: Dict[str, Any]) -> None:
    """Run a single experiment with the given configuration."""
    command = ["python3", "learning/train.py"]
    
    for key, value in experiment_config.items():
        command.append(f"{key}={value}")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*80)
    print(f"Running experiment at {timestamp}:")
    print(" ".join(command))
    print("="*80 + "\n")
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"Experiment failed with return code {result.returncode}")
    else:
        print(f"Experiment completed successfully")
    
    print("Waiting for 3 seconds before starting next experiment...")
    time.sleep(3)

def get_base_config(timestamp, dataset_variation):
    """Return the base configuration used for all experiments."""
    return {
        "model.use_images": "False",
        "model.use_audio": "True",
        "model.from_scratch": "False",
        "model.audio_model": "ast",
        "model.use_dual_audio": "True",
        "model.fusion_type": "transformer",
        
        "data.exclude_ambient": "False",
        "data.subset_size": DATA_SUBSET_SIZE,
        "data.use_subset": "true",
        "data.train_base_path": dataset_variation["train_path"],
        "data.test_base_path": dataset_variation["test_path"],
        "data.train_csv_path": os.path.join(dataset_variation["train_path"], "dataset.csv"),
        "data.test_csv_path": os.path.join(dataset_variation["test_path"], "dataset.csv"),
        "data.do_norm": dataset_variation["do_norm"],  
        
        "training.epochs": NUM_EPOCHS,
        "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        
        "logging.save_misclassified": "false",  
        "wandb.project": WANDB_PROJECT_NAME
    }

def main():
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    augmentation_types = [
        {
            "name": "Baseline-NoAugmentations",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "false",
                "data.augmentations.audio_frequency_shift": "false",
                "data.augmentations.audio_power": "false",
                "data.augmentations.audio_gaussian_noise": "false",
                "data.augmentations.audio_robot_humming": "false",
                "data.augmentations.audio_harmonic_distortion": "false",
            }
        },
        {
            "name": "FrequencyScaling",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "true",
                "data.augmentations.audio_frequency_shift": "false",
                "data.augmentations.audio_power": "false",
                "data.augmentations.audio_gaussian_noise": "false",
                "data.augmentations.audio_robot_humming": "false",
                "data.augmentations.audio_harmonic_distortion": "false",
                "data.augmentations.frequency_scaling_range": "[0.8, 1.2]",
            }
        },
        {
            "name": "PowerAugmentation",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "false",
                "data.augmentations.audio_frequency_shift": "false",
                "data.augmentations.audio_power": "true",
                "data.augmentations.audio_gaussian_noise": "false",
                "data.augmentations.audio_robot_humming": "false",
                "data.augmentations.audio_harmonic_distortion": "false",
                "data.augmentations.power_range": "[0.7, 1.3]",
            }
        },
        {
            "name": "GaussianNoise",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "false",
                "data.augmentations.audio_frequency_shift": "false",
                "data.augmentations.audio_power": "false",
                "data.augmentations.audio_gaussian_noise": "true",
                "data.augmentations.audio_robot_humming": "false",
                "data.augmentations.audio_harmonic_distortion": "false",
                "data.augmentations.noise_level_range": "[0.001, 0.01]",
            }
        },
        {
            "name": "RobotHumming",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "false",
                "data.augmentations.audio_frequency_shift": "false",
                "data.augmentations.audio_power": "false",
                "data.augmentations.audio_gaussian_noise": "false",
                "data.augmentations.audio_robot_humming": "true",
                "data.augmentations.audio_harmonic_distortion": "false",
                "data.augmentations.robot_humming_mix_ratio": "[0.1, 0.4]",
                "data.augmentations.robot_humming_audio_path": "robot_humming_samples",
            }
        },
        {
            "name": "HarmonicDistortion",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "false",
                "data.augmentations.audio_frequency_shift": "false",
                "data.augmentations.audio_power": "false",
                "data.augmentations.audio_gaussian_noise": "false",
                "data.augmentations.audio_robot_humming": "false",
                "data.augmentations.audio_harmonic_distortion": "true",
                "data.augmentations.harmonic_distortion_range": "[0.1, 0.5]",
            }
        },
        {
            "name": "AllAugmentations",
            "augmentations": {
                "data.augmentations.audio_frequency_scaling": "true",
                "data.augmentations.audio_frequency_shift": "true",
                "data.augmentations.audio_power": "true",
                "data.augmentations.audio_gaussian_noise": "true",
                "data.augmentations.audio_robot_humming": "true",
                "data.augmentations.audio_harmonic_distortion": "true",
                "data.augmentations.harmonic_distortion_range": "[0.1, 0.5]",
                "data.augmentations.robot_humming_audio_path": "robot_humming_samples",
            }   
        }
    ]
    
    all_experiments = []
    
    for dataset_var in DATASET_VARIATIONS:
        base_config = get_base_config(timestamp, dataset_var)
        
        for aug_type in augmentation_types:
            experiment = {
                **base_config,
                **aug_type["augmentations"],
                "wandb.run_name": f"Audio-{dataset_var['name']}-{aug_type['name']}-{timestamp}"
            }
            all_experiments.append(experiment)
    
    print(f"Running {len(all_experiments)} experiments across {len(DATASET_VARIATIONS)} dataset variations:")
    
    for dataset_var in DATASET_VARIATIONS:
        print(f"\n{dataset_var['name']} - {dataset_var['description']}:")
        for aug_type in augmentation_types:
            print(f"  - {aug_type['name']}")
    
    print(f"\nTotal: {len(all_experiments)} experiments\n")
    
    confirm = input("Do you want to run all these experiments? (y/n): ")
    if confirm.lower() != 'y':
        print("Experiments cancelled.")
        return
    
    for i, experiment in enumerate(all_experiments, 1):
        print(f"\nStarting experiment {i}/{len(all_experiments)}: {experiment['wandb.run_name']}")
        run_experiment(experiment)

if __name__ == "__main__":
    main() 