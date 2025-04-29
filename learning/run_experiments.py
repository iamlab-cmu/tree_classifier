import os
import subprocess
import time
import datetime
from typing import Dict, Any

# Hard-coded configuration parameters
WANDB_PROJECT_NAME = "contact-sound-classification-final-final-final"
DATA_SUBSET_SIZE = "0.30"  
NUM_EPOCHS = "50"  
EARLY_STOPPING_PATIENCE = "3"  

def run_experiment(experiment_config: Dict[str, Any]) -> None:
    """Run a single experiment with the given configuration."""
    # Format the command
    command = ["python3", "learning/main_train_2.py"]
    
    # Add all configuration overrides
    for key, value in experiment_config.items():
        command.append(f"{key}={value}")
    
    # Print experiment info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "="*80)
    print(f"Running experiment at {timestamp}:")
    print(" ".join(command))
    print("="*80 + "\n")
    
    # Run the experiment
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"Experiment failed with return code {result.returncode}")
    else:
        print(f"Experiment completed successfully")
    
    # Wait a bit before starting the next experiment
    print("Waiting for 15 seconds before starting next experiment...")
    time.sleep(15)

def main():
    # Get current timestamp for run names
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    # Define modality ablation experiments
    experiments = [
        # AST + Image (pretrained)
        {
            "model.use_images": "True",
            "model.use_audio": "True",
            "model.from_scratch": "False",  # Pretrained
            "model.audio_model": "ast",
            "model.use_dual_audio": "False",  # Only use AST
            "data.exclude_ambient": "False",
            "data.subset_size": DATA_SUBSET_SIZE,
            "data.use_subset": "true",
            "training.epochs": NUM_EPOCHS,
            "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "wandb.run_name": f"AST-Image-Pretrained-{timestamp}",
            "wandb.project": WANDB_PROJECT_NAME
        },
        
        # CLAP + Image (pretrained)
        {
            "model.use_images": "True",
            "model.use_audio": "True",
            "model.from_scratch": "False",  # Pretrained
            "model.audio_model": "clap",
            "model.use_dual_audio": "False",  # Only use CLAP
            "data.exclude_ambient": "False",
            "data.subset_size": DATA_SUBSET_SIZE,
            "data.use_subset": "true",
            "training.epochs": NUM_EPOCHS,
            "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "wandb.run_name": f"CLAP-Image-Pretrained-{timestamp}",
            "wandb.project": WANDB_PROJECT_NAME
        }
        
        # Commented out original experiments:
        # # Image-only experiments
        # {
        #     "model.use_images": "True",
        #     "model.use_audio": "False",
        #     "model.from_scratch": "False",  # Pretrained
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"ImageOnly-Pretrained-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # {
        #     "model.use_images": "True",
        #     "model.use_audio": "False",
        #     "model.from_scratch": "True",  # Not pretrained
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"ImageOnly-FromScratch-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # 
        # # Audio-only experiments with AST
        # {
        #     "model.use_images": "False",
        #     "model.use_audio": "True",
        #     "model.from_scratch": "False",  # Pretrained
        #     "model.audio_model": "ast",
        #     "model.use_dual_audio": "False",
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"AudioOnly-AST-Pretrained-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # {
        #     "model.use_images": "False",
        #     "model.use_audio": "True",
        #     "model.from_scratch": "True",  # Not pretrained
        #     "model.audio_model": "ast",
        #     "model.use_dual_audio": "False",
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"AudioOnly-AST-FromScratch-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # 
        # # Audio-only experiments with CLAP
        # {
        #     "model.use_images": "False",
        #     "model.use_audio": "True",
        #     "model.from_scratch": "False",  # Pretrained
        #     "model.audio_model": "clap",
        #     "model.use_dual_audio": "False",
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"AudioOnly-CLAP-Pretrained-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # 
        # # Audio-only experiments with dual audio (AST + CLAP)
        # {
        #     "model.use_images": "False",
        #     "model.use_audio": "True",
        #     "model.from_scratch": "False",  # Pretrained
        #     "model.audio_model": "ast",
        #     "model.use_dual_audio": "True",  # Use both AST and CLAP
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"AudioOnly-DualAudio-Pretrained-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # 
        # # Dual audio + Image experiment (full model)
        # {
        #     "model.use_images": "True",
        #     "model.use_audio": "True",
        #     "model.from_scratch": "False",  # Pretrained
        #     "model.audio_model": "ast",
        #     "model.use_dual_audio": "True",  # Use both AST and CLAP
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"Multimodal-DualAudio-Pretrained-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # },
        # {
        #     "model.use_images": "True",
        #     "model.use_audio": "True",
        #     "model.from_scratch": "True",  # Not pretrained
        #     "model.audio_model": "ast",
        #     "model.use_dual_audio": "True",  # Use both AST and CLAP
        #     "data.exclude_ambient": "False",
        #     "data.subset_size": DATA_SUBSET_SIZE,
        #     "data.use_subset": "true",
        #     "training.epochs": NUM_EPOCHS,
        #     "training.early_stopping_patience": EARLY_STOPPING_PATIENCE,
        #     "wandb.run_name": f"Multimodal-DualAudio-FromScratch-{timestamp}",
        #     "wandb.project": WANDB_PROJECT_NAME
        # }   
    ]
    
    # Print experiment summary
    print(f"Running {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['wandb.run_name']}")
    
    # Run each experiment
    for i, experiment in enumerate(experiments, 1):
        print(f"\nStarting experiment {i}/{len(experiments)}: {experiment['wandb.run_name']}")
        run_experiment(experiment)

if __name__ == "__main__":
    main() 