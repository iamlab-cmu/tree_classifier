import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import wandb


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def initialize_wandb(cfg, run_name_suffix=None):
    """Initialize WandB if enabled"""
    if not cfg.wandb.enabled:
        return None

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_name_suffix:
        run_name = f"{cfg.wandb.run_name}_{run_name_suffix}_{timestamp}"
    else:
        run_name = f"{cfg.wandb.run_name}_{timestamp}"

    from omegaconf import OmegaConf

    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg.logging.output_dir,
    )

    print(f"Initialized wandb run: {run_name}")
    return run


def setup_robot_humming_dir(cfg):
    """Create robot humming directory and README if needed"""
    if cfg.data.get("augmentations", {}).get("audio_robot_humming", False):
        from hydra.utils import to_absolute_path

        robot_humming_path = to_absolute_path(
            cfg.data.augmentations.get("robot_humming_audio_path", "")
        )
        if not os.path.exists(robot_humming_path):
            print(f"Robot humming directory not found: {robot_humming_path}")
            print("Creating directory and a README file...")
            os.makedirs(robot_humming_path, exist_ok=True)

            readme_path = os.path.join(robot_humming_path, "README.txt")
            with open(readme_path, "w") as f:
                f.write("Robot Humming Samples Directory\n")
                f.write("==============================\n\n")
                f.write(
                    "Place robot humming audio samples (WAV, MP3, OGG) in this directory.\n"
                )
                f.write(
                    "These samples will be used for data augmentation to mix with training audio.\n"
                )
                f.write(
                    "Samples should ideally be robot motor/movement sounds without contact sounds.\n"
                )

            print(f"Created directory: {robot_humming_path}")
            print(f"Added README file: {readme_path}")
            print(
                "Please add robot humming audio samples to this directory for augmentation to work."
            )


def get_device():
    """Get device to use for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def log_classification_metrics(report, category_classes, binary_mode=False, prefix=""):
    """Log classification metrics to wandb"""
    if not wandb.run:
        return

    metrics_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1 Score"])
    for cls in category_classes:
        metrics_table.add_data(
            cls,
            report[cls]["precision"],
            report[cls]["recall"],
            report[cls]["f1-score"],
        )
    metrics_table.add_data(
        "macro avg",
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
    )
    metrics_table.add_data(
        "weighted avg",
        report["weighted avg"]["precision"],
        report["weighted avg"]["recall"],
        report["weighted avg"]["f1-score"],
    )

    table_name = f"{prefix}{'binary_' if binary_mode else ''}classification_metrics"
    wandb.log({table_name: metrics_table})

    accuracy_name = f"{prefix}{'binary_' if binary_mode else ''}accuracy"
    wandb.log({accuracy_name: report["accuracy"]})


def convert_to_binary_classification(labels, preds):
    """Convert multi-class labels/predictions to binary (contact vs. no-contact)"""
    binary_labels = np.array([1 if label == 3 else 0 for label in labels])
    binary_preds = np.array([1 if pred == 3 else 0 for pred in preds])
    return binary_labels, binary_preds


def get_class_names(cfg):
    """Get class names based on configuration"""
    if cfg.data.get("use_binary_classification", False):
        return ["contact", "no-contact"]
    else:
        return cfg.data.classes


def update_num_classes(cfg):
    """Update the number of classes in the configuration based on the data configuration"""
    if cfg.data.get("use_binary_classification", False):
        print("\n" + "=" * 50)
        print("USING BINARY CLASSIFICATION: contact vs no-contact")
        print("  - 'leaf', 'twig', 'trunk' → 'contact' (class 0)")
        print("  - 'ambient' → 'no-contact' (class 1)")
        print("=" * 50 + "\n")
        cfg.model.num_classes = 2
    else:
        if len(cfg.data.classes) != cfg.model.num_classes:
            print(
                f"Warning: Mismatch between number of classes in data config ({len(cfg.data.classes)}) "
                f"and model config ({cfg.model.num_classes})"
            )
            print(f"Updating model.num_classes to {len(cfg.data.classes)}")
            cfg.model.num_classes = len(cfg.data.classes)
    return cfg

