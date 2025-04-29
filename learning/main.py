import os
import pandas as pd
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from hydra.utils import to_absolute_path

from models import MultiModalClassifier
from datasets import AudioVisualDataset, custom_collate
from training import train_model, evaluate_model
from evaluation import plot_confusion_matrix, save_misclassified_samples
from utils import (
    set_seed,
    setup_output_dir,
    initialize_wandb,
    setup_robot_humming_dir,
    get_device,
    log_classification_metrics,
    convert_to_binary_classification,
    get_class_names,
    update_num_classes,
)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """Main function to run the training pipeline with random test set resampling"""
    print(
        f"Running main training script with config: {cfg.model.use_images=}, {cfg.model.use_audio=}"
    )

    orig_cwd = hydra.utils.get_original_cwd()
    print(f"Original working directory: {orig_cwd}")

    setup_output_dir(cfg.logging.output_dir)

    setup_robot_humming_dir(cfg)

    device = get_device()

    set_seed(cfg.training.seed)

    initialize_wandb(cfg)

    cfg = update_num_classes(cfg)

    print(f"\nLoading dataset from {cfg.data.train_csv_path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Absolute path to train CSV: {to_absolute_path(cfg.data.train_csv_path)}")
    print(
        f"Absolute path to train base path: {to_absolute_path(cfg.data.train_base_path)}"
    )
    dataset_path = os.path.join(
        orig_cwd, cfg.data.train_base_path, cfg.data.train_csv_path
    )
    print(f"Looking for dataset at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} samples")

    if cfg.data.get("exclude_ambient", False):
        print("\n" + "=" * 50)
        print("EXCLUDING AMBIENT SAMPLES")
        print("=" * 50 + "\n")

        total_before = len(df)
        ambient_count = len(df[df["category"] == "ambient"])

        df = df[df["category"] != "ambient"]

        print(f"Removed {ambient_count} ambient samples from dataset")
        print(f"Dataset size reduced from {total_before} to {len(df)} samples")

        if "ambient" in cfg.data.classes:
            cfg.data.classes = [c for c in cfg.data.classes if c != "ambient"]
            cfg.model.num_classes = len(cfg.data.classes)
            print(f"Updated classes: {cfg.data.classes}")
            print(f"Updated num_classes: {cfg.model.num_classes}")

    if cfg.data.use_subset:
        print(f"\n{'=' * 50}")
        print(
            f"USING SUBSET OF DATA: {cfg.data.subset_size * 100:.1f}% of original dataset"
        )
        print(f"{'=' * 50}\n")

        df = (
            df.groupby("category", group_keys=False)
            .apply(
                lambda x: x.sample(
                    frac=cfg.data.subset_size, random_state=cfg.training.seed
                )
            )
            .reset_index(drop=True)
        )

        print(f"Subset contains {len(df)} samples")

    print("\nOriginal category distribution:")
    category_counts = df["category"].value_counts()
    print(category_counts)

    print("\nBalancing training dataset...")
    category_counts = df["category"].value_counts()
    print("Original category distribution:")
    print(category_counts)

    if cfg.data.get("category_balancing", {}).get("enabled", False):
        min_samples = cfg.data.category_balancing.min_samples_per_category
        print(f"\n{'=' * 50}")
        print(f"USING CUSTOM CATEGORY BALANCING: {min_samples} samples per category")
        print(f"{'=' * 50}\n")

        smallest_category_count = category_counts.min()

        if min_samples > smallest_category_count:
            print(
                f"Warning: Requested {min_samples} samples per category, but smallest category only has {smallest_category_count}"
            )
            print(
                f"Will balance to {smallest_category_count} samples per category instead"
            )
            minority_class_count = smallest_category_count
        else:
            print(f"Balancing all categories to {min_samples} samples")
            minority_class_count = min_samples
    else:
        minority_class_count = category_counts.min()
        minority_class = category_counts.idxmin()
        print(f"Minority class: '{minority_class}' with {minority_class_count} samples")

    balanced_df = pd.DataFrame()

    for category in category_counts.index:
        category_df = df[df["category"] == category]
        category_count = len(category_df)

        if category_count <= minority_class_count:
            downsampled_df = category_df
        else:
            downsampled_df = category_df.sample(
                minority_class_count, random_state=cfg.training.seed
            )

        balanced_df = pd.concat([balanced_df, downsampled_df])

    balanced_df = balanced_df.sample(
        frac=1, random_state=cfg.training.seed
    ).reset_index(drop=True)

    df = balanced_df

    print("\nBalanced category distribution (downsampled):")
    print(df["category"].value_counts())
    print("=" * 50 + "\n")

    category_classes = get_class_names(cfg)

    print("\n" + "=" * 50)
    print("SPLITTING DATA INTO TRAIN AND VALIDATION SETS")
    print("=" * 50 + "\n")

    train_ratio = 0.8
    val_ratio = 1 - train_ratio

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=cfg.training.seed,
        stratify=df["category"],  # Maintain class distribution
    )

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    train_dataset = AudioVisualDataset(
        train_df,
        base_path=to_absolute_path(cfg.data.train_base_path),
        img_size=cfg.data.img_size,
        use_binary_classification=cfg.data.get("use_binary_classification", False),
        augment=True,  # Enable augmentations for training data
        aug_config=cfg.data.get("augmentations", {}),
        do_norm=cfg.data.do_norm,
    )

    val_dataset = AudioVisualDataset(
        val_df,
        base_path=to_absolute_path(cfg.data.train_base_path),
        img_size=cfg.data.img_size,
        use_binary_classification=cfg.data.get("use_binary_classification", False),
        augment=False,  # No augmentations for validation data
        aug_config=cfg.data.get("augmentations", {}),
        do_norm=cfg.data.do_norm,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate,
    )

    model = MultiModalClassifier(
        num_classes=cfg.model.num_classes,
        use_images=cfg.model.use_images,
        use_audio=cfg.model.use_audio,
        pretrained=not cfg.model.from_scratch,
        audio_model=cfg.model.audio_model,
        use_dual_audio=cfg.model.use_dual_audio,
        fusion_type=cfg.model.fusion_type,
    )

    initial_model_path = os.path.join(cfg.logging.output_dir, "model_architecture.pt")
    torch.save(
        {
            "num_classes": cfg.model.num_classes,
            "use_images": cfg.model.use_images,
            "use_audio": cfg.model.use_audio,
            "pretrained": not cfg.model.from_scratch,
            "audio_model": cfg.model.audio_model,
            "use_dual_audio": cfg.model.use_dual_audio,
            "fusion_type": cfg.model.fusion_type,
        },
        initial_model_path,
    )
    print(f"Saved model architecture configuration to {initial_model_path}")

    history, val_preds, val_labels = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    model_path = os.path.join(cfg.logging.output_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model weights to {model_path}")

    print(f"\nEvaluating model on validation data...")
    val_preds_eval, val_labels_eval, val_accuracy = evaluate_model(
        model=model, data_loader=val_loader, device=device
    )
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    plot_confusion_matrix(val_labels, val_preds, category_classes, cfg)

    report = classification_report(
        val_labels,
        val_preds,
        target_names=category_classes,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nClassification Report:")
    print(
        classification_report(
            val_labels, val_preds, target_names=category_classes, zero_division=0
        )
    )

    f1_scores = []
    for class_idx, class_name in enumerate(category_classes):
        class_f1 = report[class_name]["f1-score"]
        f1_scores.append(class_f1)
        print(f"F1 score for class '{class_name}': {class_f1:.4f}")

    f1_macro = report["macro avg"]["f1-score"]
    f1_weighted = report["weighted avg"]["f1-score"]

    print(f"Macro-averaged F1: {f1_macro:.4f}")
    print(f"Weighted-averaged F1: {f1_weighted:.4f}")

    if cfg.wandb.enabled:
        import wandb

        wandb.log(
            {
                "val/f1_macro": f1_macro,
                "val/f1_weighted": f1_weighted,
                "val/accuracy": val_accuracy,
            }
        )

        for i, class_name in enumerate(category_classes):
            wandb.log({f"val/f1_{class_name}": f1_scores[i]})

    test_csv_direct = to_absolute_path(cfg.data.test_csv_path)
    test_csv_joined = os.path.join(
        to_absolute_path(cfg.data.test_base_path), cfg.data.test_csv_path
    )

    print("\n" + "=" * 50)
    print(f"ROBOT DATA CONFIGURATION:")
    print(f"use_robot_data: {cfg.data.use_robot_data}")
    print(f"test_csv_direct exists: {os.path.exists(test_csv_direct)}")
    print(f"test_csv_joined exists: {os.path.exists(test_csv_joined)}")
    print("=" * 50 + "\n")

    if cfg.data.use_robot_data and (
        os.path.exists(test_csv_direct) or os.path.exists(test_csv_joined)
    ):
        print("\n" + "=" * 50)
        print("EVALUATING ON TEST SET (ROBOT DATA)")
        print("=" * 50 + "\n")

        test_csv_path = (
            test_csv_direct if os.path.exists(test_csv_direct) else test_csv_joined
        )
        test_df = pd.read_csv(test_csv_path)
        print(f"\nLoaded test dataset with {len(test_df)} samples")

        if cfg.data.get("exclude_ambient", False):
            ambient_count = len(test_df[test_df["category"] == "ambient"])
            test_df = test_df[test_df["category"] != "ambient"]
            print(f"Removed {ambient_count} ambient samples from test dataset")
            print(f"Test dataset size reduced to {len(test_df)} samples")

        if cfg.get("test_resampling", {}).get("enabled", False):
            print("\n" + "=" * 50)
            print("PERFORMING TEST SET RANDOM RESAMPLING")
            print("=" * 50 + "\n")

            n_iterations = cfg.get("test_resampling", {}).get("iterations", 5)
            resample_fraction = cfg.get("test_resampling", {}).get("fraction", 0.8)

            print(f"\nPerforming {n_iterations} random resampling iterations")
            print(
                f"Each resample will use {resample_fraction * 100:.1f}% of the test data"
            )

            test_iterations_f1_macro = []
            test_iterations_f1_weighted = []
            test_iterations_accuracy = []
            test_iterations_class_f1 = [[] for _ in range(len(category_classes))]

            samples_per_category = {}
            for category in test_df["category"].unique():
                category_samples = len(test_df[test_df["category"] == category])
                samples_per_category[category] = int(
                    category_samples * resample_fraction
                )
                print(
                    f"Category '{category}': {samples_per_category[category]} samples per iteration"
                )

            for iteration in range(n_iterations):
                print(f"\n{'=' * 50}")
                print(f"TEST RESAMPLE ITERATION {iteration + 1}/{n_iterations}")
                print(f"{'=' * 50}\n")

                test_resample_df = pd.DataFrame()
                for category in test_df["category"].unique():
                    category_df = test_df[test_df["category"] == category]
                    sampled_df = category_df.sample(
                        n=samples_per_category[category],
                        random_state=cfg.training.seed + iteration,
                        replace=False,
                    )
                    test_resample_df = pd.concat([test_resample_df, sampled_df])

                test_resample_df = test_resample_df.sample(
                    frac=1, random_state=cfg.training.seed + iteration
                ).reset_index(drop=True)

                print(f"Resample contains {len(test_resample_df)} samples")
                print(f"Category distribution:")
                print(test_resample_df["category"].value_counts())

                test_dataset = AudioVisualDataset(
                    test_resample_df,
                    base_path=to_absolute_path(cfg.data.test_base_path),
                    img_size=cfg.data.img_size,
                    use_binary_classification=cfg.data.get(
                        "use_binary_classification", False
                    ),
                    augment=False,  # No augmentations for test data
                    aug_config=cfg.data.get("augmentations", {}),
                    do_norm=cfg.data.do_norm,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.num_workers,
                    collate_fn=custom_collate,
                )

                test_preds, test_labels, test_accuracy = evaluate_model(
                    model=model, data_loader=test_loader, device=device
                )

                print(f"Iteration {iteration + 1} Test Accuracy: {test_accuracy:.4f}")

                test_report = classification_report(
                    test_labels,
                    test_preds,
                    target_names=category_classes,
                    output_dict=True,
                )

                test_iterations_f1_macro.append(test_report["macro avg"]["f1-score"])
                test_iterations_f1_weighted.append(
                    test_report["weighted avg"]["f1-score"]
                )
                test_iterations_accuracy.append(test_accuracy)

                for i, class_name in enumerate(category_classes):
                    if class_name in test_report:
                        test_iterations_class_f1[i].append(
                            test_report[class_name]["f1-score"]
                        )

                if cfg.wandb.enabled:
                    import wandb

                    wandb.log(
                        {
                            f"test_iteration_{iteration + 1}/accuracy": test_accuracy,
                            f"test_iteration_{iteration + 1}/f1_macro": test_report[
                                "macro avg"
                            ]["f1-score"],
                            f"test_iteration_{iteration + 1}/f1_weighted": test_report[
                                "weighted avg"
                            ]["f1-score"],
                        }
                    )

                    for i, class_name in enumerate(category_classes):
                        if class_name in test_report:
                            wandb.log(
                                {
                                    f"test_iteration_{iteration + 1}/f1_{class_name}": test_report[
                                        class_name
                                    ]["f1-score"]
                                }
                            )

            print("\n" + "=" * 50)
            print("TEST SET RANDOM RESAMPLING SUMMARY")
            print("=" * 50 + "\n")

            test_mean_accuracy = np.mean(test_iterations_accuracy)
            test_std_accuracy = np.std(test_iterations_accuracy)

            test_mean_f1_macro = np.mean(test_iterations_f1_macro)
            test_std_f1_macro = np.std(test_iterations_f1_macro)
            test_var_f1_macro = np.var(test_iterations_f1_macro)

            test_mean_f1_weighted = np.mean(test_iterations_f1_weighted)
            test_std_f1_weighted = np.std(test_iterations_f1_weighted)
            test_var_f1_weighted = np.var(test_iterations_f1_weighted)

            print(
                f"Test Mean Accuracy: {test_mean_accuracy:.4f} (±{test_std_accuracy:.4f})"
            )
            print()
            print(f"Test Mean Macro-averaged F1 Score: {test_mean_f1_macro:.4f}")
            print(
                f"Test Standard Deviation of Macro-averaged F1 Scores: {test_std_f1_macro:.4f}"
            )
            print(f"Test Variance of Macro-averaged F1 Scores: {test_var_f1_macro:.4f}")
            print()

            print(f"Test Mean Weighted-averaged F1 Score: {test_mean_f1_weighted:.4f}")
            print(
                f"Test Standard Deviation of Weighted-averaged F1 Scores: {test_std_f1_weighted:.4f}"
            )
            print(
                f"Test Variance of Weighted-averaged F1 Scores: {test_var_f1_weighted:.4f}"
            )
            print()

            print("Test Per-class F1 Score Summary:")
            for i, class_name in enumerate(category_classes):
                if test_iterations_class_f1[i]:
                    class_mean = np.mean(test_iterations_class_f1[i])
                    class_std = np.std(test_iterations_class_f1[i])
                    class_var = np.var(test_iterations_class_f1[i])

                    print(f"  {class_name}:")
                    print(f"    Mean F1 Score: {class_mean:.4f}")
                    print(f"    Standard Deviation: {class_std:.4f}")
                    print(f"    Variance: {class_var:.4f}")

            if cfg.wandb.enabled:
                import wandb

                wandb.log(
                    {
                        "test_summary/accuracy_mean": test_mean_accuracy,
                        "test_summary/accuracy_std": test_std_accuracy,
                        "test_summary/f1_macro_mean": test_mean_f1_macro,
                        "test_summary/f1_macro_std": test_std_f1_macro,
                        "test_summary/f1_macro_var": test_var_f1_macro,
                        "test_summary/f1_weighted_mean": test_mean_f1_weighted,
                        "test_summary/f1_weighted_std": test_std_f1_weighted,
                        "test_summary/f1_weighted_var": test_var_f1_weighted,
                    }
                )

                for i, class_name in enumerate(category_classes):
                    if test_iterations_class_f1[i]:
                        wandb.log(
                            {
                                f"test_summary/f1_{class_name}_mean": np.mean(
                                    test_iterations_class_f1[i]
                                ),
                                f"test_summary/f1_{class_name}_std": np.std(
                                    test_iterations_class_f1[i]
                                ),
                                f"test_summary/f1_{class_name}_var": np.var(
                                    test_iterations_class_f1[i]
                                ),
                            }
                        )
        else:
            print("\n" + "=" * 50)
            print("EVALUATING ON FULL TEST SET (NO RESAMPLING)")
            print("=" * 50 + "\n")

            test_dataset = AudioVisualDataset(
                test_df,
                base_path=to_absolute_path(cfg.data.test_base_path),
                img_size=cfg.data.img_size,
                use_binary_classification=cfg.data.get(
                    "use_binary_classification", False
                ),
                augment=False,  # No augmentations for test data
                aug_config=cfg.data.get("augmentations", {}),
                do_norm=cfg.data.do_norm,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                collate_fn=custom_collate,
            )

            test_preds, test_labels, test_accuracy = evaluate_model(
                model=model, data_loader=test_loader, device=device
            )

            print(f"Test Accuracy: {test_accuracy:.4f}")

            test_report = classification_report(
                test_labels, test_preds, target_names=category_classes, output_dict=True
            )

            print("\nTest Classification Report:")
            print(
                classification_report(
                    test_labels, test_preds, target_names=category_classes
                )
            )

            if cfg.wandb.enabled:
                import wandb

                wandb.log(
                    {
                        "test/accuracy": test_accuracy,
                        "test/f1_macro": test_report["macro avg"]["f1-score"],
                        "test/f1_weighted": test_report["weighted avg"]["f1-score"],
                    }
                )

                for class_name in category_classes:
                    if class_name in test_report:
                        wandb.log(
                            {
                                f"test/f1_{class_name}": test_report[class_name][
                                    "f1-score"
                                ]
                            }
                        )
    else:
        print("\n" + "=" * 50)
        print("ROBOT DATA NOT AVAILABLE OR DISABLED")
        print("=" * 50 + "\n")
        print("Skipping test set evaluation")

    if cfg.wandb.enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()

