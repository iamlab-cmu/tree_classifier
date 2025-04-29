import os
import pandas as pd
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
    update_num_classes
)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """Main function to run the training pipeline"""
    print(f"Running main training script with config: {cfg.model.use_images=}, {cfg.model.use_audio=}")
    
    # Get the original working directory
    orig_cwd = hydra.utils.get_original_cwd()
    print(f"Original working directory: {orig_cwd}")
    
    # Create output directory
    setup_output_dir(cfg.logging.output_dir)
    
    # Setup robot humming directory if needed
    setup_robot_humming_dir(cfg)
    
    # Get device for training
    device = get_device()
    
    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    
    # Initialize wandb if enabled
    initialize_wandb(cfg)
    
    # Update number of classes based on configuration
    cfg = update_num_classes(cfg)
    
    # Load dataset
    print(f"\nLoading dataset from {cfg.data.train_csv_path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Absolute path to train CSV: {to_absolute_path(cfg.data.train_csv_path)}")
    print(f"Absolute path to train base path: {to_absolute_path(cfg.data.train_base_path)}")
    dataset_path = os.path.join(orig_cwd, cfg.data.train_base_path, cfg.data.train_csv_path)
    print(f"Looking for dataset at: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Check if we need to exclude ambient samples
    if cfg.data.get('exclude_ambient', False):
        print("\n" + "="*50)
        print("EXCLUDING AMBIENT SAMPLES")
        print("="*50 + "\n")
        
        # Count before filtering
        total_before = len(df)
        ambient_count = len(df[df['category'] == 'ambient'])
        
        # Filter out ambient samples
        df = df[df['category'] != 'ambient']
        
        print(f"Removed {ambient_count} ambient samples from dataset")
        print(f"Dataset size reduced from {total_before} to {len(df)} samples")
        
        # Update classes list and num_classes
        if 'ambient' in cfg.data.classes:
            cfg.data.classes = [c for c in cfg.data.classes if c != 'ambient']
            cfg.model.num_classes = len(cfg.data.classes)
            print(f"Updated classes: {cfg.data.classes}")
            print(f"Updated num_classes: {cfg.model.num_classes}")
    
    # Use subset of data if specified
    if cfg.data.use_subset:
        print(f"\n{'='*50}")
        print(f"USING SUBSET OF DATA: {cfg.data.subset_size*100:.1f}% of original dataset")
        print(f"{'='*50}\n")
        
        # Stratified sampling to maintain class distribution
        df = df.groupby('category', group_keys=False).apply(
            lambda x: x.sample(frac=cfg.data.subset_size, random_state=cfg.training.seed)
        ).reset_index(drop=True)
        
        print(f"Subset contains {len(df)} samples")
    
    # Display original category distribution
    print("\nOriginal category distribution:")
    category_counts = df['category'].value_counts()
    print(category_counts)
    
    # Balance the training dataset - DOWNSAMPLE to the smallest category
    print("\nBalancing training dataset...")
    category_counts = df['category'].value_counts()
    print("Original category distribution:")
    print(category_counts)
    
    # Use custom balancing configuration if enabled
    if cfg.data.get('category_balancing', {}).get('enabled', False):
        min_samples = cfg.data.category_balancing.min_samples_per_category
        print(f"\n{'='*50}")
        print(f"USING CUSTOM CATEGORY BALANCING: {min_samples} samples per category")
        print(f"{'='*50}\n")
        
        # Find the smallest category count
        smallest_category_count = category_counts.min()
        
        # If min_samples is higher than some category counts, we can't balance
        if min_samples > smallest_category_count:
            print(f"Warning: Requested {min_samples} samples per category, but smallest category only has {smallest_category_count}")
            print(f"Will balance to {smallest_category_count} samples per category instead")
            minority_class_count = smallest_category_count
        else:
            # Use the specified minimum samples count
            print(f"Balancing all categories to {min_samples} samples")
            minority_class_count = min_samples
    else:
        # Find the minority class count (original behavior)
        minority_class_count = category_counts.min()
        minority_class = category_counts.idxmin()
        print(f"Minority class: '{minority_class}' with {minority_class_count} samples")
    
    # Create a balanced dataframe by downsampling all categories to match the target count
    balanced_df = pd.DataFrame()
    
    # For each category, take a random sample of size equal to the target count
    for category in category_counts.index:
        category_df = df[df['category'] == category]
        category_count = len(category_df)
        
        if category_count <= minority_class_count:
            # If this category has fewer samples than the target, use all of them
            downsampled_df = category_df
        else:
            # Downsample this category to match the target count
            downsampled_df = category_df.sample(minority_class_count, random_state=cfg.training.seed)
        
        balanced_df = pd.concat([balanced_df, downsampled_df])
    
    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=cfg.training.seed).reset_index(drop=True)
    
    # Update the dataframe for training
    df = balanced_df
    
    # Display new category distribution
    print("\nBalanced category distribution (downsampled):")
    print(df['category'].value_counts())
    print("="*50 + "\n")
    
    # Ensure a minimum number of samples per class in test set
    val_split = cfg.data.val_split
    min_test_samples_per_class = 2  # Minimum required for each class in the test set

    # Get number of classes
    num_classes = len(df['category'].unique())

    # Calculate minimum test size required for stratification
    min_test_size = min_test_samples_per_class * num_classes / len(df)

    # Use the larger of the configured val_split or the minimum required test size
    effective_test_size = max(val_split, min_test_size)

    # Print info for debugging
    print(f"Number of classes: {num_classes}")
    print(f"Total samples: {len(df)}")
    print(f"Configured validation split: {val_split:.2f} ({int(val_split * len(df))} samples)")
    print(f"Minimum required validation split: {min_test_size:.2f} ({min_test_samples_per_class * num_classes} samples)")
    print(f"Using effective validation split: {effective_test_size:.2f} ({int(effective_test_size * len(df))} samples)")

    # Perform the train-test split with the effective test size
    train_df, val_df = train_test_split(
        df,
        test_size=effective_test_size,
        stratify=df['category'],
        random_state=cfg.training.seed
    )

    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    # Check if test CSV exists at direct path or joined with base path
    test_csv_direct = to_absolute_path(cfg.data.test_csv_path)
    test_csv_joined = os.path.join(to_absolute_path(cfg.data.test_base_path), cfg.data.test_csv_path)

    print(f"Checking for test CSV at: {test_csv_direct}")
    print(f"Alternative path: {test_csv_joined}")
    print(f"Direct path exists: {os.path.exists(test_csv_direct)}")
    print(f"Joined path exists: {os.path.exists(test_csv_joined)}")

    if cfg.data.use_robot_data and (os.path.exists(test_csv_direct) or os.path.exists(test_csv_joined)):
        print("\n" + "="*50)
        print("USING ROBOT DATA AS TEST SET")
        print(f"Test dataset: {cfg.data.test_csv_path}")
        print("="*50 + "\n")
        
        # Load test dataset using the path that exists
        test_csv_path = test_csv_direct if os.path.exists(test_csv_direct) else test_csv_joined
        test_df = pd.read_csv(test_csv_path)
        print(f"\nLoaded test dataset with {len(test_df)} samples")
        
        # Print the first few rows to debug path issues
        print("\nFirst few rows of test dataset:")
        print(test_df.head())
        
        # Verify the file paths in the test dataset
        test_base_path = to_absolute_path(cfg.data.test_base_path)
        print(f"\nTest base path: {test_base_path}")
        
        # Check if the image and audio directories exist
        img_dir = os.path.join(test_base_path, 'images')
        audio_dir = os.path.join(test_base_path, 'audio')
        
        print(f"Image directory exists: {os.path.exists(img_dir)}")
        print(f"Audio directory exists: {os.path.exists(audio_dir)}")
        
        # List the directories in the test base path
        print(f"\nDirectories in test base path:")
        try:
            dirs = [d for d in os.listdir(test_base_path) if os.path.isdir(os.path.join(test_base_path, d))]
            print(dirs)
        except Exception as e:
            print(f"Error listing directories: {e}")
        
        # Filter out ambient samples if specified
        if cfg.data.get('exclude_ambient', False):
            ambient_count = len(test_df[test_df['category'] == 'ambient'])
            test_df = test_df[test_df['category'] != 'ambient']
            print(f"Removed {ambient_count} ambient samples from test dataset")
            print(f"Test dataset size reduced to {len(test_df)} samples")
        
        # Use subset of test data if specified
        if cfg.data.get('subset_test_data', False):
            print(f"\n{'='*50}")
            print(f"USING SUBSET OF TEST DATA: {cfg.data.test_subset_size*100:.1f}% of original test dataset")
            print(f"{'='*50}\n")
            
            # Stratified sampling to maintain class distribution
            test_df = test_df.groupby('category', group_keys=False).apply(
                lambda x: x.sample(frac=cfg.data.test_subset_size, random_state=cfg.training.seed)
            ).reset_index(drop=True)
            
            print(f"Test subset contains {len(test_df)} samples")
        
        # Balance the test dataset if enabled
        print("\nBalancing test dataset...")
        test_category_counts = test_df['category'].value_counts()
        print("Original test category distribution:")
        print(test_category_counts)
        
        # Use custom balancing configuration if enabled
        test_balance_enabled = cfg.data.get('category_balancing', {}).get('balance_test_set', True)
        if cfg.data.get('category_balancing', {}).get('enabled', False) and test_balance_enabled:
            min_samples = cfg.data.category_balancing.min_samples_per_category
            print(f"\n{'='*50}")
            print(f"USING CUSTOM CATEGORY BALANCING FOR TEST SET: {min_samples} samples per category")
            print(f"{'='*50}\n")
            
            # Find the smallest test category count
            smallest_test_category_count = test_category_counts.min()
            
            # If min_samples is higher than some category counts, we can't balance
            if min_samples > smallest_test_category_count:
                print(f"Warning: Requested {min_samples} samples per category, but smallest test category only has {smallest_test_category_count}")
                print(f"Will balance to {smallest_test_category_count} samples per category instead")
                test_minority_class_count = smallest_test_category_count
            else:
                # Use the specified minimum samples count
                print(f"Balancing all test categories to {min_samples} samples")
                test_minority_class_count = min_samples
        else:
            # Find the minority class count in test data (original behavior)
            test_minority_class_count = test_category_counts.min()
            test_minority_class = test_category_counts.idxmin()
            print(f"Test minority class: '{test_minority_class}' with {test_minority_class_count} samples")
        
        # Create a balanced test dataframe
        balanced_test_df = pd.DataFrame()
        
        # For each category, take a random sample of size equal to the target count
        for category in test_category_counts.index:
            category_df = test_df[test_df['category'] == category]
            category_count = len(category_df)
            
            if category_count <= test_minority_class_count:
                # If this category has fewer samples than the target, use all of them
                downsampled_df = category_df
            else:
                # Downsample this category to match the target count
                downsampled_df = category_df.sample(test_minority_class_count, random_state=cfg.training.seed)
            
            balanced_test_df = pd.concat([balanced_test_df, downsampled_df])
        
        # Shuffle the balanced test dataframe
        balanced_test_df = balanced_test_df.sample(frac=1, random_state=cfg.training.seed).reset_index(drop=True)
        
        # Update the test dataframe
        test_df = balanced_test_df
        
        # Display new test category distribution
        print("\nBalanced test category distribution (downsampled):")
        print(test_df['category'].value_counts())
        print("="*50 + "\n")
        
        # Create test dataset
        test_dataset = AudioVisualDataset(
            test_df, 
            base_path=to_absolute_path(cfg.data.test_base_path),
            img_size=cfg.data.img_size,
            use_binary_classification=cfg.data.get('use_binary_classification', False),
            augment=False,  # No augmentations for test data
            aug_config=cfg.data.get('augmentations', {}),
            do_norm=cfg.data.do_norm
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg.training.batch_size,
            shuffle=False, 
            num_workers=cfg.training.num_workers,
            collate_fn=custom_collate
        )
        
        print(f"Test samples: {len(test_dataset)}")
    else:
        print("No test set (robot data) available")
    
    # Create datasets with absolute paths for base_path
    train_dataset = AudioVisualDataset(
        train_df, 
        base_path=to_absolute_path(cfg.data.train_base_path),
        img_size=cfg.data.img_size,
        use_binary_classification=cfg.data.get('use_binary_classification', False),
        augment=True,  # Enable augmentations for training data
        aug_config=cfg.data.get('augmentations', {}),
        do_norm=cfg.data.do_norm
    )
    
    # For validation dataset, specify the correct base path if using a separate dataset
    val_dataset = AudioVisualDataset(
        val_df, 
        base_path=to_absolute_path(cfg.data.train_base_path),
        img_size=cfg.data.img_size,
        use_binary_classification=cfg.data.get('use_binary_classification', False),
        augment=False,  # No augmentations for validation data
        aug_config=cfg.data.get('augmentations', {}),
        do_norm=cfg.data.do_norm
    )
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with the custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size,
        shuffle=True, 
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size,
        shuffle=False, 
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )
    
    # Initialize model with modality flags, pretrained option, and audio model type
    model = MultiModalClassifier(
        num_classes=cfg.model.num_classes,
        use_images=cfg.model.use_images,
        use_audio=cfg.model.use_audio,
        pretrained=not cfg.model.from_scratch,
        audio_model=cfg.model.audio_model,
        use_dual_audio=cfg.model.use_dual_audio,
        fusion_type=cfg.model.fusion_type
    )
    
    # Train model
    history, val_preds, val_labels = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device
    )
    
    # Get class names
    category_classes = get_class_names(cfg)
    
    # Plot confusion matrix
    plot_confusion_matrix(val_labels, val_preds, category_classes, cfg)
    
    # Generate classification report
    report = classification_report(val_labels, val_preds, target_names=category_classes, 
                                  output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=category_classes, 
                               zero_division=0))
    
    # Log classification metrics to wandb
    log_classification_metrics(report, category_classes)
    
    # Add binary classification metrics (contact vs. no-contact) regardless of training mode
    if not cfg.data.get('use_binary_classification', False):
        print("\nAdding binary classification metrics (contact vs. no-contact)...")
        
        # Convert multi-class labels to binary (contact vs. no-contact)
        binary_val_labels, binary_val_preds = convert_to_binary_classification(val_labels, val_preds)
        
        # Plot binary confusion matrix
        binary_classes = ['contact', 'no-contact']
        plot_confusion_matrix(binary_val_labels, binary_val_preds, binary_classes, cfg, suffix="_binary")
        
        # Generate binary classification report
        binary_report = classification_report(binary_val_labels, binary_val_preds, 
                                             target_names=binary_classes, output_dict=True)
        print("\nBinary Classification Report (Contact vs. No-Contact):")
        print(classification_report(binary_val_labels, binary_val_preds, target_names=binary_classes))
        
        # Log binary metrics to wandb
        log_classification_metrics(binary_report, binary_classes, binary_mode=True)
    
    # Save misclassified samples
    # print("\nSaving misclassified samples for analysis...")
    # save_misclassified_samples(
    #     val_loader=val_loader,
    #     model=model,
    #     val_preds=val_preds,
    #     val_labels=val_labels,
    #     cfg=cfg,
    #     device=device
    # )
    
    # Evaluate on test set (robot data) if available
    if test_loader is not None:
        print("\n" + "="*50)
        print("EVALUATING ON TEST SET (ROBOT DATA)")
        print("="*50 + "\n")
        
        # Load the best model for evaluation
        best_model_path = os.path.join(cfg.logging.output_dir, 'best_contact_sound_model.pth')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("Warning: Best model checkpoint not found. Using current model state.")
        
        # Evaluate on test set
        test_preds, test_labels, test_accuracy = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device
        )
        
        print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
        
        # Generate test set classification report
        test_report = classification_report(test_labels, test_preds, target_names=category_classes, output_dict=True)
        print("\nTest Set Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=category_classes))
        
        # Calculate and log variance metrics for robot evaluation
        class_accuracies = {}
        for cls_idx, cls_name in enumerate(category_classes):
            # Get indices for this class
            cls_indices = [i for i, label in enumerate(test_labels) if label == cls_idx]
            if cls_indices:
                # Calculate accuracy for this class
                cls_correct = sum(1 for i in cls_indices if test_preds[i] == test_labels[i])
                cls_accuracy = cls_correct / len(cls_indices)
                class_accuracies[cls_name] = cls_accuracy
        
        # Calculate variance and log metrics
        if len(class_accuracies) > 1:
            accuracy_values = list(class_accuracies.values())
            accuracy_variance = np.var(accuracy_values)
            accuracy_std_dev = np.std(accuracy_values)
            
            print(f"\nRobot Evaluation Metrics:")
            print(f"Class accuracies: {class_accuracies}")
            print(f"Accuracy variance: {accuracy_variance:.4f}")
            print(f"Accuracy standard deviation: {accuracy_std_dev:.4f}")
            
            # Log to wandb if enabled
            if cfg.wandb.enabled:
                import wandb
                wandb.log({
                    "robot_eval/accuracy_variance": accuracy_variance,
                    "robot_eval/accuracy_std_dev": accuracy_std_dev
                })
                # Log individual class accuracies
                for cls_name, cls_acc in class_accuracies.items():
                    wandb.log({f"robot_eval/class_accuracy_{cls_name}": cls_acc})
        
        # Plot test set confusion matrix
        plot_confusion_matrix(test_labels, test_preds, category_classes, cfg, suffix="_test")
        
        # Log test metrics to wandb
        log_classification_metrics(test_report, category_classes, prefix="test_")
        
        # If not using binary classification, add binary metrics for test set too
        if not cfg.data.get('use_binary_classification', False):
            print("\nAdding binary classification metrics for test set (contact vs. no-contact)...")
            
            # Convert multi-class labels to binary (contact vs. no-contact)
            binary_test_labels, binary_test_preds = convert_to_binary_classification(test_labels, test_preds)
            
            # Plot binary confusion matrix for test set
            binary_classes = ['contact', 'no-contact']
            plot_confusion_matrix(binary_test_labels, binary_test_preds, binary_classes, cfg, suffix="_test_binary")
            
            # Generate binary classification report for test set
            binary_test_report = classification_report(binary_test_labels, binary_test_preds, 
                                                 target_names=binary_classes, output_dict=True)
            print("\nBinary Test Set Classification Report (Contact vs. No-Contact):")
            print(classification_report(binary_test_labels, binary_test_preds, target_names=binary_classes))
            
            # Log binary test metrics to wandb
            log_classification_metrics(binary_test_report, binary_classes, binary_mode=True, prefix="test_")
        
        # Save misclassified samples from test set
        # print("\nSaving misclassified samples from test set for analysis...")
        # save_misclassified_samples(
        #     val_loader=test_loader,  # Using test_loader instead of val_loader
        #     model=model,
        #     val_preds=test_preds,
        #     val_labels=test_labels,
        #     cfg=cfg,
        #     device=device
        # )

    # Finish wandb run after all logging is complete
    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main() 