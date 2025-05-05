import os
import pandas as pd
import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
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
    update_num_classes
)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """Main function to run the training pipeline with 5-fold cross-validation"""
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
    
    # Get category classes
    category_classes = get_class_names(cfg)
    
    # Prepare for 5-fold cross-validation
    print("\n" + "="*50)
    print("STARTING 5-FOLD CROSS-VALIDATION")
    print("="*50 + "\n")
    
    # Initialize arrays to store results
    all_f1_scores = []
    all_f1_weighted = []
    all_f1_macro = []
    fold_reports = []
    
    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.training.seed)
    
    # Get the labels for stratification
    labels = df['category'].map(lambda x: category_classes.index(x))
    
    # For each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{n_splits}")
        print(f"{'='*50}\n")
        
        # Split the data
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        
        # Create datasets
        train_dataset = AudioVisualDataset(
            train_df, 
            base_path=to_absolute_path(cfg.data.train_base_path),
            img_size=cfg.data.img_size,
            use_binary_classification=cfg.data.get('use_binary_classification', False),
            augment=True,  # Enable augmentations for training data
            aug_config=cfg.data.get('augmentations', {}),
            do_norm=cfg.data.do_norm
        )
        
        val_dataset = AudioVisualDataset(
            val_df, 
            base_path=to_absolute_path(cfg.data.train_base_path),
            img_size=cfg.data.img_size,
            use_binary_classification=cfg.data.get('use_binary_classification', False),
            augment=False,  # No augmentations for validation data
            aug_config=cfg.data.get('augmentations', {}),
            do_norm=cfg.data.do_norm
        )
        
        # Create data loaders
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
        
        # Initialize a fresh model for this fold
        # Note: Using 'pretrained' parameter which is deprecated but still functional
        # The deprecation warning is expected and can be ignored
        model = MultiModalClassifier(
            num_classes=cfg.model.num_classes,
            use_images=cfg.model.use_images,
            use_audio=cfg.model.use_audio,
            pretrained=not cfg.model.from_scratch,
            audio_model=cfg.model.audio_model,
            use_dual_audio=cfg.model.use_dual_audio,
            fusion_type=cfg.model.fusion_type
        )

        # Save initial model architecture for consistent loading during testing
        if fold == 0:
            # Save model architecture (without weights) for reference
            initial_model_path = os.path.join(cfg.logging.output_dir, 'model_architecture.pt')
            # Save a copy of the model architecture definition
            torch.save({
                'num_classes': cfg.model.num_classes,
                'use_images': cfg.model.use_images,
                'use_audio': cfg.model.use_audio,
                'pretrained': not cfg.model.from_scratch,
                'audio_model': cfg.model.audio_model,
                'use_dual_audio': cfg.model.use_dual_audio,
                'fusion_type': cfg.model.fusion_type
            }, initial_model_path)
            print(f"Saved model architecture configuration to {initial_model_path}")

        # Train model on this fold
        history, val_preds, val_labels = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device
        )
        
        # Save model for this fold
        fold_model_path = os.path.join(cfg.logging.output_dir, f'fold_{fold+1}_model.pth')
        torch.save(model.state_dict(), fold_model_path)
        print(f"Saved fold {fold+1} model weights to {fold_model_path}")
        
        # Evaluate the model on validation data
        print(f"\nEvaluating fold {fold+1} model on validation data...")
        val_preds_eval, val_labels_eval, val_accuracy = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device
        )
        print(f"Fold {fold+1} Validation Accuracy: {val_accuracy:.4f}")
        
        # Plot confusion matrix for this fold
        plot_confusion_matrix(val_labels, val_preds, category_classes, cfg, suffix=f"_fold_{fold+1}")
        
        # Generate classification report
        report = classification_report(val_labels, val_preds, target_names=category_classes,
                                     output_dict=True, zero_division=0)
        fold_reports.append(report)
        
        print(f"\nFold {fold+1} Classification Report:")
        print(classification_report(val_labels, val_preds, target_names=category_classes,
                                   zero_division=0))
        
        # Calculate per-class F1 scores
        fold_f1_scores = []
        for class_idx, class_name in enumerate(category_classes):
            # Calculate class-specific F1 score
            class_f1 = report[class_name]['f1-score']
            fold_f1_scores.append(class_f1)
            print(f"F1 score for class '{class_name}': {class_f1:.4f}")
        
        # Calculate macro and weighted F1 scores
        f1_macro = report['macro avg']['f1-score']
        f1_weighted = report['weighted avg']['f1-score']
        all_f1_macro.append(f1_macro)
        all_f1_weighted.append(f1_weighted)
        all_f1_scores.append(fold_f1_scores)
        
        print(f"Macro-averaged F1: {f1_macro:.4f}")
        print(f"Weighted-averaged F1: {f1_weighted:.4f}")
        
        # Log to wandb if enabled
        if cfg.wandb.enabled:
            import wandb
            wandb.log({
                f"fold_{fold+1}/f1_macro": f1_macro,
                f"fold_{fold+1}/f1_weighted": f1_weighted,
                f"fold_{fold+1}/val_accuracy": val_accuracy
            })
            
            # Log per-class metrics
            for i, class_name in enumerate(category_classes):
                wandb.log({
                    f"fold_{fold+1}/f1_{class_name}": fold_f1_scores[i]
                })
    
    # Calculate cross-validation summary statistics
    all_f1_scores = np.array(all_f1_scores)
    mean_f1_scores = np.mean(all_f1_scores, axis=0)
    std_f1_scores = np.std(all_f1_scores, axis=0)
    var_f1_scores = np.var(all_f1_scores, axis=0)
    
    mean_f1_macro = np.mean(all_f1_macro)
    std_f1_macro = np.std(all_f1_macro)
    var_f1_macro = np.var(all_f1_macro)
    
    mean_f1_weighted = np.mean(all_f1_weighted)
    std_f1_weighted = np.std(all_f1_weighted)
    var_f1_weighted = np.var(all_f1_weighted)
    
    # Print cross-validation summary
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50 + "\n")
    
    print(f"Mean Macro-averaged F1 Score: {mean_f1_macro:.4f}")
    print(f"Standard Deviation of Macro-averaged F1 Scores: {std_f1_macro:.4f}")
    print(f"Variance of Macro-averaged F1 Scores: {var_f1_macro:.4f}")
    print()
    
    print(f"Mean Weighted-averaged F1 Score: {mean_f1_weighted:.4f}")
    print(f"Standard Deviation of Weighted-averaged F1 Scores: {std_f1_weighted:.4f}")
    print(f"Variance of Weighted-averaged F1 Scores: {var_f1_weighted:.4f}")
    print()
    
    print("Per-class F1 Score Summary:")
    for i, class_name in enumerate(category_classes):
        print(f"  {class_name}:")
        print(f"    Mean F1 Score: {mean_f1_scores[i]:.4f}")
        print(f"    Standard Deviation: {std_f1_scores[i]:.4f}")
        print(f"    Variance: {var_f1_scores[i]:.4f}")
    
    # Log cross-validation summary to wandb
    if cfg.wandb.enabled:
        import wandb
        wandb.log({
            "cv_summary/f1_macro_mean": mean_f1_macro,
            "cv_summary/f1_macro_std": std_f1_macro,
            "cv_summary/f1_macro_var": var_f1_macro,
            "cv_summary/f1_weighted_mean": mean_f1_weighted,
            "cv_summary/f1_weighted_std": std_f1_weighted,
            "cv_summary/f1_weighted_var": var_f1_weighted
        })
        
        # Log per-class summary metrics
        for i, class_name in enumerate(category_classes):
            wandb.log({
                f"cv_summary/f1_{class_name}_mean": mean_f1_scores[i],
                f"cv_summary/f1_{class_name}_std": std_f1_scores[i],
                f"cv_summary/f1_{class_name}_var": var_f1_scores[i]
            })
    
    # Evaluate on test set (robot data) if available
    test_csv_direct = to_absolute_path(cfg.data.test_csv_path)
    test_csv_joined = os.path.join(to_absolute_path(cfg.data.test_base_path), cfg.data.test_csv_path)
    
    if cfg.data.use_robot_data and (os.path.exists(test_csv_direct) or os.path.exists(test_csv_joined)):
        print("\n" + "="*50)
        print("EVALUATING ON TEST SET (ROBOT DATA)")
        print("="*50 + "\n")
        
        # Load test dataset using the path that exists
        test_csv_path = test_csv_direct if os.path.exists(test_csv_direct) else test_csv_joined
        test_df = pd.read_csv(test_csv_path)
        print(f"\nLoaded test dataset with {len(test_df)} samples")
        
        # Process test dataset as in the original code...
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
            # Check if there's a specific value for test set balancing
            # Handle the case where test_min_samples_per_category might not exist
            try:
                min_samples = cfg.data.category_balancing.test_min_samples_per_category
                print(f"\n{'='*50}")
                print(f"USING TEST-SPECIFIC CATEGORY BALANCING: {min_samples} samples per category")
                print(f"{'='*50}\n")
            except (AttributeError, KeyError):
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
        
        # Test the model from each fold on the test dataset and report average results
        test_fold_f1_macro = []
        test_fold_f1_weighted = []
        test_fold_class_f1 = [[] for _ in range(len(category_classes))]
        
        for fold in range(n_splits):
            print(f"\nEvaluating fold {fold+1} model on test set...")
            # Load the model for this fold
            fold_model_path = os.path.join(cfg.logging.output_dir, f'fold_{fold+1}_model.pth')
            initial_model_path = os.path.join(cfg.logging.output_dir, 'model_architecture.pt')
            
            if os.path.exists(fold_model_path) and os.path.exists(initial_model_path):
                try:
                    # Load the model architecture configuration
                    print(f"Loading model architecture configuration from {initial_model_path}")
                    model_config = torch.load(initial_model_path)
                    
                    # Reconstruct a model with the same architecture
                    # Note: Using 'pretrained' parameter which is deprecated but still functional
                    # The deprecation warning is expected and can be ignored
                    model = MultiModalClassifier(
                        num_classes=model_config['num_classes'],
                        use_images=model_config['use_images'],
                        use_audio=model_config['use_audio'],
                        pretrained=model_config['pretrained'],
                        audio_model=model_config['audio_model'],
                        use_dual_audio=model_config['use_dual_audio'],
                        fusion_type=model_config['fusion_type']
                    )
                    model.to(device)
                    
                    # Now load the trained weights
                    print(f"Loading trained weights from {fold_model_path}")
                    model.load_state_dict(torch.load(fold_model_path))
                    print(f"Successfully loaded fold {fold+1} model")
                    
                    # Evaluate on test set
                    test_preds, test_labels, test_accuracy = evaluate_model(
                        model=model,
                        data_loader=test_loader,
                        device=device
                    )
                    
                    print(f"Fold {fold+1} Test Accuracy: {test_accuracy:.4f}")
                    
                    # Generate test classification report
                    test_report = classification_report(test_labels, test_preds, 
                                                      target_names=category_classes, 
                                                      output_dict=True)
                    
                    # Store F1 scores
                    test_fold_f1_macro.append(test_report['macro avg']['f1-score'])
                    test_fold_f1_weighted.append(test_report['weighted avg']['f1-score'])
                    
                    # Store per-class F1 scores
                    for i, class_name in enumerate(category_classes):
                        if class_name in test_report:
                            test_fold_class_f1[i].append(test_report[class_name]['f1-score'])
                    
                    # Log to wandb if enabled
                    if cfg.wandb.enabled:
                        import wandb
                        wandb.log({
                            f"test_fold_{fold+1}/accuracy": test_accuracy,
                            f"test_fold_{fold+1}/f1_macro": test_report['macro avg']['f1-score'],
                            f"test_fold_{fold+1}/f1_weighted": test_report['weighted avg']['f1-score']
                        })
                        
                        # Log per-class metrics
                        for i, class_name in enumerate(category_classes):
                            if class_name in test_report:
                                wandb.log({
                                    f"test_fold_{fold+1}/f1_{class_name}": test_report[class_name]['f1-score']
                                })
                except Exception as e:
                    print(f"Error loading or evaluating fold {fold+1} model: {str(e)}")
            else:
                if not os.path.exists(initial_model_path):
                    print(f"Warning: Initial model architecture not found at {initial_model_path}")
                if not os.path.exists(fold_model_path):
                    print(f"Warning: Model weights not found for fold {fold+1} at {fold_model_path}")
        
        # Calculate and report test set summary statistics
        if test_fold_f1_macro:
            print("\n" + "="*50)
            print("TEST SET EVALUATION SUMMARY ACROSS ALL FOLDS")
            print("="*50 + "\n")
            
            test_mean_f1_macro = np.mean(test_fold_f1_macro)
            test_std_f1_macro = np.std(test_fold_f1_macro)
            test_var_f1_macro = np.var(test_fold_f1_macro)
            
            test_mean_f1_weighted = np.mean(test_fold_f1_weighted)
            test_std_f1_weighted = np.std(test_fold_f1_weighted)
            test_var_f1_weighted = np.var(test_fold_f1_weighted)
            
            print(f"Test Mean Macro-averaged F1 Score: {test_mean_f1_macro:.4f}")
            print(f"Test Standard Deviation of Macro-averaged F1 Scores: {test_std_f1_macro:.4f}")
            print(f"Test Variance of Macro-averaged F1 Scores: {test_var_f1_macro:.4f}")
            print()
            
            print(f"Test Mean Weighted-averaged F1 Score: {test_mean_f1_weighted:.4f}")
            print(f"Test Standard Deviation of Weighted-averaged F1 Scores: {test_std_f1_weighted:.4f}")
            print(f"Test Variance of Weighted-averaged F1 Scores: {test_var_f1_weighted:.4f}")
            print()
            
            print("Test Per-class F1 Score Summary:")
            for i, class_name in enumerate(category_classes):
                if test_fold_class_f1[i]:
                    class_mean = np.mean(test_fold_class_f1[i])
                    class_std = np.std(test_fold_class_f1[i])
                    class_var = np.var(test_fold_class_f1[i])
                    
                    print(f"  {class_name}:")
                    print(f"    Mean F1 Score: {class_mean:.4f}")
                    print(f"    Standard Deviation: {class_std:.4f}")
                    print(f"    Variance: {class_var:.4f}")
            
            # Log test summary to wandb
            if cfg.wandb.enabled:
                import wandb
                wandb.log({
                    "test_summary/f1_macro_mean": test_mean_f1_macro,
                    "test_summary/f1_macro_std": test_std_f1_macro,
                    "test_summary/f1_macro_var": test_var_f1_macro,
                    "test_summary/f1_weighted_mean": test_mean_f1_weighted,
                    "test_summary/f1_weighted_std": test_std_f1_weighted,
                    "test_summary/f1_weighted_var": test_var_f1_weighted
                })
                
                # Log per-class test summary metrics
                for i, class_name in enumerate(category_classes):
                    if test_fold_class_f1[i]:
                        wandb.log({
                            f"test_summary/f1_{class_name}_mean": np.mean(test_fold_class_f1[i]),
                            f"test_summary/f1_{class_name}_std": np.std(test_fold_class_f1[i]),
                            f"test_summary/f1_{class_name}_var": np.var(test_fold_class_f1[i])
                        })
    else:
        print("No test set (robot data) available")
    
    # Finish wandb run after all logging is complete
    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main() 