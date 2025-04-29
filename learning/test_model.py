import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the directory containing main_train_2.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Then use the correct import
from main_train_2 import MultiModalClassifier, AudioVisualDataset, custom_collate, evaluate_model

# =============================================
# HARD-CODED CONFIGURATION - MODIFY THESE VALUES
# =============================================
MODEL_PATH = '/home/dorry/Desktop/research/outputs/2025-04-19/13-47-33/outputs/2025-04-19/plz-work/best_contact_sound_model.pth'  # Path to your .pth model file
TEST_CSV = '/home/dorry/Desktop/research/audio_visual_dataset_robo/dataset.csv'                   # Path to your test CSV file
DATA_PATH = '/home/dorry/Desktop/research/audio_visual_dataset_robo'                           # Base path to your dataset
OUTPUT_DIR = './eval_results'                  # Directory to save results
BATCH_SIZE = 8                                 # Batch size for evaluation
NUM_WORKERS = 4                                # Number of worker threads for data loading
USE_BINARY = False                             # Use binary classification (contact vs no-contact)
IMG_SIZE = 224                                 # Size of input images
NUM_CLASSES = 4                                # Number of classes
USE_IMAGES = True                              # Use image modality
USE_AUDIO = True                               # Use audio modality
AUDIO_MODEL = 'ast'                            # Audio model type ('ast' or 'clap')
USE_CPU = False                                # Force CPU usage even if CUDA is available
USE_MFCC = True                                # Whether to use MFCC features if available
VIDEO_MODE = True  # Set to True if your dataset uses video files instead of separate image/audio files
# =============================================


def plot_confusion_matrix(y_true, y_pred, classes, output_dir, suffix=""):
    """
    Generate and plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        output_dir: Directory to save the plot
        suffix: Optional suffix for the output filename
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with appropriate size based on number of classes
    plt.figure(figsize=(10 + len(classes)//3, 8 + len(classes)//3))
    
    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save locally
    cm_path = os.path.join(output_dir, f'confusion_matrix{suffix}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    plt.close()  # Close the figure to free memory


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset to evaluate on
        device: Device to run evaluation on
        
    Returns:
        predictions, true_labels, accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Only pass the modalities that are enabled in the model
            images = batch['image'].to(device) if model.use_images else None
            audio = batch['audio'].to(device) if model.use_audio else None
            mfcc = batch['mfcc'].to(device) if 'mfcc' in batch and hasattr(model, 'use_mfcc') and model.use_mfcc else None
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(x_img=images, x_audio=audio, x_mfcc=mfcc)
            _, predicted = torch.max(outputs, 1)
            
            # Statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0
    return all_preds, all_labels, accuracy


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set device
    if USE_CPU:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define class names based on binary flag
    if USE_BINARY:
        class_names = ['contact', 'no-contact']
        num_classes = 2
    else:
        class_names = ['leaf', 'twig', 'trunk', 'ambient']
        num_classes = NUM_CLASSES
    
    print(f"Class names: {class_names}")
    
    # Load the model
    model = MultiModalClassifier(
        num_classes=num_classes,
        use_images=USE_IMAGES,
        use_audio=USE_AUDIO,
        pretrained=False,  # Not relevant for evaluation
        audio_model=AUDIO_MODEL,
        use_dual_audio=False,  # Set to False by default
        fusion_type="transformer",  # Default fusion type
        use_mfcc=USE_MFCC  # Keep the MFCC flag
    )
    
    # Load model weights
    print(f"Loading model from {MODEL_PATH}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Handle loading model to CPU if needed
    try:
        if device.type == 'cpu':
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with different settings...")
        try:
            # Try loading with strict=False
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded with strict=False")
        except Exception as e2:
            print(f"Still failed with error: {e2}")
            print("Check model path and structure")
            return
    
    model.to(device)
    
    # Check if test CSV file exists
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV file not found: {TEST_CSV}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")
    
    # Load test dataset
    print(f"Loading test data from {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df)} test samples")
    
    # Display first few rows to verify
    print("\nFirst few rows of the test data:")
    print(test_df.head())
    
    # Verify that required columns exist
    required_columns = ['category', 'image_file', 'audio_file']
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in test data: {missing_columns}")
        print("Available columns: ", test_df.columns.tolist())
    
    # Create test dataset
    try:
        test_dataset = AudioVisualDataset(
            test_df,
            base_path=DATA_PATH,
            img_size=IMG_SIZE,
            spec_size=(224, 224),
            use_binary_classification=USE_BINARY
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate
        )
    except Exception as e:
        print(f"Error creating dataset/dataloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate model
    print("Starting evaluation...")
    try:
        test_preds, test_labels, test_accuracy = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device
        )
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=class_names))
        
        # Plot confusion matrix
        plot_confusion_matrix(test_labels, test_preds, class_names, OUTPUT_DIR)
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'true_label': [class_names[l] for l in test_labels],
            'predicted_label': [class_names[p] for p in test_preds],
            'correct': [1 if p == l else 0 for p, l in zip(test_preds, test_labels)]
        })
        
        # Add file paths from the test dataframe if available
        if 'image_file' in test_df.columns:
            results_df['image_file'] = test_df['image_file'].values
        if 'audio_file' in test_df.columns:
            results_df['audio_file'] = test_df['audio_file'].values
        if 'video' in test_df.columns:
            results_df['video_file'] = test_df['video'].values
        
        # Save results
        results_path = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # If not using binary classification originally, also report binary metrics
        if not USE_BINARY and len(class_names) > 2:
            print("\nCalculating binary metrics (contact vs. no-contact)...")
            # Convert multiclass to binary: 'ambient' -> 'no-contact', everything else -> 'contact'
            binary_classes = ['contact', 'no-contact']
            binary_test_labels = np.array([1 if class_names[label] == 'ambient' else 0 for label in test_labels])
            binary_test_preds = np.array([1 if class_names[pred] == 'ambient' else 0 for pred in test_preds])
            
            # Calculate binary accuracy
            binary_accuracy = (binary_test_labels == binary_test_preds).mean()
            print(f"Binary Accuracy: {binary_accuracy:.4f}")
            
            # Print binary classification report
            print("\nBinary Classification Report:")
            print(classification_report(binary_test_labels, binary_test_preds, target_names=binary_classes))
            
            # Plot binary confusion matrix
            plot_confusion_matrix(binary_test_labels, binary_test_preds, binary_classes, OUTPUT_DIR, suffix="_binary")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
