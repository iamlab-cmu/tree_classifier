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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_train_2 import (
    MultiModalClassifier,
    AudioVisualDataset,
    custom_collate,
    evaluate_model,
)

MODEL_PATH = "/home/dorry/Desktop/research/outputs/2025-04-19/13-47-33/outputs/2025-04-19/plz-work/best_contact_sound_model.pth"  # Path to your .pth model file
TEST_CSV = "/home/dorry/Desktop/research/audio_visual_dataset_robo/dataset.csv"  # Path to your test CSV file
DATA_PATH = "/home/dorry/Desktop/research/audio_visual_dataset_robo"  # Base path to your dataset
OUTPUT_DIR = "./eval_results"  # Directory to save results
BATCH_SIZE = 8  # Batch size for evaluation
NUM_WORKERS = 4  # Number of worker threads for data loading
USE_BINARY = False  # Use binary classification (contact vs no-contact)
IMG_SIZE = 224  # Size of input images
NUM_CLASSES = 4  # Number of classes
USE_IMAGES = True  # Use image modality
USE_AUDIO = True  # Use audio modality
AUDIO_MODEL = "ast"  # Audio model type ('ast' or 'clap')
USE_CPU = False  # Force CPU usage even if CUDA is available
USE_MFCC = True  # Whether to use MFCC features if available
VIDEO_MODE = True  # Set to True if your dataset uses video files instead of separate image/audio files


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
    cm = confusion_matrix(y_true, y_pred)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10 + len(classes) // 3, 8 + len(classes) // 3))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"confusion_matrix{suffix}.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    plt.close()  # Close the figure to free memory


def evaluate_model(model, data_loader, device="cuda"):
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
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch["image"].to(device) if model.use_images else None
            audio = batch["audio"].to(device) if model.use_audio else None
            mfcc = (
                batch["mfcc"].to(device)
                if "mfcc" in batch and hasattr(model, "use_mfcc") and model.use_mfcc
                else None
            )
            labels = batch["label"].to(device)

            outputs = model(x_img=images, x_audio=audio, x_mfcc=mfcc)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0
    return all_preds, all_labels, accuracy


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if USE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if USE_BINARY:
        class_names = ["contact", "no-contact"]
        num_classes = 2
    else:
        class_names = ["leaf", "twig", "trunk", "ambient"]
        num_classes = NUM_CLASSES

    print(f"Class names: {class_names}")

    model = MultiModalClassifier(
        num_classes=num_classes,
        use_images=USE_IMAGES,
        use_audio=USE_AUDIO,
        pretrained=False,  # Not relevant for evaluation
        audio_model=AUDIO_MODEL,
        use_dual_audio=False,  # Set to False by default
        fusion_type="transformer",  # Default fusion type
        use_mfcc=USE_MFCC,  # Keep the MFCC flag
    )

    print(f"Loading model from {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        if device.type == "cpu":
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with different settings...")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded with strict=False")
        except Exception as e2:
            print(f"Still failed with error: {e2}")
            print("Check model path and structure")
            return

    model.to(device)

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV file not found: {TEST_CSV}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")

    print(f"Loading test data from {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df)} test samples")

    print("\nFirst few rows of the test data:")
    print(test_df.head())

    required_columns = ["category", "image_file", "audio_file"]
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in test data: {missing_columns}")
        print("Available columns: ", test_df.columns.tolist())

    try:
        test_dataset = AudioVisualDataset(
            test_df,
            base_path=DATA_PATH,
            img_size=IMG_SIZE,
            spec_size=(224, 224),
            use_binary_classification=USE_BINARY,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=custom_collate,
        )
    except Exception as e:
        print(f"Error creating dataset/dataloader: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Starting evaluation...")
    try:
        test_preds, test_labels, test_accuracy = evaluate_model(
            model=model, data_loader=test_loader, device=device
        )

        print(f"\nTest Accuracy: {test_accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, target_names=class_names))

        plot_confusion_matrix(test_labels, test_preds, class_names, OUTPUT_DIR)

        results_df = pd.DataFrame(
            {
                "true_label": [class_names[l] for l in test_labels],
                "predicted_label": [class_names[p] for p in test_preds],
                "correct": [
                    1 if p == l else 0 for p, l in zip(test_preds, test_labels)
                ],
            }
        )

        if "image_file" in test_df.columns:
            results_df["image_file"] = test_df["image_file"].values
        if "audio_file" in test_df.columns:
            results_df["audio_file"] = test_df["audio_file"].values
        if "video" in test_df.columns:
            results_df["video_file"] = test_df["video"].values

        results_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

        if not USE_BINARY and len(class_names) > 2:
            print("\nCalculating binary metrics (contact vs. no-contact)...")
            binary_classes = ["contact", "no-contact"]
            binary_test_labels = np.array(
                [1 if class_names[label] == "ambient" else 0 for label in test_labels]
            )
            binary_test_preds = np.array(
                [1 if class_names[pred] == "ambient" else 0 for pred in test_preds]
            )

            binary_accuracy = (binary_test_labels == binary_test_preds).mean()
            print(f"Binary Accuracy: {binary_accuracy:.4f}")

            print("\nBinary Classification Report:")
            print(
                classification_report(
                    binary_test_labels, binary_test_preds, target_names=binary_classes
                )
            )

            plot_confusion_matrix(
                binary_test_labels,
                binary_test_preds,
                binary_classes,
                OUTPUT_DIR,
                suffix="_binary",
            )

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
