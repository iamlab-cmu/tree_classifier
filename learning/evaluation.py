import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent plots from displaying


def plot_confusion_matrix(y_true, y_pred, classes, cfg, suffix=""):
    """
    Generate and plot a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        cfg: Configuration object
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

    cm_path = os.path.join(cfg.logging.output_dir, f"confusion_matrix{suffix}.png")
    plt.savefig(cm_path)

    if cfg.wandb.enabled:
        wandb.log({f"confusion_matrix{suffix}": wandb.Image(cm_path)})

    plt.close()  # Close the figure to free memory


def save_misclassified_samples(
    val_loader, model, val_preds, val_labels, cfg, device="cuda"
):
    """
    Save misclassified samples to a folder for analysis.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    missed_base_dir = os.path.join(os.getcwd(), "missed")
    os.makedirs(missed_base_dir, exist_ok=True)

    misclassified_dir = os.path.join(missed_base_dir, timestamp)
    os.makedirs(misclassified_dir, exist_ok=True)

    print(f"\nSaving misclassified samples to: {misclassified_dir}")
    print(f"Current working directory: {os.getcwd()}")

    if cfg.data.get("use_binary_classification", False):
        class_names = ["contact", "no-contact"]
    else:
        class_names = cfg.data.classes

    print(f"Using class names: {class_names}")

    misclassified_indices = [
        i for i, (pred, label) in enumerate(zip(val_preds, val_labels)) if pred != label
    ]

    print(
        f"\nFound {len(misclassified_indices)} misclassified samples out of {len(val_labels)} validation samples"
    )
    if len(misclassified_indices) == 0:
        print("No misclassified samples to save.")
        return

    csv_path = os.path.join(misclassified_dir, "misclassified_samples.csv")
    with open(csv_path, "w") as f:
        f.write(
            "sample_idx,image_file,audio_file,true_label,predicted_label,saved_image_path,saved_spectrogram_path\n"
        )

    all_samples = []
    for batch in val_loader:
        for i in range(len(batch["label"])):
            sample = {
                "image": batch["image"][i],
                "audio": batch["audio"][i],
                "label": batch["label"][i].item(),
                "category": batch["category"][i],
            }
            all_samples.append(sample)

    created_folders = []
    for true_class in class_names:
        for pred_class in class_names:
            if true_class != pred_class:
                folder_path = os.path.join(
                    misclassified_dir, f"{true_class}_as_{pred_class}"
                )
                os.makedirs(folder_path, exist_ok=True)
                created_folders.append(folder_path)

    print(
        f"Created {len(created_folders)} folders for different misclassification types"
    )

    successful_saves = 0
    for idx in misclassified_indices:
        if idx >= len(all_samples):
            print(
                f"Warning: Index {idx} out of range for all_samples (length: {len(all_samples)})"
            )
            continue

        sample = all_samples[idx]
        true_label = int(val_labels[idx])
        pred_label = int(val_preds[idx])

        true_class = class_names[true_label]
        pred_class = class_names[pred_label]

        print(f"\nProcessing sample {idx}: true={true_class}, predicted={pred_class}")

        sample_id = f"{idx}"

        folder_path = os.path.join(misclassified_dir, f"{true_class}_as_{pred_class}")
        print(f"Saving to folder: {folder_path}")

        img_tensor = sample["image"]
        img_np = img_tensor.permute(1, 2, 0).numpy()  # CHW to HWC

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        true_text = f"True: {true_class}"
        text_size = cv2.getTextSize(true_text, font, font_scale, thickness)[0]
        text_x = 10
        text_y = 30
        cv2.putText(
            img_bgr,
            true_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        pred_text = f"Pred: {pred_class}"
        text_y += text_size[1] + 10
        cv2.putText(
            img_bgr,
            pred_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
        )

        img_path = os.path.join(folder_path, f"{sample_id}_image.png")
        img_success = cv2.imwrite(img_path, img_bgr)
        if img_success:
            print(f"Successfully saved image to: {img_path}")
        else:
            print(f"Failed to save image to: {img_path}")

        audio_tensor = sample["audio"]
        audio_np = audio_tensor.squeeze().numpy()  # Remove batch dimension

        non_zero_cols = np.where(np.sum(audio_np, axis=0) > 0.1)[0]
        if len(non_zero_cols) > 0:
            start_col = max(0, non_zero_cols[0] - 10)
            end_col = min(audio_np.shape[1], non_zero_cols[-1] + 10)

            audio_np_trimmed = audio_np[:, start_col:end_col]

            title = f"True: {true_class}, Predicted: {pred_class}\nContent duration: {(end_col - start_col) / audio_np.shape[1]:.2f} of total"

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.imshow(audio_np, aspect="auto", origin="lower")
            plt.title("Full Spectrogram")
            plt.colorbar(format="%+2.0f dB")

            plt.subplot(2, 1, 2)
            plt.imshow(audio_np_trimmed, aspect="auto", origin="lower")
            plt.title("Focused on Audio Content")
            plt.colorbar(format="%+2.0f dB")

            plt.suptitle(title)
            plt.tight_layout()
        else:
            plt.figure(figsize=(10, 4))
            plt.imshow(audio_np, aspect="auto", origin="lower")
            plt.title(
                f"True: {true_class}, Predicted: {pred_class}\n(No significant audio content detected)"
            )
            plt.colorbar(format="%+2.0f dB")
            plt.tight_layout()

        spec_path = os.path.join(folder_path, f"{sample_id}_spectrogram.png")
        plt.savefig(spec_path)
        plt.close()
        print(f"Saved spectrogram to: {spec_path}")

        if os.path.exists(img_path) and os.path.exists(spec_path):
            successful_saves += 1
        else:
            print(f"Warning: One or more files not found after saving:")
            if not os.path.exists(img_path):
                print(f"  - Image not found: {img_path}")
            if not os.path.exists(spec_path):
                print(f"  - Spectrogram not found: {spec_path}")

        with open(csv_path, "a") as f:
            try:
                idx_int = int(idx)

                if hasattr(val_loader.dataset, "dataframe"):
                    df = val_loader.dataset.dataframe

                    if "image_file" in df.columns and "audio_file" in df.columns:
                        image_file = df.iloc[idx_int]["image_file"]
                        audio_file = df.iloc[idx_int]["audio_file"]
                    else:
                        image_file = "unknown"
                        audio_file = "unknown"
                else:
                    image_file = "unknown"
                    audio_file = "unknown"
            except Exception as e:
                print(f"Error accessing file paths: {e}")
                image_file = "unknown"
                audio_file = "unknown"

            f.write(
                f"{idx},{image_file},{audio_file},{true_class},{pred_class},{img_path},{spec_path}\n"
            )

    print(
        f"\nSuccessfully saved {successful_saves} out of {len(misclassified_indices)} misclassified samples"
    )
    print(f"Saved misclassified samples to {misclassified_dir}")
    print(f"See {csv_path} for a list of all misclassified samples")

    print("\nContents of misclassified samples directory:")
    for root, dirs, files in os.walk(misclassified_dir):
        level = root.replace(misclassified_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

    if successful_saves > 0:
        try:
            for root, dirs, files in os.walk(misclassified_dir):
                for f in files:
                    if f.endswith("_image.png"):
                        test_img_path = os.path.join(root, f)
                        test_img = cv2.imread(test_img_path)
                        if test_img is not None:
                            print(
                                f"\nSuccessfully verified that image file can be read: {test_img_path}"
                            )
                            print(f"Image dimensions: {test_img.shape}")
                        else:
                            print(
                                f"\nWarning: Image file exists but cannot be read: {test_img_path}"
                            )
                        break
                if test_img is not None:
                    break
        except Exception as e:
            print(f"Error verifying image file: {e}")

    if cfg.wandb.enabled:
        try:
            import pandas as pd

            wandb.log(
                {
                    "misclassified_samples_csv": wandb.Table(
                        dataframe=pd.read_csv(csv_path)
                    )
                }
            )
        except Exception as e:
            print(f"Error logging to wandb: {e}")

