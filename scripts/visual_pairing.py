import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
import torchaudio
import torch
import cv2

import matplotlib

if not hasattr(matplotlib, "colormaps"):
    matplotlib.colormaps = matplotlib.cm

try:
    import librosa
    import librosa.display
except ImportError as e:
    if "cannot import name 'colormaps'" in str(e):
        import sys
        import types

        mock_display = types.ModuleType("librosa.display")
        sys.modules["librosa.display"] = mock_display

        import librosa

        def specshow(data, **kwargs):
            return plt.imshow(data, origin="lower", aspect="auto", **kwargs)

        mock_display.specshow = specshow
    else:
        raise e

from matplotlib.gridspec import GridSpec


def create_spectrogram(audio_path, disable_normalization=True):
    """
    Generate a spectrogram from an audio file while preserving amplitude differences
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        if not disable_normalization:
            rms = torch.sqrt(torch.mean(waveform**2))
            if rms > 0:
                waveform = waveform / rms

        print(
            f"  Audio stats: min={waveform.min().item():.6f}, max={waveform.max().item():.6f}, rms={torch.sqrt(torch.mean(waveform**2)).item():.6f}"
        )

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=128
        )(waveform)

        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        if disable_normalization:
            spectrogram = mel_spec_db.numpy().squeeze()
        else:
            mel_spec_db_norm = (mel_spec_db - mel_spec_db.mean()) / (
                mel_spec_db.std() + 1e-10
            )
            spectrogram = mel_spec_db_norm.numpy().squeeze()

        return spectrogram

    except Exception as e:
        print(f"Error processing audio {audio_path}: {str(e)}")
        return np.zeros((128, 1024))


def load_and_square_image(image_path):
    """
    Load an image and make it square by center cropping, with a black border
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to read image: {image_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        min_dim = min(h, w)

        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2

        square_img = image[start_y : start_y + min_dim, start_x : start_x + min_dim]

        border_size = 3
        square_img_with_border = cv2.copyMakeBorder(
            square_img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Black border
        )

        return square_img_with_border

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def main():
    dataset_path = (
        "/home/dorry/Desktop/research/audio_visual_dataset_no_norm/dataset.csv"
    )
    dataset_dir = "/home/dorry/Desktop/research/audio_visual_dataset_no_norm"

    dataset = pd.read_csv(dataset_path)

    audio_col = "audio_file"
    image_col = "image_file"
    class_column = "category"

    ordered_categories = ["leaf", "twig", "trunk", "ambient"]

    available_categories = dataset[class_column].unique()

    unique_classes = [cat for cat in ordered_categories if cat in available_categories]

    for cat in available_categories:
        if cat not in unique_classes:
            unique_classes.append(cat)

    if len(unique_classes) > 4:
        unique_classes = unique_classes[:4]
        print(f"Using these 4 categories: {unique_classes}")
    elif len(unique_classes) < 4:
        print(f"Warning: Only {len(unique_classes)} categories found: {unique_classes}")

    num_classes = len(unique_classes)
    samples_per_class = 4  # We want 4 samples per category/class

    fig = plt.figure(figsize=(5 * num_classes, 4 * samples_per_class))

    outer_gs = GridSpec(
        samples_per_class, num_classes, figure=fig, hspace=0.3, wspace=0.3
    )

    plt.rcdefaults()

    for class_idx, class_name in enumerate(unique_classes):
        class_data = dataset[dataset[class_column] == class_name]

        sample_indices = list(range(len(class_data)))
        if len(sample_indices) > samples_per_class:
            sample_indices = random.sample(sample_indices, samples_per_class)

        fig.text(
            0.125 + 0.25 * class_idx,
            0.95,
            class_name,
            fontsize=16,
            ha="center",
            va="center",
        )

        for i, sample_idx in enumerate(sample_indices):
            if i < samples_per_class:  # Just a safeguard
                image_path = os.path.join(
                    dataset_dir, class_data.iloc[sample_idx][image_col]
                )
                audio_path = os.path.join(
                    dataset_dir, class_data.iloc[sample_idx][audio_col]
                )

                print(f"Class: {class_name}, Sample {i + 1}:")
                print(f"  Image: {image_path}")
                print(f"  Audio: {audio_path}")

                cell = outer_gs[i, class_idx].subgridspec(
                    1, 2, wspace=0.1, width_ratios=[1, 1]
                )

                ax_img = fig.add_subplot(cell[0, 0])

                if os.path.exists(image_path):
                    square_img = load_and_square_image(image_path)
                    if square_img is not None:
                        img = ax_img.imshow(square_img)

                        from matplotlib.patches import Rectangle

                        height, width = square_img.shape[:2]
                        rect = Rectangle(
                            (-0.5, -0.5),  # Starting position (lower left corner)
                            width + 0.99,  # Width with a small adjustment
                            height + 0.99,  # Height with a small adjustment
                            fill=False,
                            edgecolor="black",
                            linewidth=3,
                        )
                        ax_img.add_patch(rect)
                    else:
                        ax_img.text(
                            0.5,
                            0.5,
                            "Image processing failed",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

                    ax_img.set_title("Image", fontsize=10)
                    ax_img.axis("off")
                else:
                    ax_img.text(
                        0.5,
                        0.5,
                        f"Image not found",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    ax_img.axis("off")
                    print(f"Image not found: {image_path}")

                ax_spec = fig.add_subplot(cell[0, 1])

                if os.path.exists(audio_path):
                    try:
                        spectrogram = create_spectrogram(
                            audio_path, disable_normalization=True
                        )

                        img_spec = ax_spec.imshow(
                            spectrogram,
                            aspect="auto",
                            origin="lower",
                            cmap="viridis",
                            vmin=-80,
                            vmax=0,
                        )

                        ax_spec.set_title("Spectrogram", fontsize=10)

                        ax_spec.set_xticks([])
                        ax_spec.set_yticks([])
                        ax_spec.set_xlabel("")
                        ax_spec.set_ylabel("")
                    except Exception as e:
                        ax_spec.text(
                            0.5,
                            0.5,
                            f"Error creating spectrogram",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
                        ax_spec.axis("off")
                        print(f"Error with {audio_path}: {e}")
                else:
                    ax_spec.text(
                        0.5,
                        0.5,
                        f"Audio not found",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    ax_spec.axis("off")
                    print(f"Audio not found: {audio_path}")

                ax_img.set_aspect("equal")

                ax_img.set_adjustable("box")
                ax_spec.set_adjustable("box")

                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(3)  # Keep original thickness for image

                for spine in ax_spec.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1.5)  # Half the thickness for spectrogram

                for ax in [ax_img, ax_spec]:
                    ax.set_frame_on(True)
                    ax.patch.set_edgecolor("black")

                ax_img.patch.set_linewidth(3)
                ax_spec.patch.set_linewidth(1.5)  # Half the thickness

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room at the top for titles

    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "class_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

