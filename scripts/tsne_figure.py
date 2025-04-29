import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import cv2
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm
import time
import matplotlib.font_manager as fm


def extract_audio_features(audio_path):
    """
    Extract audio features using librosa
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        y_audio = waveform.numpy().squeeze()

        mfccs = librosa.feature.mfcc(y=y_audio, sr=sample_rate, n_mfcc=20).mean(axis=1)

        rms = librosa.feature.rms(y=y_audio).mean(axis=1)

        chroma = librosa.feature.chroma_stft(y=y_audio, sr=sample_rate).mean(axis=1)

        contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sample_rate).mean(
            axis=1
        )

        centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sample_rate).mean(
            axis=1
        )

        rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sample_rate).mean(
            axis=1
        )

        zcr = librosa.feature.zero_crossing_rate(y_audio).mean(axis=1)

        bandwidth = librosa.feature.spectral_bandwidth(y=y_audio, sr=sample_rate).mean(
            axis=1
        )

        flatness = librosa.feature.spectral_flatness(y=y_audio).mean(axis=1)

        onset_env = librosa.onset.onset_strength(y=y_audio, sr=sample_rate)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)

        y_harmonic = librosa.effects.harmonic(y_audio)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sample_rate).mean(axis=1)

        mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sample_rate)
        mel_spec_mean = np.mean(mel_spec, axis=1)[:10]  # Take first 10 bands
        mel_spec_var = np.var(mel_spec, axis=1)[:10]  # Take first 10 bands

        features = np.hstack(
            [
                mfccs,  # 20 features
                rms,  # 1 feature
                chroma,  # 12 features
                contrast,  # 7 features
                centroid,  # 1 feature
                rolloff,  # 1 feature
                zcr,  # 1 feature
                bandwidth,  # 1 feature
                flatness,  # 1 feature
                tempo,  # 1 feature
                tonnetz,  # 6 features
                mel_spec_mean,  # 10 features
                mel_spec_var,  # 10 features
            ]
        )

        return features

    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {str(e)}")
        import traceback

        print(traceback.format_exc())  # Print stack trace for debugging
        return None


def main():
    dataset_path = (
        "/home/dorry/Desktop/research/learning/audio_visual_dataset_no_norm/dataset.csv"
    )
    dataset_dir = "/home/dorry/Desktop/research/learning/audio_visual_dataset_no_norm"
    dataset_path_robo = "/home/dorry/Desktop/research/learning/audio_visual_dataset_robo_no_norm/dataset.csv"
    dataset_dir_robo = (
        "/home/dorry/Desktop/research/learning/audio_visual_dataset_robo_no_norm"
    )

    colors = {
        "leaf": "#00FF00",  # green
        "twig": "#FF0000",  # red
        "trunk": "#0000FF",  # blue
        "ambient": "#CCCC00",  # darker yellow
    }

    markers = {
        "leaf": "o",  # circle
        "twig": "o",  # circle (changed from square)
        "trunk": "o",  # circle (changed from triangle)
        "ambient": "o",  # circle (changed from diamond)
    }

    plt.rcParams["font.family"] = "serif"  # Use any serif font

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception as e:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception as e:
            print("Warning: Could not set seaborn style. Using default style.")
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.alpha"] = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    def process_and_plot_dataset(dataset_path, dataset_dir, ax, title):
        print(f"Loading dataset from {dataset_path}")
        dataset = pd.read_csv(dataset_path)

        audio_col = "audio_file"
        image_col = "image_file"
        class_column = "category"

        print(f"Dataset loaded with {len(dataset)} entries")
        print(f"Categories: {dataset[class_column].unique()}")
        print(f"Category distribution: \n{dataset[class_column].value_counts()}")

        category_counts = dataset[class_column].value_counts()
        min_category_size = category_counts.min()
        print(
            f"Balancing dataset by downsampling to {min_category_size} samples per category"
        )

        balanced_dataset = pd.DataFrame()
        for category in dataset[class_column].unique():
            category_data = dataset[dataset[class_column] == category]
            sampled_data = category_data.sample(n=min_category_size, random_state=42)
            balanced_dataset = pd.concat([balanced_dataset, sampled_data])

        balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        print(f"Balanced dataset created with {len(balanced_dataset)} entries")
        print(
            f"Balanced category distribution: \n{balanced_dataset[class_column].value_counts()}"
        )

        subset_fraction = 0.70
        ro = 0.01
        subset_size = int(len(balanced_dataset) * subset_fraction)
        dataset_subset = balanced_dataset.sample(n=subset_size, random_state=42)
        print(
            f"Using {subset_size} samples ({subset_fraction * 100:.0f}% of balanced dataset)"
        )
        print(
            f"Subset category distribution: \n{dataset_subset[class_column].value_counts()}"
        )

        print(f"Extracting audio features for {title}...")
        features = []
        labels = []
        successful_files = 0

        for idx, row in tqdm(dataset_subset.iterrows(), total=len(dataset_subset)):
            try:
                audio_path = os.path.join(dataset_dir, row[audio_col])
                category = row[class_column]

                if not os.path.exists(audio_path):
                    print(f"File does not exist: {audio_path}")
                    continue

                audio_features = extract_audio_features(audio_path)

                if audio_features is not None and len(audio_features) > 0:
                    features.append(audio_features)
                    labels.append(category)
                    successful_files += 1
                else:
                    print(f"No features extracted from {audio_path}")

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue

        print(
            f"\nSuccessfully processed {successful_files} out of {len(dataset_subset)} files"
        )

        if len(features) == 0:
            raise ValueError(
                f"No features were successfully extracted from any audio files for {title}"
            )

        X = np.array(features)
        y = np.array(labels)

        print(f"Feature array shape: {X.shape}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"Performing t-SNE for {title}...")
        start_time = time.time()
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        X_tsne = tsne.fit_transform(X_scaled)
        print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")

        ax.grid(False)

        centroids = {}

        handles = []
        for category in np.unique(y):
            mask = y == category

            points = X_tsne[mask]

            if len(points) == 0:
                print(f"Warning: No points found for category '{category}'")
                continue

            centroid = np.mean(points, axis=0)
            centroids[category] = centroid

            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=colors.get(category, "gray"),
                marker=markers.get(category, "o"),
                label=category,
                alpha=0.7,
                s=60,
                edgecolor="none",  # Remove borders by setting edgecolor to 'none'
                linewidth=0,  # Set linewidth to 0 to ensure no border
            )

            if title == "Probe Data":
                handles.append(scatter)

        for category, centroid in centroids.items():
            ax.text(
                centroid[0],
                centroid[1],
                "",  # Empty text
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                    boxstyle="round,pad=0.5",
                    mutation_scale=1.0,
                ),
            )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("t-SNE dimension 1", fontsize=14)
        ax.set_ylabel("t-SNE dimension 2", fontsize=14)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.tick_params(axis="both", which="major", labelsize=12)

        return handles

    handles = process_and_plot_dataset(dataset_path, dataset_dir, ax1, "Probe Data")
    process_and_plot_dataset(dataset_path_robo, dataset_dir_robo, ax2, "Robot Data")

    fig.legend(
        handles=handles,
        labels=colors.keys(),
        title="Tree Region",
        fontsize=14,
        title_fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),  # Position at bottom center
        ncol=len(colors),  # Put all categories in one row
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.8,
        markerscale=1.5,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend at the bottom

    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)

    output_path_png = os.path.join(output_dir, "comparative_tsne_visualization.png")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"High-resolution PNG saved to {output_path_png}")

    try:
        output_path_pdf = os.path.join(output_dir, "comparative_tsne_visualization.pdf")
        plt.savefig(output_path_pdf, format="pdf", bbox_inches="tight")
        print(f"Vector PDF saved to {output_path_pdf}")
    except Exception as e:
        print(f"Warning: Could not save PDF. {str(e)}")

    plt.show()


if __name__ == "__main__":
    main()

