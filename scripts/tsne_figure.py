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

# Define feature extraction functions for audio
def extract_audio_features(audio_path):
    """
    Extract audio features using librosa
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Convert to numpy array
        y_audio = waveform.numpy().squeeze()
        
        # Extract features
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y_audio, sr=sample_rate, n_mfcc=20).mean(axis=1)
        
        # RMS energy (Root Mean Square)
        rms = librosa.feature.rms(y=y_audio).mean(axis=1)
        
        # Chroma feature
        chroma = librosa.feature.chroma_stft(y=y_audio, sr=sample_rate).mean(axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sample_rate).mean(axis=1)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sample_rate).mean(axis=1)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sample_rate).mean(axis=1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y_audio).mean(axis=1)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y_audio, sr=sample_rate).mean(axis=1)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y_audio).mean(axis=1)
        
        # Tempo estimation
        onset_env = librosa.onset.onset_strength(y=y_audio, sr=sample_rate)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate)
        
        # Tonnetz - tonal centroid features
        y_harmonic = librosa.effects.harmonic(y_audio)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sample_rate).mean(axis=1)
        
        # Mel spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sample_rate)
        mel_spec_mean = np.mean(mel_spec, axis=1)[:10]  # Take first 10 bands
        mel_spec_var = np.var(mel_spec, axis=1)[:10]    # Take first 10 bands
        
        # Combine all features
        features = np.hstack([
            mfccs,              # 20 features
            rms,                # 1 feature
            chroma,             # 12 features 
            contrast,           # 7 features
            centroid,           # 1 feature
            rolloff,            # 1 feature
            zcr,                # 1 feature
            bandwidth,          # 1 feature
            flatness,           # 1 feature
            tempo,              # 1 feature
            tonnetz,            # 6 features
            mel_spec_mean,      # 10 features
            mel_spec_var        # 10 features
        ])
        
        # Print feature dimensions for debugging
        # print(f"Feature vector size: {len(features)}")
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print stack trace for debugging
        return None

def main():
    # Set the path to your dataset
    dataset_path = "/home/dorry/Desktop/research/learning/audio_visual_dataset_no_norm/dataset.csv"
    dataset_dir = "/home/dorry/Desktop/research/learning/audio_visual_dataset_no_norm"
    dataset_path_robo = "/home/dorry/Desktop/research/learning/audio_visual_dataset_robo_no_norm/dataset.csv"
    dataset_dir_robo = "/home/dorry/Desktop/research/learning/audio_visual_dataset_robo_no_norm"
    
    # Define colors for each category - updated with darker yellow
    colors = {
        'leaf': '#00FF00',    # green
        'twig': '#FF0000',    # red
        'trunk': '#0000FF',   # blue
        'ambient': '#CCCC00'  # darker yellow
    }
    
    # Update all markers to be circles
    markers = {
        'leaf': 'o',      # circle
        'twig': 'o',      # circle (changed from square)
        'trunk': 'o',     # circle (changed from triangle)
        'ambient': 'o'    # circle (changed from diamond)
    }
    
    # Remove Times New Roman requirement since we're using placeholders
    plt.rcParams['font.family'] = 'serif'  # Use any serif font
    
    # Try setting the seaborn style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception as e:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception as e:
            print("Warning: Could not set seaborn style. Using default style.")
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Function to process each dataset and plot on the given axis
    def process_and_plot_dataset(dataset_path, dataset_dir, ax, title):
        # Read the dataset CSV
        print(f"Loading dataset from {dataset_path}")
        dataset = pd.read_csv(dataset_path)
        
        # Get column names from your dataset
        audio_col = 'audio_file'
        image_col = 'image_file'
        class_column = 'category'
        
        # Print dataset info
        print(f"Dataset loaded with {len(dataset)} entries")
        print(f"Categories: {dataset[class_column].unique()}")
        print(f"Category distribution: \n{dataset[class_column].value_counts()}")
        
        # Balance the dataset by downsampling to the smallest category size
        category_counts = dataset[class_column].value_counts()
        min_category_size = category_counts.min()
        print(f"Balancing dataset by downsampling to {min_category_size} samples per category")
        
        # Create a balanced subset
        balanced_dataset = pd.DataFrame()
        for category in dataset[class_column].unique():
            category_data = dataset[dataset[class_column] == category]
            # Randomly sample to match the smallest category size
            sampled_data = category_data.sample(n=min_category_size, random_state=42)
            balanced_dataset = pd.concat([balanced_dataset, sampled_data])
        
        # Shuffle the balanced dataset
        balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Print balanced dataset info
        print(f"Balanced dataset created with {len(balanced_dataset)} entries")
        print(f"Balanced category distribution: \n{balanced_dataset[class_column].value_counts()}")
        
        subset_fraction = 0.70
        ro = 0.01
        subset_size = int(len(balanced_dataset) * subset_fraction)
        dataset_subset = balanced_dataset.sample(n=subset_size, random_state=42)
        print(f"Using {subset_size} samples ({subset_fraction*100:.0f}% of balanced dataset)")
        print(f"Subset category distribution: \n{dataset_subset[class_column].value_counts()}")
        
        # Extract features
        print(f"Extracting audio features for {title}...")
        features = []
        labels = []
        successful_files = 0
        
        for idx, row in tqdm(dataset_subset.iterrows(), total=len(dataset_subset)):
            try:
                audio_path = os.path.join(dataset_dir, row[audio_col])
                category = row[class_column]
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    print(f"File does not exist: {audio_path}")
                    continue
                    
                # Extract audio features
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
        
        print(f"\nSuccessfully processed {successful_files} out of {len(dataset_subset)} files")
        
        if len(features) == 0:
            raise ValueError(f"No features were successfully extracted from any audio files for {title}")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Feature array shape: {X.shape}")
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform t-SNE
        print(f"Performing t-SNE for {title}...")
        start_time = time.time()
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
        
        # Turn off grid
        ax.grid(False)
        
        # Compute and store centroids for each class
        centroids = {}
        
        # Plot each category with enhanced markers
        handles = []
        for category in np.unique(y):
            mask = y == category
            
            # Extract points for this category
            points = X_tsne[mask]
            
            # Skip empty categories
            if len(points) == 0:
                print(f"Warning: No points found for category '{category}'")
                continue
            
            # Compute centroid
            centroid = np.mean(points, axis=0)
            centroids[category] = centroid
            
            # Plot with enhanced markers - removed borders
            scatter = ax.scatter(
                points[:, 0], 
                points[:, 1], 
                c=colors.get(category, 'gray'),
                marker=markers.get(category, 'o'),
                label=category,
                alpha=0.7,
                s=60,
                edgecolor='none',  # Remove borders by setting edgecolor to 'none'
                linewidth=0        # Set linewidth to 0 to ensure no border
            )
            
            # Only collect handles from the first plot
            if title == "Probe Data":
                handles.append(scatter)
        
        # Add annotations for centroids - replaced with empty rectangular placeholders
        for category, centroid in centroids.items():
            # Create a blank rectangular placeholder instead of text label
            ax.text(
                centroid[0], centroid[1],
                "",  # Empty text
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    facecolor='white', 
                    alpha=0.8, 
                    edgecolor='black',
                    linewidth=0.5,
                    boxstyle='round,pad=0.5',
                    mutation_scale=1.0
                )
            )
        
        # Set title and labels
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE dimension 2', fontsize=14)
        
        # Remove tick labels (grid numbers) while keeping the axes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Keep tick marks but hide the labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        return handles
    
    # Process both datasets
    handles = process_and_plot_dataset(dataset_path, dataset_dir, ax1, "Probe Data")
    process_and_plot_dataset(dataset_path_robo, dataset_dir_robo, ax2, "Robot Data")
    
    # Add a single legend for the entire figure
    fig.legend(
        handles=handles,
        labels=colors.keys(),
        title='Tree Region', 
        fontsize=14,
        title_fontsize=15,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),  # Position at bottom center
        ncol=len(colors),            # Put all categories in one row
        frameon=True, 
        facecolor='white', 
        edgecolor='gray',
        framealpha=0.8,
        markerscale=1.5
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend at the bottom
    
    # Save high-resolution figures for publication
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as high-resolution PNG
    output_path_png = os.path.join(output_dir, "comparative_tsne_visualization.png")
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"High-resolution PNG saved to {output_path_png}")
    
    # Try to save PDF, but catch any errors
    try:
        # Also save as PDF for vector graphics in publications
        output_path_pdf = os.path.join(output_dir, "comparative_tsne_visualization.pdf")
        plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
        print(f"Vector PDF saved to {output_path_pdf}")
    except Exception as e:
        print(f"Warning: Could not save PDF. {str(e)}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()