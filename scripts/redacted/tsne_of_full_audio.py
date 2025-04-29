import numpy as np
import librosa
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def extract_features(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Basic features - ensure these are scalar values
    rms = float(librosa.feature.rms(y=y)[0].mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=y)[0].mean())
    spec_cent = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    spec_bw = float(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean())
    spec_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean())
    spec_flatness = float(librosa.feature.spectral_flatness(y=y)[0].mean())
    
    # Additional spectral features
    spec_contrast_enhanced = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6).mean(axis=1)
    
    # Enhanced MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = mfccs.mean(axis=1)
    mfccs_delta = librosa.feature.delta(mfccs).mean(axis=1)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2).mean(axis=1)
    
    # Enhanced chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12).mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1)
    
    # Mel spectrogram statistics
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = mel_spec.mean(axis=1)[:20]  # Take first 20 bands
    mel_var = mel_spec.var(axis=1)[:20]    # Take first 20 bands
    
    # Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)
    
    # Convert scalar values to arrays for concatenation
    basic_features = np.array([rms, zcr, spec_cent, spec_bw, spec_rolloff, spec_flatness])
    
    # Ensure all arrays are 1D and have consistent shapes
    features = np.concatenate([
        basic_features,
        mfccs_mean.flatten(),
        mfccs_delta.flatten(),
        mfccs_delta2.flatten(),
        chroma.flatten(),
        chroma_cens.flatten(),
        spec_contrast_enhanced.flatten(),
        tonnetz.flatten(),
        mel_mean.flatten(),
        mel_var.flatten()
    ])
    
    return features

def main():
    # Load data from cleaned_sounds directory
    base_dir = 'cleaned_sounds'
    X = []
    y = []
    
    # Process each category
    all_files = []
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            print(f"Found {category} audio files")
            files_in_category = [f for f in os.listdir(category_path) 
                               if f.startswith('cleaned_') and f.endswith('.wav')]
            print(f"Number of files in {category}: {len(files_in_category)}")
            for file in files_in_category:
                all_files.append((category_path, file, category))

    print(f"Total number of files to process: {len(all_files)}")
    if len(all_files) == 0:
        raise ValueError(f"No audio files found in {base_dir}. Check if the directory exists and contains .wav files.")

    # Process files with progress bar
    print("Processing audio files...")
    successful_files = 0
    for category_path, file, category in tqdm(all_files, desc="Extracting features"):
        try:
            file_path = os.path.join(category_path, file)
            print(f"\nProcessing file: {file_path}")  # Debug print
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue
                
            # Try to load the audio file and print its properties
            y_audio, sr = librosa.load(file_path)
            print(f"Audio loaded - Duration: {len(y_audio)/sr:.2f}s, Sample rate: {sr}Hz")
            
            features = extract_features(file_path)
            if features is not None and len(features) > 0:
                print(f"Features extracted successfully - length: {len(features)}")
                X.append(features)
                y.append(category)
                successful_files += 1
            else:
                print(f"No features extracted from {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full stack trace
            continue

    print(f"\nSuccessfully processed {successful_files} out of {len(all_files)} files")
    
    if len(X) == 0:
        raise ValueError("No features were successfully extracted from any audio files")

    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature array shape: {X.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform t-SNE
    print("Performing t-SNE...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Define colors for categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    categories = sorted(set(y))
    
    # Plot each category
    for category, color in zip(categories, colors):
        mask = y == category
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=category, alpha=0.7, c=color)
    
    plt.title('Full Audio Files Visualization (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    
    # Save the plot
    plt.savefig('full_audio_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 