import numpy as np
import librosa
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_features(audio_path):
    """
    Extract both mel spectrogram and traditional audio features
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # 1. Mel Spectrogram Features
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=sr/2,
        n_fft=2048,
        hop_length=512
    )
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    mel_features = np.mean(mel_spect_db, axis=1)
    
    # 2. Traditional Audio Features
    # Basic features
    rms = librosa.feature.rms(y=y)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    # Combine all features
    basic_features = np.array([rms, zcr, spec_cent, spec_bw, rolloff])
    all_features = np.concatenate([
        mel_features,          # Mel spectrogram features
        basic_features,        # Basic audio features
        mfccs_mean,           # MFCCs
        chroma_mean,          # Chroma
        contrast_mean,        # Spectral Contrast
        tonnetz_mean          # Tonnetz
    ])
    
    return all_features

def plot_tsne_segments():
    segments_dir = 'robo_segments'
    categories = ['leaf', 'twig', 'trunk']
    
    # Print debugging information
    print(f"Looking for segment directories in: {segments_dir}")
    print(f"Checking for categories: {categories}")
    
    # Check if the base directory exists
    if not os.path.exists(segments_dir):
        print(f"ERROR: Base directory '{segments_dir}' does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print("Available directories in current location:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}")
        raise ValueError("Base segments directory not found!")
    
    # Check which categories exist
    existing_categories = []
    for category in categories:
        category_path = os.path.join(segments_dir, category)
        if os.path.exists(category_path):
            existing_categories.append(category)
            print(f"Found category directory: {category_path}")
        else:
            print(f"Category directory not found: {category_path}")
    
    if not existing_categories:
        raise ValueError("No segment directories found!")
    
    # Collect features and labels
    features = []
    labels = []
    
    print("Extracting combined features from segments...")
    for category in existing_categories:
        category_path = os.path.join(segments_dir, category)
        files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        
        print(f"Found {len(files)} WAV files in {category_path}")
        
        for file in tqdm(files, desc=f"Processing {category}"):
            try:
                file_path = os.path.join(category_path, file)
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    if not features:
        raise ValueError("No features could be extracted from the segments!")
    
    # Convert to numpy arrays
    X = np.array(features)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform t-SNE
    print("Performing t-SNE on combined features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=2000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for categories
    colors = {'leaf': 'green', 'twig': 'brown', 'trunk': 'gray'}
    markers = {'leaf': 'o', 'twig': 's', 'trunk': '^'}
    
    # Plot each category with different colors and markers
    for category in existing_categories:
        mask = np.array(labels) == category
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[category], 
                   marker=markers[category],
                   label=category, 
                   alpha=0.6,
                   s=100)  # Increased marker size
    
    plt.title('t-SNE visualization of Combined Audio Features')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add feature count information
    plt.text(0.02, 0.98, 
             f'Total features: {X.shape[1]}\n'
             f'Mel bands: 128\n'
             f'Other features: {X.shape[1]-128}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        plot_tsne_segments()
    except Exception as e:
        print(f"Error: {str(e)}")
