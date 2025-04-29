import numpy as np
import librosa
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tqdm import tqdm
import seaborn as sns

def extract_features(audio_path):
    """Extract both mel spectrogram and traditional audio features"""
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
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    
    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    
    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)
    
    # Combine all features
    basic_features = np.array([rms, zcr, spec_cent, spec_bw, spectral_rolloff])
    all_features = np.concatenate([
        mel_features,          # Mel spectrogram features
        basic_features,        # Basic audio features
        mfccs,                # MFCCs
        chroma,               # Chroma
        spectral_contrast,    # Spectral Contrast
        tonnetz               # Tonnetz
    ])
    
    return all_features

def analyze_features(X, y, feature_names, categories):
    """Analyze and visualize feature importance"""
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate and display mean features for each category
    mean_features_by_category = {}
    for category in categories:
        mask = y == category
        category_features = X_scaled[mask]
        mean_features = category_features.mean(axis=0)
        mean_features_by_category[category] = mean_features
        
        print(f"\n{category.upper()}:")
        for name, value in zip(feature_names, mean_features):
            print(f"{name}: {value:.3f}")
    
    # Create heatmap of mean features
    plt.figure(figsize=(15, 10))
    categories_no_ambient = [cat for cat in categories if cat != 'ambient']
    mean_features_matrix = np.array([mean_features_by_category[cat] for cat in categories_no_ambient])
    
    # Create heatmap with selected features for better visibility
    selected_features = feature_names[128:] # Skip mel bands for heatmap
    sns.heatmap(mean_features_matrix[:, 128:], 
                xticklabels=selected_features,
                yticklabels=categories_no_ambient,
                cmap='coolwarm',
                center=0,
                vmin=-2, vmax=2)
    plt.xticks(rotation=45, ha='right')
    plt.title('Normalized Mean Feature Values by Category (Excluding Mel Bands)')
    plt.tight_layout()
    plt.savefig('feature_heatmap.png')
    plt.close()
    
    return X_scaled

def create_3d_animation(X_scaled, y, categories, feature_names, output_file='pca_3d_animation.gif'):
    # Perform PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print feature importance for non-mel features
    print("\nFeature importance based on PCA (Top features excluding mel bands):")
    for i, component in enumerate(pca.components_):
        print(f"\nPrincipal Component {i+1} (Explains {pca.explained_variance_ratio_[i]:.2%} of variance):")
        feature_importance = list(zip(feature_names[128:], component[128:]))  # Skip mel bands
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, weight in feature_importance[:5]:
            print(f"{feat}: {weight:.3f}")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    for category, color in zip(categories, colors):
        mask = y == category
        category_points = X_pca[mask]
        
        ax.scatter(category_points[:, 0], category_points[:, 1], category_points[:, 2], 
                  label=category, alpha=0.7, c=color)
        
        centroid = np.mean(category_points, axis=0)
        ax.scatter(centroid[0], centroid[1], centroid[2], 
                  c=color, marker='*', s=200, edgecolor='black', linewidth=1,
                  label=f'{category} centroid')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title('Audio Categories Visualization (3D PCA)')
    plt.legend()
    
    def rotate(frame):
        ax.view_init(elev=20., azim=frame)
        return fig,
    
    anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), 
                                 interval=50, blit=True)
    anim.save(output_file, writer='pillow')
    plt.close()
    
    return X_pca, pca

def main():
    # Define feature names
    feature_names = [f'Mel_{i+1}' for i in range(128)] + \
                   ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff'] + \
                   [f'MFCC_{i+1}' for i in range(13)] + \
                   [f'Chroma_{i+1}' for i in range(12)] + \
                   [f'Spectral_Contrast_{i+1}' for i in range(7)] + \
                   [f'Tonnetz_{i+1}' for i in range(6)]
    
    categories = ['leaf', 'twig', 'trunk']  # Removed 'ambient'
    X = []
    y = []
    
    # Load and process audio files
    print("Extracting features from audio files...")
    for category in categories:
        category_path = os.path.join('./output', category)
        if not os.path.exists(category_path):
            continue
            
        files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        for file in tqdm(files, desc=f"Processing {category}"):
            try:
                file_path = os.path.join(category_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(category)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal features: {X.shape[1]}")
    print(f"Mel bands: 128")
    print(f"Other features: {X.shape[1]-128}")
    
    # Analyze features
    X_scaled = analyze_features(X, y, feature_names, categories)
    
    # Create PCA visualization
    X_pca, pca = create_3d_animation(X_scaled, y, categories, feature_names)
    
    print("\nAnalysis complete! Generated files:")
    print("- feature_heatmap.png")
    print("- pca_3d_animation.gif")

if __name__ == "__main__":
    main() 