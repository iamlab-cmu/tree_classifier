import numpy as np
import librosa
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    rms = librosa.feature.rms(y=y)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)
    
    features = np.concatenate([[rms, zcr, spec_cent, spec_bw], mfccs, chroma, spectral_contrast, tonnetz])
    return features

def analyze_features(X, y, feature_names, categories):
    """Analyze and visualize feature importance"""
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
    sns.heatmap(mean_features_matrix, 
                xticklabels=feature_names, 
                yticklabels=categories_no_ambient,
                cmap='coolwarm',
                center=0,
                vmin=-2, vmax=2)
    plt.xticks(rotation=45, ha='right')
    plt.title('Normalized Mean Feature Values by Category')
    plt.tight_layout()
    plt.savefig('feature_heatmap.png')
    plt.close()
    
    return X_scaled

def create_2d_visualization(X_scaled, y, categories, feature_names, output_file='pca_2d_plot.png'):
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print feature importance
    print("\nFeature importance based on PCA:")
    for i, component in enumerate(pca.components_):
        print(f"\nPrincipal Component {i+1} (Explains {pca.explained_variance_ratio_[i]:.2%} of variance):")
        feature_importance = list(zip(feature_names, component))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, weight in feature_importance[:5]:
            print(f"{feat}: {weight:.3f}")
    
    # Create 2D visualization
    plt.figure(figsize=(12, 8))
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    for category, color in zip(categories, colors):
        mask = y == category
        category_points = X_pca[mask]
        
        # Plot points
        plt.scatter(category_points[:, 0], category_points[:, 1],
                   label=category, alpha=0.7, c=color)
        
        # Plot centroid
        centroid = np.mean(category_points, axis=0)
        plt.scatter(centroid[0], centroid[1],
                   c=color, marker='*', s=200, edgecolor='black', linewidth=1,
                   label=f'{category} centroid')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Audio Categories Visualization (2D PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add confidence ellipses
    for category, color in zip(categories, colors):
        mask = y == category
        category_points = X_pca[mask]
        
        if len(category_points) >= 2:  # Need at least 2 points for covariance
            cov = np.cov(category_points.T)
            eigenvals, eigenvecs = np.linalg.eig(cov)
            
            # Calculate the angle
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            # Create the ellipse
            from matplotlib.patches import Ellipse
            ellip = Ellipse(xy=np.mean(category_points, axis=0),
                          width=2*np.sqrt(eigenvals[0])*2,
                          height=2*np.sqrt(eigenvals[1])*2,
                          angle=angle,
                          color=color,
                          fill=False,
                          linestyle='--',
                          alpha=0.5)
            plt.gca().add_patch(ellip)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return X_pca, pca

def main():
    # Define feature names
    feature_names = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + \
                   [f'MFCC_{i+1}' for i in range(13)] + \
                   [f'Chroma_{i+1}' for i in range(12)] + \
                   [f'Spectral_Contrast_{i+1}' for i in range(7)] + \
                   [f'Tonnetz_{i+1}' for i in range(6)]
    
    categories = ['ambient', 'leaf', 'twig', 'trunk']
    X = []
    y = []
    
    # Load and process audio files
    for category in categories:
        category_path = os.path.join('./output', category)
        for file in os.listdir(category_path):
            if file.endswith('.wav'):
                file_path = os.path.join(category_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(category)
    
    X = np.array(X)
    y = np.array(y)
    
    # Analyze features
    X_scaled = analyze_features(X, y, feature_names, categories)
    
    # Create PCA visualization
    X_pca, pca = create_2d_visualization(X_scaled, y, categories, feature_names)
    
    print("\nAnalysis complete! Generated files:")
    print("- feature_heatmap.png")
    print("- pca_2d_plot.png")

if __name__ == "__main__":
    main() 