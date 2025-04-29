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
from itertools import combinations
import itertools

def extract_all_features(audio_path):
    """Extract all possible features separately including mel spectrogram"""
    y, sr = librosa.load(audio_path)
    
    # Mel spectrogram features
    mel_spect = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=sr/2, n_fft=2048, hop_length=512
    )
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    features_dict = {
        'mel': np.mean(mel_spect_db, axis=1),  # 128 features
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1),
        'mfcc_delta': librosa.feature.delta(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)).mean(axis=1),
        'mfcc_delta2': librosa.feature.delta(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), order=2).mean(axis=1),
        'rms': np.array([librosa.feature.rms(y=y)[0].mean()]),
        'zcr': np.array([librosa.feature.zero_crossing_rate(y=y)[0].mean()]),
        'spectral_centroid': np.array([librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()]),
        'spectral_bandwidth': np.array([librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()]),
        'spectral_rolloff': np.array([librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()]),
        'spectral_flatness': np.array([librosa.feature.spectral_flatness(y=y)[0].mean()]),
        'chroma': librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
        'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1),
        'chroma_cens': librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1),
        'tonnetz': librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1),
        'tempogram': np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)
    }
    
    return features_dict

def calculate_centroid_distances(X_pca, y, categories):
    """Calculate the sum of pairwise distances between category centroids"""
    centroids = {}
    for category in categories:
        mask = y == category
        category_points = X_pca[mask]
        centroids[category] = np.mean(category_points, axis=0)
    
    # Calculate sum of pairwise distances
    total_distance = 0
    for cat1, cat2 in itertools.combinations(categories, 2):
        distance = np.linalg.norm(centroids[cat1] - centroids[cat2])
        total_distance += distance
    
    return total_distance

def evaluate_feature_combination(X, y, categories, feature_names):
    """Perform PCA and evaluate centroid distances"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    distance = calculate_centroid_distances(X_pca, y, categories)
    variance_explained = sum(pca.explained_variance_ratio_)
    
    # Combine distance and variance explained as a score
    score = distance * variance_explained
    
    return score, X_pca, pca

def main():
    categories = ['leaf', 'twig', 'trunk']
    
    # Load all features first
    print("Extracting features from segments...")
    all_features_data = []
    labels = []
    segments_dir = './segments'
    
    for category in categories:
        category_path = os.path.join(segments_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} does not exist")
            continue
            
        files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        for file in tqdm(files, desc=f"Processing {category}"):
            try:
                file_path = os.path.join(category_path, file)
                features = extract_all_features(file_path)
                all_features_data.append(features)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    # Define feature groups with (number of features, description)
    feature_groups = {
        'mel': (128, 'Mel Spectrogram'),
        'mfcc': (20, 'MFCC'),
        'mfcc_delta': (20, 'MFCC Delta'),
        'mfcc_delta2': (20, 'MFCC Delta2'),
        'basic': (6, 'Basic'),  # rms, zcr, spectral_centroid, bandwidth, rolloff, flatness
        'chroma_all': (36, 'All Chroma'),  # combining all chroma features
        'spectral_contrast': (7, 'Spectral Contrast'),
        'tonnetz': (6, 'Tonnetz'),
        'tempogram': (384, 'Tempogram')  # temporal features
    }
    
    # Try different combinations of feature groups
    best_score = -1
    best_combination = None
    best_X_pca = None
    best_pca = None
    
    print("\nEvaluating feature combinations...")
    for r in range(1, len(feature_groups) + 1):
        for combination in tqdm(itertools.combinations(feature_groups.keys(), r)):
            # Combine features for this combination
            X = []
            feature_names = []
            
            for sample in all_features_data:
                sample_features = []
                for feature_group in combination:
                    if feature_group == 'basic':
                        basic_features = np.concatenate([
                            sample['rms'],
                            sample['zcr'],
                            sample['spectral_centroid'],
                            sample['spectral_bandwidth'],
                            sample['spectral_rolloff'],
                            sample['spectral_flatness']
                        ])
                        sample_features.extend(basic_features)
                    elif feature_group == 'chroma_all':
                        chroma_features = np.concatenate([
                            sample['chroma'],
                            sample['chroma_cqt'],
                            sample['chroma_cens']
                        ])
                        sample_features.extend(chroma_features)
                    else:
                        sample_features.extend(sample[feature_group])
                X.append(sample_features)
            
            X = np.array(X)
            y = np.array(labels)
            
            # Evaluate this combination
            score, X_pca, pca = evaluate_feature_combination(X, y, categories, feature_names)
            
            if score > best_score:
                best_score = score
                best_combination = combination
                best_X_pca = X_pca
                best_pca = pca
    
    print("\nBest feature combination:", best_combination)
    print("Score:", best_score)
    
    # Create visualization with the best combination
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'leaf': '#99FF99', 'twig': '#8B4513', 'trunk': '#808080'}
    markers = {'leaf': 'o', 'twig': 's', 'trunk': '^'}
    
    # Plot points and centroids
    for category in categories:
        mask = y == category
        category_points = best_X_pca[mask]
        
        # Plot points
        ax.scatter(category_points[:, 0], 
                  category_points[:, 1], 
                  category_points[:, 2],
                  label=category,
                  alpha=0.7,
                  c=colors[category],
                  marker=markers[category])
        
        # Plot centroid
        centroid = np.mean(category_points, axis=0)
        ax.scatter(centroid[0], centroid[1], centroid[2],
                  c=colors[category],
                  marker='*',
                  s=200,
                  edgecolor='black',
                  linewidth=1,
                  label=f'{category} centroid')
    
    ax.set_xlabel(f'PC1 ({best_pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({best_pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({best_pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title('Optimal Feature Combination PCA')
    plt.legend()
    
    # Save feature combination information
    with open('best_feature_combination.txt', 'w') as f:
        f.write(f"Best feature combination: {best_combination}\n")
        f.write(f"Score: {best_score}\n")
        f.write("\nFeature groups used:\n")
        for group in best_combination:
            f.write(f"- {group}: {feature_groups[group][0]} features\n")
        f.write(f"\nTotal features: {sum(feature_groups[group][0] for group in best_combination)}")
    
    def rotate(frame):
        ax.view_init(elev=20., azim=frame)
        return fig,
    
    anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2),
                                 interval=50, blit=True)
    anim.save('optimal_pca_3d_animation.gif', writer='pillow')
    plt.close()
    
    print("\nAnalysis complete! Generated files:")
    print("- optimal_pca_3d_animation.gif")
    print("- best_feature_combination.txt")

if __name__ == "__main__":
    main() 