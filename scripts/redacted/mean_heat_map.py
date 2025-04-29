import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio

# Function to extract features from audio file
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
    
    # Combine features into a single flat array
    features = np.concatenate([[rms, zcr, spec_cent, spec_bw], mfccs, chroma, spectral_contrast, tonnetz])
    return features

def extract_features_full(audio_path):
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

def extract_features_reduced(audio_path):
    y, sr = librosa.load(audio_path)
    
    rms = librosa.feature.rms(y=y)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4).mean(axis=1)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)[-3:]
    
    features = np.concatenate([[rms, zcr, spec_cent, spec_bw], mfccs, tonnetz])
    return features

categories = ['ambient', 'leaf', 'twig', 'trunk']
X = []
y = []
filenames = []

for category in categories:
    category_path = os.path.join('./sounds', category)
    for file in os.listdir(category_path):
        if file.endswith('.wav'):
            file_path = os.path.join(category_path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(category)
            filenames.append(file)

X = np.array(X)
y = np.array(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create feature names list
feature_names = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + \
                [f'MFCC_{i+1}' for i in range(13)] + \
                [f'Chroma_{i+1}' for i in range(12)] + \
                [f'Spectral_Contrast_{i+1}' for i in range(7)] + \
                [f'Tonnetz_{i+1}' for i in range(6)]

# Calculate and display mean features for each category using scaled features
print("Mean features by category:")
mean_features_by_category = {}
for category in categories:
    mask = y == category
    category_features = X_scaled[mask]  # Using scaled features for better visualization
    mean_features = category_features.mean(axis=0)
    mean_features_by_category[category] = mean_features
    
    print(f"\n{category.upper()}:")
    for name, value in zip(feature_names, mean_features):
        print(f"{name}: {value:.3f}")

# Create a heatmap of mean features using scaled values, excluding ambient
plt.figure(figsize=(15, 10))
categories_no_ambient = [cat for cat in categories if cat != 'ambient']
mean_features_matrix = np.array([mean_features_by_category[cat] for cat in categories_no_ambient])
sns.heatmap(mean_features_matrix, 
            xticklabels=feature_names, 
            yticklabels=categories_no_ambient,
            cmap='coolwarm',
            center=0,
            vmin=-2, vmax=2)  # Setting limits for better color contrast
plt.xticks(rotation=45, ha='right')
plt.title('Normalized Mean Feature Values by Category')
plt.tight_layout()
plt.show()

# Calculate feature importance using PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("\nFeature importance based on PCA:")
for i, component in enumerate(pca.components_):
    print(f"\nPrincipal Component {i+1} (Explains {pca.explained_variance_ratio_[i]:.2%} of variance):")
    feature_importance = list(zip(feature_names, component))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, weight in feature_importance[:5]:
        print(f"{feat}: {weight:.3f}")

# Calculate and print total explained variance
total_variance = sum(pca.explained_variance_ratio_)
print(f"\nTotal explained variance by first 3 components: {total_variance:.2%}")

# Create two separate feature sets
X_full = []
X_reduced = []
y = []
categories = ['leaf', 'twig', 'trunk']

for category in categories:
    category_path = os.path.join('./sounds', category)
    for file in os.listdir(category_path):
        if file.endswith('.wav'):
            file_path = os.path.join(category_path, file)
            X_full.append(extract_features_full(file_path))
            X_reduced.append(extract_features_reduced(file_path))
            y.append(category)

X_full = np.array(X_full)
X_reduced = np.array(X_reduced)
y = np.array(y)

# Create feature names lists
feature_names_full = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + \
                    [f'MFCC_{i+1}' for i in range(13)] + \
                    [f'Chroma_{i+1}' for i in range(12)] + \
                    [f'Spectral_Contrast_{i+1}' for i in range(7)] + \
                    [f'Tonnetz_{i+1}' for i in range(6)]

feature_names_reduced = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + \
                       [f'MFCC_{i+1}' for i in range(4)] + \
                       [f'Tonnetz_{i+1}' for i in range(4, 7)]

# Scale both feature sets
scaler_full = StandardScaler()
scaler_reduced = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)
X_reduced_scaled = scaler_reduced.fit_transform(X_reduced)

# Calculate mean features for both sets
mean_features_full = {}
mean_features_reduced = {}

for category in categories:
    mask = y == category
    mean_features_full[category] = X_full_scaled[mask].mean(axis=0)
    mean_features_reduced[category] = X_reduced_scaled[mask].mean(axis=0)

# Create side-by-side heatmaps
plt.figure(figsize=(20, 8))

# Full features heatmap
plt.subplot(1, 2, 1)
mean_features_matrix_full = np.array([mean_features_full[cat] for cat in categories])
sns.heatmap(mean_features_matrix_full, 
            xticklabels=feature_names_full, 
            yticklabels=categories,
            cmap='coolwarm',
            center=0,
            vmin=-2, vmax=2)
plt.xticks(rotation=45, ha='right')
plt.title('Full Feature Set')

# Reduced features heatmap
plt.subplot(1, 2, 2)
mean_features_matrix_reduced = np.array([mean_features_reduced[cat] for cat in categories])
sns.heatmap(mean_features_matrix_reduced, 
            xticklabels=feature_names_reduced, 
            yticklabels=categories,
            cmap='coolwarm',
            center=0,
            vmin=-2, vmax=2)
plt.xticks(rotation=45, ha='right')
plt.title('Reduced Feature Set')

plt.tight_layout()
plt.show()

# Calculate and compare PCA results for both sets
pca_full = PCA(n_components=3)
pca_reduced = PCA(n_components=3)

X_pca_full = pca_full.fit_transform(X_full_scaled)
X_pca_reduced = pca_reduced.fit_transform(X_reduced_scaled)

print("\nFull feature set - Total explained variance:", sum(pca_full.explained_variance_ratio_).round(4))
print("Reduced feature set - Total explained variance:", sum(pca_reduced.explained_variance_ratio_).round(4))
