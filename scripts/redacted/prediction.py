from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Combine features
    features = np.concatenate([[rms, zcr, spec_cent, spec_bw], mfccs])
    return features

# Group visualization functions and constants
def create_3d_pca_plot(X_pca, y, categories, pca):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#66B2FF', '#99FF99', '#FFCC99']
    for category, color in zip(categories, colors):
        mask = y == category
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                  label=category, alpha=0.7, c=color)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title('Audio Categories Visualization (3D PCA)')
    plt.legend()
    return fig, ax

categories = ['leaf', 'twig', 'trunk']
X = []
y = []
filenames = []

for category in categories:
    category_path = os.path.join('./output', category)
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

# Apply PCA with 3 components
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Create 3D plot with animation
fig, ax = create_3d_pca_plot(X_pca, y, categories, pca)

# Animation function
def rotate(frame):
    ax.view_init(elev=20., azim=frame)
    return fig,

# Create animation
anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), 
                    interval=50, blit=True)

# Save as GIF
anim.save('3d_plot_rotation.gif', writer='pillow')

plt.show()

# Print explained variance ratios
print("\nExplained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.2%}")

# Feature analysis
feature_names = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + [f'MFCC_{i+1}' for i in range(13)]

# Calculate mean features for each category
print("\nMean features by category:")
for category in categories:
    mask = y == category
    category_features = X_scaled[mask]
    mean_features = category_features.mean(axis=0)
    print(f"\n{category.upper()}:")
    for name, value in zip(feature_names[:6], mean_features[:6]): 
        print(f"{name}: {value:.3f}")
true_labels = y
true_labels = y

X_for_clustering = X_scaled

inertias = []
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_for_clustering)
    
    inertias.append(kmeans.inertia_)
    
    silhouette_scores.append(silhouette_score(X_for_clustering, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K, inertias, 'bx-')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(K, silhouette_scores, 'rx-')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.tight_layout()
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_for_clustering)

categories_unique = np.unique(true_labels)
cluster_category_counts = np.zeros((optimal_k, len(categories_unique)))

for i in range(optimal_k):
    for j, category in enumerate(categories_unique):
        cluster_category_counts[i, j] = np.sum(
            (cluster_labels == i) & (true_labels == category)
        )

plt.figure(figsize=(10, 8))
sns.heatmap(cluster_category_counts, 
            xticklabels=categories_unique,
            yticklabels=[f'Cluster {i+1}' for i in range(optimal_k)],
            annot=True, 
            fmt='g',
            cmap='YlOrRd')
plt.title('Cluster vs Category Distribution')
plt.xlabel('True Categories')
plt.ylabel('Clusters')
plt.show()

for i in range(optimal_k):
    cluster_mask = cluster_labels == i
    cluster_features = X_for_clustering[cluster_mask]
    
    cluster_size = np.sum(cluster_mask)
    cluster_percentage = (cluster_size / len(X_for_clustering)) * 100
    
    print(f"\nCluster {i+1}:")
    print(f"Size: {cluster_size} samples ({cluster_percentage:.1f}% of total)")
    print("\nCategory distribution:")
    for category in categories_unique:
        category_count = np.sum((true_labels == category) & cluster_mask)
        if cluster_size > 0:
            category_percentage = (category_count / cluster_size) * 100
            print(f"{category}: {category_count} samples ({category_percentage:.1f}%)")
    
    centroid = kmeans.cluster_centers_[i]
    feature_names = ['RMS', 'ZCR', 'Spectral Centroid', 'Spectral Bandwidth'] + [f'MFCC_{i+1}' for i in range(13)]
    
    feature_importance = np.abs(centroid)
    top_indices = np.argsort(feature_importance)[-5:]
    print("\nTop features at centroid:")
    for idx in reversed(top_indices):
        print(f"{feature_names[idx]}: {centroid[idx]:.3f}")

silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
print(f"\nOverall Clustering Metrics:")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Inertia: {kmeans.inertia_:.3f}")

def purity_score(y_true, y_pred):
    contingency_matrix = cluster_category_counts
    return np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

purity = purity_score(true_labels, cluster_labels)
print(f"Cluster Purity: {purity:.3f}")