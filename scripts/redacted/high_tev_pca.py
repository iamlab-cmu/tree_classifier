import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Function to extract only the features shown in the heatmap
def extract_reduced_features(audio_path):
    y, sr = librosa.load(audio_path)
    
    # Basic features
    rms = librosa.feature.rms(y=y)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    
    # Get first 4 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4).mean(axis=1)
    
    # Get Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)
    
    # Combine features into a single flat array
    features = np.concatenate([[rms, zcr, spec_cent, spec_bw], mfccs, tonnetz])
    return features

# Load and process data
categories = ['ambient', 'leaf', 'twig', 'trunk']
X = []
y = []

for category in categories:
    category_path = os.path.join('./output', category)
    for file in os.listdir(category_path):
        if file.endswith('.wav'):
            file_path = os.path.join(category_path, file)
            features = extract_reduced_features(file_path)
            X.append(features)
            y.append(category)

X = np.array(X)
y = np.array(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each category with different colors
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
for category, color in zip(categories, colors):
    mask = y == category
    category_points = X_pca[mask]
    
    # Plot points
    ax.scatter(category_points[:, 0], category_points[:, 1], category_points[:, 2], 
              label=category, alpha=0.7, c=color)
    
    # Calculate and plot centroid
    centroid = np.mean(category_points, axis=0)
    ax.scatter(centroid[0], centroid[1], centroid[2], 
              c=color, marker='*', s=200, edgecolor='black', linewidth=1,
              label=f'{category} centroid')

# Add labels and title
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
ax.set_title('Audio Categories Visualization (3D PCA) - Reduced Feature Set')
plt.legend()

# Animation function
def rotate(frame):
    ax.view_init(elev=30., azim=frame)  # Changed elev for better perspective
    return fig,

# Create animation with smoother rotation
anim = FuncAnimation(fig, rotate, 
                    frames=np.arange(0, 360, 2),  # 2-degree steps for smoother rotation
                    interval=30,  # Faster animation (30ms between frames)
                    blit=True)

# Save as GIF with higher quality
anim.save('3d_plot_high_tev.gif', 
          writer='pillow',
          fps=30)  # Higher FPS for smoother animation

plt.show()

# Print explained variance information
print("\nExplained variance ratios:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))