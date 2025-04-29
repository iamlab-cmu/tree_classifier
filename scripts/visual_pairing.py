import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
import torchaudio
import torch
import cv2

# Import matplotlib and set up colormaps compatibility for older versions
import matplotlib
if not hasattr(matplotlib, 'colormaps'):
    matplotlib.colormaps = matplotlib.cm
    
# Now try to import librosa
try:
    import librosa
    import librosa.display
except ImportError as e:
    # Special handling for older matplotlib versions without 'colormaps'
    if "cannot import name 'colormaps'" in str(e):
        # Patch librosa.display before importing
        import sys
        import types
        
        # Create a mock module for librosa.display
        mock_display = types.ModuleType('librosa.display')
        sys.modules['librosa.display'] = mock_display
        
        # Only then import librosa
        import librosa
        
        # Define minimal display functions needed
        def specshow(data, **kwargs):
            return plt.imshow(data, origin='lower', aspect='auto', **kwargs)
        
        # Add the function to the mock module
        mock_display.specshow = specshow
    else:
        raise e

from matplotlib.gridspec import GridSpec

def create_spectrogram(audio_path, disable_normalization=True):
    """
    Generate a spectrogram from an audio file while preserving amplitude differences
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Only normalize by RMS if explicitly requested
        if not disable_normalization:
            rms = torch.sqrt(torch.mean(waveform**2))
            if rms > 0:
                waveform = waveform / rms
        
        # Print audio stats to help debug
        print(f"  Audio stats: min={waveform.min().item():.6f}, max={waveform.max().item():.6f}, rms={torch.sqrt(torch.mean(waveform**2)).item():.6f}")
        
        # Convert to mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        # Convert to decibels with a reference value
        # Using amplitude_to_DB maintains relative differences better than plain log
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Skip normalization to preserve amplitude differences
        if disable_normalization:
            spectrogram = mel_spec_db.numpy().squeeze()
        else:
            # Apply normalization if requested (original behavior)
            mel_spec_db_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-10)
            spectrogram = mel_spec_db_norm.numpy().squeeze()
        
        return spectrogram
        
    except Exception as e:
        print(f"Error processing audio {audio_path}: {str(e)}")
        # Return an empty spectrogram
        return np.zeros((128, 1024))

def load_and_square_image(image_path):
    """
    Load an image and make it square by center cropping, with a black border
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to read image: {image_path}")
            return None
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make square by center cropping
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # Calculate crop coordinates
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        
        # Crop to square
        square_img = image[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Add a black border to the image (3 pixels wide)
        border_size = 3
        square_img_with_border = cv2.copyMakeBorder(
            square_img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black border
        )
        
        return square_img_with_border
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    # Use the correct dataset path
    dataset_path = "/home/dorry/Desktop/research/audio_visual_dataset_no_norm/dataset.csv"
    dataset_dir = "/home/dorry/Desktop/research/audio_visual_dataset_no_norm"
    
    # Read the dataset
    dataset = pd.read_csv(dataset_path)
    
    # Use the correct column names
    audio_col = 'audio_file'
    image_col = 'image_file'
    class_column = 'category'
    
    # Set desired order of categories
    ordered_categories = ['leaf', 'twig', 'trunk', 'ambient']
    
    # Get available categories from dataset
    available_categories = dataset[class_column].unique()
    
    # Filter to only include categories that are actually in the dataset
    # and in the desired order
    unique_classes = [cat for cat in ordered_categories if cat in available_categories]
    
    # Add any categories that are in the dataset but not in our ordered list
    for cat in available_categories:
        if cat not in unique_classes:
            unique_classes.append(cat)
    
    # If there are more than 4 categories, just use the first 4
    if len(unique_classes) > 4:
        unique_classes = unique_classes[:4]
        print(f"Using these 4 categories: {unique_classes}")
    # If there are fewer than 4 categories, we'll work with what we have
    elif len(unique_classes) < 4:
        print(f"Warning: Only {len(unique_classes)} categories found: {unique_classes}")
    
    num_classes = len(unique_classes)
    samples_per_class = 4  # We want 4 samples per category/class
    
    # Create figure for visualization with a white background
    fig = plt.figure(figsize=(5*num_classes, 4*samples_per_class))
    
    # Create a grid with 4 rows (samples) and num_classes columns
    outer_gs = GridSpec(samples_per_class, num_classes, figure=fig, 
                       hspace=0.3, wspace=0.3)
    
    # Reset to default style (white background)
    plt.rcdefaults()
    
    # For each category, get 4 random pairs of images and audio files
    for class_idx, class_name in enumerate(unique_classes):
        class_data = dataset[dataset[class_column] == class_name]
        
        # Randomly select 4 samples (or fewer if there aren't enough)
        sample_indices = list(range(len(class_data)))
        if len(sample_indices) > samples_per_class:
            # Randomly select samples_per_class indices
            sample_indices = random.sample(sample_indices, samples_per_class)
        
        # Add class title at the top of each column
        fig.text(0.125 + 0.25*class_idx, 0.95, class_name, 
                fontsize=16, ha='center', va='center')
        
        for i, sample_idx in enumerate(sample_indices):
            if i < samples_per_class:  # Just a safeguard
                # Use full paths from the dataset
                image_path = os.path.join(dataset_dir, class_data.iloc[sample_idx][image_col])
                audio_path = os.path.join(dataset_dir, class_data.iloc[sample_idx][audio_col])
                
                # Print path information for each sample
                print(f"Class: {class_name}, Sample {i+1}:")
                print(f"  Image: {image_path}")
                print(f"  Audio: {audio_path}")
                
                # Create a cell in our grid for this sample with equal width columns
                cell = outer_gs[i, class_idx].subgridspec(1, 2, wspace=0.1, width_ratios=[1, 1])
                
                # Image subplot (left half of the cell)
                ax_img = fig.add_subplot(cell[0, 0])
                
                if os.path.exists(image_path):
                    # Load and square the image
                    square_img = load_and_square_image(image_path)
                    if square_img is not None:
                        # Display the squared image
                        img = ax_img.imshow(square_img)
                        
                        # Add a specific border around the image itself
                        from matplotlib.patches import Rectangle
                        height, width = square_img.shape[:2]
                        rect = Rectangle(
                            (-0.5, -0.5),  # Starting position (lower left corner)
                            width + 0.99,  # Width with a small adjustment
                            height + 0.99,  # Height with a small adjustment
                            fill=False,
                            edgecolor='black',
                            linewidth=3
                        )
                        ax_img.add_patch(rect)
                    else:
                        # If image processing failed, show a placeholder
                        ax_img.text(0.5, 0.5, "Image processing failed", ha='center', va='center', fontsize=8)
                    
                    ax_img.set_title("Image", fontsize=10)
                    ax_img.axis('off')
                else:
                    ax_img.text(0.5, 0.5, f"Image not found", ha='center', va='center', fontsize=8)
                    ax_img.axis('off')
                    print(f"Image not found: {image_path}")
                
                # Spectrogram subplot (right half of the cell)
                ax_spec = fig.add_subplot(cell[0, 1])
                
                if os.path.exists(audio_path):
                    try:
                        # Generate spectrogram without normalization to show amplitude differences
                        spectrogram = create_spectrogram(audio_path, disable_normalization=True)
                        
                        # Display the spectrogram with a fixed color range to ensure comparable visualizations
                        img_spec = ax_spec.imshow(spectrogram, aspect='auto', origin='lower', 
                                                  cmap='viridis', vmin=-80, vmax=0)
                        
                        # Set simple title without RMS value
                        ax_spec.set_title('Spectrogram', fontsize=10)
                        
                        # Remove all axes ticks, numbers and labels
                        ax_spec.set_xticks([])
                        ax_spec.set_yticks([])
                        ax_spec.set_xlabel('')
                        ax_spec.set_ylabel('')
                    except Exception as e:
                        ax_spec.text(0.5, 0.5, f"Error creating spectrogram", ha='center', va='center', fontsize=8)
                        ax_spec.axis('off')
                        print(f"Error with {audio_path}: {e}")
                else:
                    ax_spec.text(0.5, 0.5, f"Audio not found", ha='center', va='center', fontsize=8)
                    ax_spec.axis('off')
                    print(f"Audio not found: {audio_path}")
                
                # Make image square
                ax_img.set_aspect('equal')
                
                # Alternative to set_box_aspect for older matplotlib versions
                # This removes extra whitespace around the plots
                ax_img.set_adjustable('box')
                ax_spec.set_adjustable('box')
                
                # Add borders with different widths for image and spectrogram
                # Image gets 3px border as before
                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(3)  # Keep original thickness for image
                
                # Spectrogram gets a thinner border (1.5px)
                for spine in ax_spec.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.5)  # Half the thickness for spectrogram
                
                # Set frame properties for both subplots
                for ax in [ax_img, ax_spec]:
                    # Remove any padding that might hide the border
                    ax.set_frame_on(True)
                    ax.patch.set_edgecolor('black')
                
                # Set different linewidths for the patch borders too
                ax_img.patch.set_linewidth(3)
                ax_spec.patch.set_linewidth(1.5)  # Half the thickness
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room at the top for titles
    
    # Create output directory if it doesn't exist
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, "class_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Optionally display the figure
    plt.show()

if __name__ == "__main__":
    main()