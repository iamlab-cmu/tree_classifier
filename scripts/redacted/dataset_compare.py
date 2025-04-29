import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Fix for older matplotlib versions
import warnings
warnings.filterwarnings("ignore")

# Import matplotlib and patch it before importing librosa
import matplotlib
if not hasattr(matplotlib, 'colormaps'):
    # For older matplotlib versions, create a compatible interface
    matplotlib.colormaps = matplotlib.cm
    # Make it subscriptable
    def _get_cmap(name):
        return matplotlib.cm.get_cmap(name)
    matplotlib.colormaps.__getitem__ = _get_cmap

# Now import librosa
import librosa
import librosa.display

def plot_audio_and_spectrogram(y, sr, ax_wave, ax_spec, title=None, n_fft=2048, hop_length=512):
    """
    Plot the waveform and spectrogram of an audio signal on the provided axes.
    
    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        ax_wave: Matplotlib axis for waveform
        ax_spec: Matplotlib axis for spectrogram
        title (str, optional): Title for the plots
        n_fft (int): FFT window size for spectrogram
        hop_length (int): Hop length for spectrogram
    """
    # Plot waveform
    times = np.arange(len(y)) / float(sr)
    ax_wave.plot(times, y)
    ax_wave.set_title("Waveform" if title is None else f"{title} - Waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    
    # Plot spectrogram
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Use matplotlib's imshow instead of librosa's specshow
    img = ax_spec.imshow(D_db, aspect='auto', origin='lower', 
                    extent=[0, len(y)/sr, 0, sr/2], cmap='magma')
    ax_spec.set_title("Spectrogram" if title is None else f"{title} - Spectrogram")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (s)")
    
    return img

def create_grid_visualization(input_dir, output_path, sr=None, n_files=None, grid_cols=2, figsize=(15, 10)):
    """
    Create a single grid visualization for all audio files in the input directory.
    
    Args:
        input_dir (str): Directory containing audio files
        output_path (str): Path to save the visualization
        sr (int, optional): Sample rate for loading the audio. If None, uses the file's sample rate.
        n_files (int, optional): Maximum number of files to process. If None, process all files.
        grid_cols (int): Number of columns in the grid
        figsize (tuple): Base figure size (width, height) - will be scaled based on number of files
    """
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    
    if n_files is not None:
        audio_files = audio_files[:n_files]
    
    n_files = len(audio_files)
    print(f"Found {n_files} audio files")
    
    if n_files == 0:
        print("No audio files found in the directory")
        return
    
    # Calculate grid dimensions
    grid_rows = (n_files + grid_cols - 1) // grid_cols  # Ceiling division
    
    # Create figure with appropriate size
    # Scale figure size based on number of rows and columns
    fig_width = figsize[0] * (grid_cols / 2)
    fig_height = figsize[1] * (grid_rows / 2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create a grid of subplots - 2 rows per audio file (waveform and spectrogram)
    gs = GridSpec(grid_rows * 2, grid_cols)
    
    # Process each audio file
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        # Calculate row and column position
        row = (i // grid_cols) * 2  # Multiply by 2 because each file takes 2 rows
        col = i % grid_cols
        
        # Load audio file
        audio_path = os.path.join(input_dir, audio_file)
        y, file_sr = librosa.load(audio_path, sr=sr)
        
        # Get filename for title
        filename = os.path.splitext(audio_file)[0]
        
        # Create axes for this audio file
        ax_wave = fig.add_subplot(gs[row, col])
        ax_spec = fig.add_subplot(gs[row + 1, col])
        
        # Plot waveform and spectrogram
        img = plot_audio_and_spectrogram(y, file_sr, ax_wave, ax_spec, title=filename)
    
    # Add a colorbar for the last spectrogram
    cbar = fig.colorbar(img, ax=fig.axes[-1], format="%+2.0f dB")
    cbar.set_label('Amplitude (dB)')
    
    plt.suptitle("Audio Dataset Visualization", fontsize=16)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grid visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create visualizations for audio files")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--output_path", default="visualizations/audio_grid.png", help="Path to save the visualization")
    parser.add_argument("--sr", type=int, default=None, help="Sample rate for loading audio")
    parser.add_argument("--n_files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--grid_cols", type=int, default=2, help="Number of columns in the grid")
    
    args = parser.parse_args()
    
    # Create grid visualization
    create_grid_visualization(
        args.input_dir, 
        args.output_path, 
        sr=args.sr, 
        n_files=args.n_files,
        grid_cols=args.grid_cols
    )

if __name__ == "__main__":
    main()
