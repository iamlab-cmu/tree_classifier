import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import argparse

def remove_noise_and_plot(audio_path, noise_path, output_dir=None):
    """
    Remove noise from a single audio file and plot FFT comparison.
    
    Args:
        audio_path (str): Path to the audio file to process
        noise_path (str): Path to the noise sample file
        output_dir (str): Directory to save the processed file (optional)
    
    Returns:
        str: Path to the processed audio file
    """
    print(f"Processing file: {audio_path}")
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    print(f"Loaded audio: {len(y)/sr:.2f} seconds, {sr} Hz")
    
    # Load the noise sample
    noise_y, noise_sr = librosa.load(noise_path)
    print(f"Loaded noise sample: {len(noise_y)/noise_sr:.2f} seconds, {noise_sr} Hz")
    
    # Ensure both audio files have the same sample rate
    if sr != noise_sr:
        print(f"Resampling noise from {noise_sr} Hz to {sr} Hz")
        noise_y = librosa.resample(noise_y, orig_sr=noise_sr, target_sr=sr)
    
    # Apply noise reduction
    print("Applying noise reduction...")
    reduced_noise = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_y,
        prop_decrease=1.0,
        stationary=False
    )
    
    # Save processed audio if output directory is provided
    output_filename = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, os.path.basename(audio_path))
        sf.write(output_filename, reduced_noise, sr)
        print(f"Saved processed audio to: {output_filename}")
    
    # Create plots directory
    plots_dir = 'preprocessing_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot time domain comparison
    plt.figure(figsize=(15, 10))
    
    # Time domain plots
    plt.subplot(2, 2, 1)
    time = np.arange(len(y)) / sr
    plt.plot(time, y)
    plt.title('Original Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(time, reduced_noise)
    plt.title('Noise-Reduced Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Calculate FFTs
    fft_original = np.fft.fft(y)
    fft_cleaned = np.fft.fft(reduced_noise)
    freqs = np.fft.fftfreq(len(y), 1/sr)
    
    # Plot FFT comparison
    plt.subplot(2, 2, 3)
    plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_original[:len(freqs)//2]) + 1e-10))
    plt.title('FFT of Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_cleaned[:len(freqs)//2]) + 1e-10))
    plt.title('FFT of Noise-Reduced Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    
    plt.suptitle(f'Noise Reduction Analysis: {os.path.basename(audio_path)}', size=14)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(plots_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_noise_reduction.png")
    plt.savefig(plot_filename)
    print(f"Saved analysis plot to: {plot_filename}")
    
    plt.show()
    
    return output_filename

if __name__ == "__main__":
    # HARD-CODED PATHS FOR TESTING
    # Uncomment and modify these lines to use hard-coded paths
    # --------------------------------------------------------
    # audio_file = "/path/to/your/audio_file.wav"
    # noise_file = "./humming.wav"
    # output_dir = "./processed"
    # 
    # if os.path.exists(audio_file) and os.path.exists(noise_file):
    #     remove_noise_and_plot(audio_file, noise_file, output_dir)
    #     exit(0)
    # --------------------------------------------------------
    
    # Command-line argument parsing (used if hard-coded paths are commented out)
    parser = argparse.ArgumentParser(description='Process a single audio file to remove noise and plot results')
    parser.add_argument('audio_file', help='Path to the audio file to process')
    parser.add_argument('--noise', '-n', default='./humming.wav', 
                      help='Path to the noise sample file (default: ./humming.wav)')
    parser.add_argument('--output-dir', '-o', default=None,
                      help='Directory to save the processed audio file (optional)')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' does not exist")
        exit(1)
    
    if not os.path.exists(args.noise):
        print(f"Error: Noise file '{args.noise}' does not exist")
        exit(1)
    
    # Process the single file
    remove_noise_and_plot(args.audio_file, args.noise, args.output_dir) 