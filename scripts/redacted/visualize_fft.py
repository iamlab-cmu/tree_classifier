import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_fft_comparison(audio_path):
    # Load the audio file
    print(f"Loading audio file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Audio loaded: length={len(y)}, sample_rate={sr}")
    
    # Create the plot
    fig = plt.figure(figsize=(15, 10))
    
    # Time domain subplot
    plt.subplot(2, 2, 1)
    time = np.arange(len(y)) / sr
    plt.plot(time, y, 'b-', label='Signal')
    plt.title('Time Domain - Original')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Compute and plot spectrogram
    plt.subplot(2, 2, 2)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Manual spectrogram plotting to avoid librosa.display dependency
    plt.imshow(D_db, aspect='auto', origin='lower', 
               extent=[0, len(y)/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Original')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Apply window function
    print("Applying Hanning window")
    window = np.hanning(len(y))
    y_windowed = y * window
    
    # Time domain subplot (windowed)
    plt.subplot(2, 2, 3)
    plt.plot(time, y_windowed, 'b-', label='Windowed Signal')
    plt.title('Time Domain - After Windowing')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Compute and plot windowed spectrogram
    plt.subplot(2, 2, 4)
    D_windowed = librosa.stft(y_windowed, n_fft=2048, hop_length=512)
    D_windowed_db = librosa.amplitude_to_db(np.abs(D_windowed), ref=np.max)
    
    # Manual spectrogram plotting to avoid librosa.display dependency
    plt.imshow(D_windowed_db, aspect='auto', origin='lower', 
               extent=[0, len(y)/sr, 0, sr/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - After Windowing')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    plt.suptitle(f'Audio Analysis: {Path(audio_path).name}', size=14)
    plt.tight_layout()
    
    # Save the plot
    output_path = f'{Path(audio_path).stem}_fft_analysis.png'
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.show()
    print("Plot displayed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize FFT of an audio file')
    parser.add_argument('audio_file', help='Path to the audio file to analyze')
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' does not exist")
    else:
        print(f"Analyzing file: {args.audio_file}")
        plot_fft_comparison(args.audio_file) 