import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import soundfile as sf
import noisereduce as nr

def analyze_audio(audio_path, noise_path=None, output_dir=None, apply_noise_reduction=False):
    """
    Simplified audio analysis tool that shows:
    1. Original waveform
    2. Noise-reduced waveform (if requested)
    
    Args:
        audio_path (str): Path to the audio file to analyze
        noise_path (str): Path to noise sample for noise reduction (optional)
        output_dir (str): Directory to save processed audio (optional)
        apply_noise_reduction (bool): Whether to apply noise reduction
    """
    # Load the audio file
    print(f"Loading audio file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Audio loaded: length={len(y)}, sample_rate={sr}")
    
    # Apply noise reduction if requested
    y_processed = y.copy()
    output_filename = None
    
    if apply_noise_reduction and noise_path:
        print(f"Applying noise reduction using noise sample: {noise_path}")
        # Load the noise sample
        noise_y, noise_sr = librosa.load(noise_path)
        
        # Ensure both audio files have the same sample rate
        if sr != noise_sr:
            print(f"Resampling noise from {noise_sr} Hz to {sr} Hz")
            noise_y = librosa.resample(noise_y, orig_sr=noise_sr, target_sr=sr)
        
        # Apply noise reduction
        y_processed = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_y,
            prop_decrease=1,
            stationary=False
        )
        
        # Save processed audio
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"noise_reduced_{os.path.basename(audio_path)}")
        else:
            # If no output directory specified, save in the current directory
            output_filename = f"noise_reduced_{os.path.basename(audio_path)}"
        
        sf.write(output_filename, y_processed, sr)
        print(f"Saved noise-reduced audio to: {output_filename}")
    
    # Create time axis for plotting
    time = np.arange(len(y)) / sr
    
    # Create the figure with appropriate size
    if apply_noise_reduction:
        # Two waveforms side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original waveform
        ax1.plot(time, y, 'b-')
        ax1.set_title('Original Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Noise-reduced waveform
        ax2.plot(time, y_processed, 'g-')
        ax2.set_title('Noise-Reduced Waveform')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
    else:
        # Just one waveform
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, y, 'b-')
        ax.set_title('Audio Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    
    plt.suptitle(f'Audio Analysis: {Path(audio_path).name}', size=14)
    plt.tight_layout()
    
    # Save the plot
    output_path = f'{Path(audio_path).stem}_waveform_analysis.png'
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.show()
    print("Plot displayed")
    
    return output_filename

if __name__ == "__main__":
    # HARD-CODED PATHS FOR TESTING - UNCOMMENT AND MODIFY THESE LINES
    audio_file = "RoboOutput/umass_tree1_robot_leaf2/audio/umass_tree1_robot_leaf2.wav"
    noise_file = "./robo_humming.wav"  # Optional, set to None if not using noise reduction
    output_dir = "./processed_audio"
    
    # Run with hard-coded paths
    if os.path.exists(audio_file):
        output_file = analyze_audio(
            audio_path=audio_file,
            noise_path=noise_file if noise_file and os.path.exists(noise_file) else None,
            output_dir=output_dir,
            apply_noise_reduction=True if noise_file and os.path.exists(noise_file) else False
        )
        
        if output_file:
            print(f"Processed audio saved to: {output_file}")
    else:
        print(f"Error: Audio file '{audio_file}' does not exist") 