import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2

def load_audio(audio_path):
    """
    Load and preprocess an audio file for spectrogram generation
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        torch.Tensor: Processed mel spectrogram
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return torch.zeros((1, 128, 1024), dtype=torch.float32)
    
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Handle empty or short waveforms
        if waveform.numel() == 0 or waveform.shape[1] < sample_rate * 0.5:  # Less than 0.5 seconds
            print(f"Warning: Very short or empty audio in {audio_path}")
            return torch.zeros((1, 128, 1024), dtype=torch.float32)
        
        # Normalize audio by RMS value
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 0:
            waveform = waveform / rms
        
        # Convert to mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        # Convert to decibels
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Normalize to match AST expectations
        mel_spec_db_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-10)
        
        # Ensure consistent shape for AST
        # Target shape: [1, 128, 1024]
        target_length = 1024
        current_length = mel_spec_db_norm.shape[2]
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            log_mel_spectrogram = torch.nn.functional.pad(mel_spec_db_norm, (0, padding))
        else:
            # Trim to target length
            log_mel_spectrogram = mel_spec_db_norm[:, :, :target_length]
        
        # Make sure we have the right dtype
        log_mel_spectrogram = log_mel_spectrogram.to(torch.float32)
        
        return log_mel_spectrogram
        
    except Exception as e:
        print(f"Error processing audio for {audio_path}: {str(e)}")
        # Return a dummy spectrogram with correct shape and dtype
        return torch.zeros((1, 128, 1024), dtype=torch.float32)

def generate_spectrogram(audio_path, output_dir=None, filename=None):
    """
    Generate and save a spectrogram from an audio file
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str, optional): Directory to save the spectrogram. Defaults to 'spectrograms' in current directory.
        filename (str, optional): Filename for the saved spectrogram. Defaults to timestamp + original filename.
        
    Returns:
        str: Path to the saved spectrogram image
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "spectrograms")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename if not provided
    if filename is None:
        audio_filename = os.path.basename(audio_path)
        base_name = os.path.splitext(audio_filename)[0]
        filename = f"{timestamp}_{base_name}.png"
    
    # Full path for the output file
    output_path = os.path.join(output_dir, filename)
    
    # Load original waveform for time domain plot
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
    
    # Normalize waveform for better visualization
    waveform = waveform / (waveform.abs().max() + 1e-10)
    
    # Load and process audio for spectrogram
    spectrogram = load_audio(audio_path)
    
    # Convert to numpy for visualization
    audio_np = spectrogram.squeeze().numpy()
    
    # Find the non-zero content in the spectrogram to focus visualization
    # This helps trim empty regions in the spectrogram
    non_zero_cols = np.where(np.sum(audio_np, axis=0) > 0.1)[0]
    
    # Create a more detailed visualization
    plt.figure(figsize=(12, 8))
    
    # Plot the waveform in time domain in the first subplot
    plt.subplot(2, 1, 1)
    waveform_np = waveform.numpy()
    time_axis = np.arange(0, waveform_np.shape[1]) / sample_rate
    plt.plot(time_axis, waveform_np[0])
    plt.title("Waveform (Time Domain)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot the spectrogram in the second subplot
    plt.subplot(2, 1, 2)
    
    # Calculate frames per second based on hop_length and sample_rate
    # hop_length = 512, sample_rate = 16000
    frames_per_second = 16000 / 512  # ~31.25 frames per second
    
    if len(non_zero_cols) > 0:
        # Find start of content with some padding
        start_col = max(0, non_zero_cols[0] - 5)
        
        # Calculate end column to show exactly 1 second of content
        # We want to show exactly one second of audio content
        end_col = min(audio_np.shape[1], start_col + int(frames_per_second))
        
        # Trim the spectrogram to focus on content
        audio_np_trimmed = audio_np[:, start_col:end_col]
        
        # Plot the trimmed spectrogram
        plt.imshow(audio_np_trimmed, aspect='auto', origin='lower')
        plt.title(f"Spectrogram (1 Second from Content Start)")
    else:
        # If no significant content found, show the first second
        end_col = min(audio_np.shape[1], int(frames_per_second))
        audio_np_trimmed = audio_np[:, 0:end_col]
        plt.imshow(audio_np_trimmed, aspect='auto', origin='lower')
        plt.title("Spectrogram (First 1 Second)")
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (mel bins)")
    
    # Add x-axis ticks in seconds
    num_frames = audio_np_trimmed.shape[1]
    plt.xticks(
        np.linspace(0, num_frames-1, 5),
        [f"{i:.1f}" for i in np.linspace(0, num_frames/frames_per_second, 5)]
    )
    plt.xlabel("Time (seconds)")
    
    plt.suptitle(f"Audio Analysis: {os.path.basename(audio_path)}")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    print(f"Saved audio analysis to: {output_path}")
    plt.close()
    
    return output_path

if __name__ == "__main__":
    # Hardcoded audio file path
    audio_path = "/home/dorry/Desktop/research/audio_visual_dataset_robo/audio/umass_tree8_robot_leaf2_segment_3_window_0_contact.wav"
    
    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        # Try an alternative path
        audio_path = "audio_visual_dataset/leaf_1.wav"
        if not os.path.exists(audio_path):
            print(f"Error: Alternative audio file not found: {audio_path}")
            print("Please update the hardcoded path in the script.")
            exit(1)
    
    # Generate the spectrogram
    output_dir = os.path.join(os.getcwd(), "spectrograms")
    generate_spectrogram(audio_path, output_dir)
    
    print(f"Spectrogram generation complete. Check the '{output_dir}' directory.")
