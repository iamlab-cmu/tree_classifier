import os
import sys
import argparse
import glob
import cv2
import numpy as np
import librosa
import shutil
from tqdm import tqdm
from cv_bridge import CvBridge
from collections import namedtuple
from contextlib import contextmanager
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip, ColorClip
from omegaconf import OmegaConf

# Add the scripts directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Now import from the local directory
from extract import RosBagExtractor
from segment_audio import segment_audio, get_audio_envelope

# Define the suppress_stdout context manager here, outside of any function
@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout

def find_bags_recursive(directory):
    """Find all .bag files recursively in the given directory."""
    return glob.glob(os.path.join(directory, '**/*.bag'), recursive=True)

# Define default values for missing config parameters
default_config = {
    'window_length_seconds': 1.0,
    'window_stride_seconds': 0.1,
    'non_contact_threshold_factor': 0.5,
    'enable_squeezing': True,
    'dynamic_threshold_offset': 0.15,
    'squeeze_duration': 0.3,
    'min_duration': 0.25  # Add minimum segment duration
}

# Create a function to safely get config values with defaults
def get_config(cfg, key, default):
    try:
        # First try data section
        if hasattr(cfg, 'data') and hasattr(cfg.data, key):
            return getattr(cfg.data, key)
        # Then try segmentation section
        elif hasattr(cfg, 'segmentation') and hasattr(cfg.segmentation, key):
            return getattr(cfg.segmentation, key)
        # Fall back to the default value
        else:
            return default
    except (AttributeError, KeyError):
        return default

def process_bag(bag_path, output_dir=None, fps=None, cfg=None, is_robot=True, no_visuals=False):
    """Process a single bag file."""
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    
    if output_dir:
        bag_output_dir = os.path.join(output_dir, bag_name)
    else:
        bag_output_dir = os.path.join('output', bag_name)
    
    print(f"Processing {os.path.basename(bag_path)}...", end=" ", flush=True)
    
    # Create the extractor
    extractor = RosBagExtractor(bag_path, bag_output_dir)
    
    # Try to extract images, skip if not available
    try:
        with suppress_stdout():
            extractor.extract_images()
    except Exception as e:
        print(f"\nSkipping {os.path.basename(bag_path)}: No image data")
        return
    
    # Try to extract audio, skip if not available
    try:
        with suppress_stdout():
            extractor.extract_audio()
    except Exception as e:
        print(f"\nSkipping {os.path.basename(bag_path)}: No audio data")
        return
    
    # Get the calculated FPS from the extractor
    calculated_fps = getattr(extractor, 'fps', None)
    
    # Use the calculated FPS or default to 30 if not available
    if calculated_fps is not None and calculated_fps > 0:
        fps = calculated_fps
    else:
        fps = 30.0
    
    print(f"(Using FPS: {fps:.1f}) ", end="", flush=True)
    
    print("Creating video...", end=" ", flush=True)
    
    # Get the image files
    image_dir = os.path.join(bag_output_dir, 'images')
    audio_file = os.path.join(bag_output_dir, 'audio', f'{bag_name}.wav')
    output_video = os.path.join(bag_output_dir, f'{bag_name}_video.mp4')
    
    # Check if we have images and audio
    if not os.path.exists(image_dir) or not os.path.exists(audio_file):
        print(f"\nError: Missing images or audio for {bag_name}")
        return
    
    # Get the image files
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if not image_files:
        print(f"\nError: No image files found in {image_dir}")
        return
    
    try:
        # If no config was passed, load it from file
        if cfg is None:
            # Load config from the default location
            config_path = os.path.join(script_dir, 'config', 'config.yaml')
            if os.path.exists(config_path):
                cfg = OmegaConf.load(config_path)
            else:
                # Create empty config if file doesn't exist
                cfg = OmegaConf.create({"data": {}, "segmentation": {}})
        
        # Load a sample image to get dimensions
        sample_img = cv2.imread(image_files[0])
        img_height, img_width = sample_img.shape[:2]
        
        # Load audio for waveform visualization
        audio_data, sr = librosa.load(audio_file, sr=None)
        
        # Get audio segments using config values
        window_length_seconds = get_config(cfg, 'window_length_seconds', default_config['window_length_seconds'])
        window_stride_seconds = get_config(cfg, 'window_stride_seconds', default_config['window_stride_seconds'])
        
        # Suppress stderr for segment_audio to avoid errors
        stderr_fd = sys.stderr.fileno()
        with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
            sys.stderr.flush()
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), stderr_fd)
                segments = segment_audio(
                    audio_file, 
                    window_length_seconds=get_config(cfg, 'window_length_seconds', default_config['window_length_seconds']),
                    window_stride_seconds=get_config(cfg, 'window_stride_seconds', default_config['window_stride_seconds']),
                    non_contact_threshold_factor=get_config(cfg, 'non_contact_threshold_factor', default_config['non_contact_threshold_factor']),
                    enable_squeezing=get_config(cfg, 'enable_squeezing', default_config['enable_squeezing']),
                    squeeze_factor_seconds=get_config(cfg, 'squeeze_duration', default_config['squeeze_duration']),
                    min_segment_duration=get_config(cfg, 'min_duration', default_config['min_duration']),
                    dynamic_threshold_offset=get_config(cfg, 'dynamic_threshold_offset', default_config['dynamic_threshold_offset']),
                    cfg=cfg,
                    is_robot=is_robot
                )
            finally:
                sys.stderr.flush()
                os.dup2(copied.fileno(), stderr_fd)
        
        # Get audio envelope for visualization
        envelope = get_audio_envelope(audio_data, frame_length=512, hop_length=128)
        window_size = int(0.02 * sr)
        envelope_smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Calculate dynamic threshold for visualization
        noise_floor = np.percentile(envelope_smoothed, 10)
        signal_peak = np.percentile(envelope_smoothed, 90)
        dynamic_threshold = noise_floor + (signal_peak - noise_floor) * get_config(cfg, 'dynamic_threshold_offset', default_config['dynamic_threshold_offset'])
        non_contact_threshold = dynamic_threshold * get_config(cfg, 'non_contact_threshold_factor', default_config['non_contact_threshold_factor'])
        
        # Calculate the appropriate duration based on the number of frames and FPS
        # This ensures the video plays at the correct speed
        duration = len(image_files) / fps
        
        # Create video clip from image sequence with the specified FPS
        video_clip = ImageSequenceClip(image_files, fps=fps)
        
        # Set the duration explicitly to match the calculated duration
        video_clip = video_clip.set_duration(duration)
        
        # Add audio
        audio_clip = AudioFileClip(audio_file)
        
        # Trim audio if it's longer than the video
        if audio_clip.duration > duration:
            audio_clip = audio_clip.subclip(0, duration)
        
        video_clip = video_clip.set_audio(audio_clip)
        
        # If no_visuals is True, skip waveform visualization
        if no_visuals:
            final_clip = video_clip
        else:
            # Create a separate clip for the audio waveform
            waveform_height = 100  # Height of the waveform visualization
            total_frames = len(image_files)
            
            def make_waveform_frame(t):
                # Create a blank frame for the waveform
                frame = np.ones((waveform_height, img_width, 3), dtype=np.uint8) * 255
                
                # Calculate which part of the audio to display
                frame_idx = int(t * fps)
                
                # Find the maximum amplitude in the audio data for scaling
                max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 1.0
                
                # Draw the waveform
                for i in range(img_width):
                    # Map pixel position to audio position
                    audio_pos = int((i / img_width) * len(audio_data))
                    if audio_pos < len(audio_data):
                        # Get amplitude and scale to fit the full height
                        # Scale from [-max_amplitude, max_amplitude] to [0, waveform_height]
                        normalized_amp = (audio_data[audio_pos] / max_amplitude)  # Now in range [-1, 1]
                        amplitude = int((0.5 - 0.5 * normalized_amp) * waveform_height)  # Map to [0, waveform_height]
                        
                        # Draw a vertical line for this sample
                        cv2.line(frame, 
                                (i, waveform_height // 2),  # Center point
                                (i, amplitude),  # Amplitude point
                                (0, 0, 255),  # Red color
                                1)  # Line thickness
                
                # Draw the envelope and thresholds
                for i in range(img_width):
                    env_pos = int((i / img_width) * len(envelope_smoothed))
                    if env_pos < len(envelope_smoothed):
                        # Scale envelope to fit in the frame
                        env_val = envelope_smoothed[env_pos] / max_amplitude
                        env_y = int((1.0 - env_val) * waveform_height)
                        # Ensure it's within bounds
                        env_y = max(0, min(waveform_height-1, env_y))
                        # Draw a point for the envelope
                        cv2.circle(frame, (i, env_y), 1, (0, 128, 0), -1)
                
                # Draw threshold lines
                threshold_y = int((1.0 - dynamic_threshold/max_amplitude) * waveform_height)
                threshold_y = max(0, min(waveform_height-1, threshold_y))
                cv2.line(frame, (0, threshold_y), (img_width, threshold_y), (0, 255, 0), 1)
                
                non_contact_y = int((1.0 - non_contact_threshold/max_amplitude) * waveform_height)
                non_contact_y = max(0, min(waveform_height-1, non_contact_y))
                cv2.line(frame, (0, non_contact_y), (img_width, non_contact_y), (255, 0, 255), 1)
                
                # Draw segment boundaries
                current_time = t
                current_sample = int(current_time * sr)
                
                # Draw all segment boundaries
                for i, (start, end, is_contact, _) in enumerate(segments):
                    # Convert sample positions to pixel positions
                    start_x = int((start / len(audio_data)) * img_width)
                    end_x = int((end / len(audio_data)) * img_width)
                    
                    # Choose color based on segment type
                    color = (255, 0, 0) if is_contact else (0, 0, 255)  # Red for contact, Blue for non-contact
                    
                    # Draw vertical lines at segment boundaries
                    cv2.line(frame, (start_x, 0), (start_x, waveform_height), color, 2)
                    cv2.line(frame, (end_x, 0), (end_x, waveform_height), color, 2)
                    
                    # Add segment label
                    label = f"S{i+1}"
                    cv2.putText(frame, label, (start_x + 5, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Highlight current segment
                    if start <= current_sample <= end:
                        # Draw a semi-transparent overlay for the current segment
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (start_x, 0), (end_x, waveform_height), 
                                     color, -1)  # Filled rectangle
                        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                        
                        # Add "Current" label
                        cv2.putText(frame, "CURRENT", (start_x + 5, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Add a progress indicator
                progress = t / duration if duration > 0 else 0
                progress_x = int(progress * img_width)
                cv2.line(frame, 
                        (progress_x, 0), 
                        (progress_x, waveform_height), 
                        (0, 255, 0),  # Green color
                        2)  # Line thickness
                
                # Add FPS indicator
                cv2.putText(frame, f"FPS: {fps:.1f}", (img_width - 80, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                return frame
            
            # Create a separate clip for the waveform with the same duration as the video
            waveform_clip = ColorClip(size=(img_width, waveform_height), 
                                     color=(0, 0, 0), 
                                     duration=duration)
            waveform_clip = waveform_clip.set_make_frame(make_waveform_frame)
            
            # Composite the main video and the waveform
            final_clip = CompositeVideoClip([
                video_clip,
                waveform_clip.set_position(('center', img_height))
            ], size=(img_width, img_height + waveform_height))
        
        # Write the result with minimal logging
        with suppress_stdout():
            final_clip.write_videofile(
                output_video,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(bag_output_dir, 'temp-audio.m4a'),
                remove_temp=True,
                verbose=False,
                logger=None,
                fps=fps  # Explicitly set the FPS for the output video
            )
        print("Done")
        
    except Exception as e:
        print(f"\nError creating video: {str(e)}")

def main():
    # Simple command line argument parsing
    parser = argparse.ArgumentParser(description='Extract data from ROS bags')
    parser.add_argument('path', help='Path to ROS bag file or directory containing ROS bags')
    parser.add_argument('-o', '--output', help='Output directory for extracted files', default='output')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search for bag files in subdirectories')
    parser.add_argument('--robot', action='store_true', help='Specify if processing robot data (default: true)', default=True)
    parser.add_argument('--probe', action='store_true', help='Specify if processing probe data')
    parser.add_argument('--no-visuals', action='store_true', help='Generate video without waveform visualizations')

    args = parser.parse_args()
    
    # If probe is specified, override the robot flag
    is_robot = not args.probe
    
    # Load config directly
    config_path = os.path.join(script_dir, 'config', 'config.yaml')
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        # Create empty config if file doesn't exist
        cfg = OmegaConf.create({"data": {}, "segmentation": {}})

    if os.path.isfile(args.path):
        process_bag(args.path, args.output, cfg=cfg, is_robot=is_robot, no_visuals=args.no_visuals)
    elif os.path.isdir(args.path):
        if args.recursive:
            bag_files = find_bags_recursive(args.path)
        else:
            bag_files = [os.path.join(args.path, f) for f in os.listdir(args.path) 
                       if f.endswith('.bag')]
        
        if not bag_files:
            print(f"No bag files found in {args.path}")
            return

        for bag_path in bag_files:
            process_bag(bag_path, args.output, cfg=cfg, is_robot=is_robot, no_visuals=args.no_visuals)
    else:
        print(f"Error: Path not found: {args.path}")

if __name__ == '__main__':
    main()