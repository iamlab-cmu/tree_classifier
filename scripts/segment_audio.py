import librosa
import numpy as np
import soundfile as sf
import os
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import noisereduce as nr
import tempfile
import hydra
import sys
import glob
import matplotlib.pyplot as plt


def get_audio_envelope(y, frame_length=2048, hop_length=512):
    if len(y) < frame_length:
        frame_length = len(y)
        hop_length = frame_length // 4

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    rms_interpolated = np.interp(
        np.arange(len(y)), np.linspace(0, len(y), len(rms[0])), rms[0]
    )
    return rms_interpolated


def create_windows(y, sr, window_length_seconds=1.0, window_stride_seconds=0.1):
    """
    Creates overlapping windows from an audio signal.

    Args:
        y (np.ndarray): Audio signal
        sr (int): Sample rate
        window_length_seconds (float): Length of each window in seconds
        window_stride_seconds (float): Stride between consecutive windows in seconds
                                      (smaller values create more overlap)

    Returns:
        list: List of audio windows (numpy arrays)

    Notes:
        - Increasing window_stride_seconds will reduce the number of windows proportionally
        - For example, doubling the stride will approximately halve the number of windows
        - The function ensures all windows have exactly the same length (window_length_seconds)
    """
    # Convert time values to sample counts
    window_length = int(window_length_seconds * sr)
    window_stride = int(window_stride_seconds * sr)
    windows = []

    # Simple approach: keep creating windows until we can't fit any more
    start = 0
    while start + window_length <= len(y):
        window = y[start : start + window_length]
        windows.append(window)
        start += window_stride

    return windows


def preprocess_audio(audio_file, cfg=None, is_robot=False):
    """
    Preprocess audio by removing noise.

    Args:
        audio_file (str): Path to audio file to process
        cfg (DictConfig): Configuration settings
        is_robot (bool): Whether processing robot data (True) or probe data (False)

    Returns:
        tuple: (preprocessed audio array, sample rate)
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Determine noise file path
    noise_file = "./robo_humming.wav"
    if cfg and hasattr(cfg, 'preprocessing'):
        if is_robot and hasattr(cfg.preprocessing, 'robot_noise_file'):
            noise_file = cfg.preprocessing.robot_noise_file
        elif not is_robot and hasattr(cfg.preprocessing, 'probe_noise_file'):
            noise_file = cfg.preprocessing.probe_noise_file
        elif hasattr(cfg.preprocessing, 'noise_file'):
            noise_file = cfg.preprocessing.noise_file
    
    # Check if noise file exists and handle relative paths
    if noise_file and not os.path.isabs(noise_file):
        # Try to find the file in the current directory
        if not os.path.exists(noise_file):
            # Try with hydra's original working directory if available
            try:
                if 'hydra' in sys.modules:
                    original_cwd = hydra.utils.get_original_cwd()
                    abs_noise_file = os.path.join(original_cwd, noise_file)
                    if os.path.exists(abs_noise_file):
                        noise_file = abs_noise_file
            except:
                # If that fails, just use the original path
                pass

    # Check if noise file exists
    if os.path.exists(noise_file):
        # Load the noise sample
        try:
            noise_y, noise_sr = librosa.load(noise_file)

            # Ensure both audio files have the same sample rate
            if sr != noise_sr:
                noise_y = librosa.resample(noise_y, orig_sr=noise_sr, target_sr=sr)

            # Apply noise reduction
            y = nr.reduce_noise(
                y=y, sr=sr, y_noise=noise_y, prop_decrease=1.0, stationary=False
            )
        except Exception as e:
            print(f"Error during denoising: {e}")
            # Continue with original audio if denoising fails

    return y, sr


def segment_audio(
    audio_file,
    window_length_seconds=1.0,
    window_stride_seconds=0.1,
    non_contact_threshold_factor=0.5,
    enable_squeezing=True,
    squeeze_factor_seconds=0.3,
    min_segment_duration=0.25,
    dynamic_threshold_offset=0.15,
    cfg=None,
    is_robot=False,
    return_filtered_segments=False
):
    """
    Segments audio file into contact and non-contact regions with overlapping windows.

    Args:
        audio_file (str): Path to audio file
        window_length_seconds (float): Length of each window in seconds
        window_stride_seconds (float): Stride between windows in seconds (for contact segments)
                                      Non-contact segments will use double this stride
        non_contact_threshold_factor (float): Factor to multiply with the dynamic threshold
                                             to create a stricter threshold for non-contact regions
        enable_squeezing (bool): Whether to enable the segment squeezing mechanism
        squeeze_factor_seconds (float): Maximum gap between segments in seconds that will be merged
        min_segment_duration (float): Minimum duration of a segment in seconds (before windowing)
        dynamic_threshold_offset (float): Offset to add to the dynamic threshold
        cfg (DictConfig): Configuration settings
        is_robot (bool): Whether processing robot data (True) or probe data (False)
        return_filtered_segments (bool): Whether to also return filtered segments

    Returns:
        If return_filtered_segments is False (default):
            list of tuples: Each tuple contains (start_idx, end_idx, is_contact, windows)
        If return_filtered_segments is True:
            tuple: (all_segments, filtered_segments)
    """
    # Define the get_config function to ensure consistent parameter retrieval
    def get_config(key, default):
        try:
            # First try to get from segmentation section
            if cfg and hasattr(cfg, 'segmentation') and hasattr(cfg.segmentation, key):
                return getattr(cfg.segmentation, key)
            # Then try from data section
            elif cfg and hasattr(cfg, 'data') and hasattr(cfg.data, key):
                return getattr(cfg.data, key)
            # Fall back to the provided default
            else:
                return default
        except (AttributeError, KeyError):
            return default

    # Get parameters from config if available, otherwise use function arguments
    window_length_seconds = get_config('window_length_seconds', window_length_seconds)
    window_stride_seconds = get_config('window_stride_seconds', window_stride_seconds)
    non_contact_threshold_factor = get_config('non_contact_threshold_factor', non_contact_threshold_factor)
    enable_squeezing = get_config('enable_squeezing', enable_squeezing)
    dynamic_threshold_offset = get_config('dynamic_threshold_offset', dynamic_threshold_offset)
    
    # Get min_segment_duration from config (with min_duration as key in segmentation section)
    min_segment_duration = get_config('min_duration', min_segment_duration)
    
    # Check if squeeze_duration exists in the config, fall back to squeeze_factor_seconds
    if cfg and hasattr(cfg, 'segmentation') and hasattr(cfg.segmentation, 'squeeze_duration'):
        squeeze_factor_seconds = cfg.segmentation.squeeze_duration
    
    # First preprocess the audio with the appropriate noise reduction
    y, sr = preprocess_audio(audio_file, cfg=cfg, is_robot=is_robot)
    
    # Process audio
    min_samples = int(min_segment_duration * sr)
    squeeze_samples = int(squeeze_factor_seconds * sr)  # Convert seconds to samples
    # Convert window length to samples - used for final filtering
    window_length_samples = int(window_length_seconds * sr)
    
    envelope = get_audio_envelope(y, frame_length=512, hop_length=128)
    window_size = int(0.02 * sr)
    envelope_smoothed = np.convolve(
        envelope, np.ones(window_size) / window_size, mode="same"
    )

    # Use more sensitive dynamic thresholding
    noise_floor = np.percentile(envelope_smoothed, 10)
    signal_peak = np.percentile(envelope_smoothed, 90)
    dynamic_threshold = (
        noise_floor + (signal_peak - noise_floor) * dynamic_threshold_offset
    )

    # Create a stricter threshold for non-contact regions
    non_contact_threshold = dynamic_threshold * non_contact_threshold_factor

    # Add debug information about thresholds
    print("\n=== CONTACT DETECTION THRESHOLDS ===")
    print(f"Noise floor (10th percentile): {noise_floor:.6f}")
    print(f"Signal peak (90th percentile): {signal_peak:.6f}")
    print(f"Dynamic threshold: {dynamic_threshold:.6f}")
    print(f"Non-contact threshold: {non_contact_threshold:.6f}")
    print("==================================\n")

    # Create contact mask based on dynamic threshold
    contact_mask = envelope_smoothed > dynamic_threshold
    
    # Print statistics about the contact mask
    contact_percentage = np.mean(contact_mask) * 100
    print(f"Contact mask statistics: {contact_percentage:.2f}% of samples above threshold")
    print(f"Total samples in mask: {len(contact_mask)}, contact samples: {np.sum(contact_mask)}")

    # Find change points
    change_points = np.where(np.diff(contact_mask))[0]

    if len(change_points) == 0:
        if np.any(contact_mask):
            segment_audio = y[0 : len(y)]
            # Only create windows if the segment is long enough
            if len(segment_audio) >= window_length_samples:
                windows = create_windows(
                    segment_audio, sr, window_length_seconds, window_stride_seconds
                )
                return [(0, len(y), True, windows)]
            else:
                print(f"Skipping sole segment: too short for a full window ({len(segment_audio)} samples < {window_length_samples} samples)")
                return []
        else:
            segment_audio = y[0 : len(y)]
            # Only create windows if the segment is long enough
            if len(segment_audio) >= window_length_samples:
                windows = create_windows(
                    segment_audio, sr, window_length_seconds, window_stride_seconds
                )
                return [(0, len(y), False, windows)]
            else:
                print(f"Skipping sole segment: too short for a full window ({len(segment_audio)} samples < {window_length_samples} samples)")
                return []

    # Create initial segments
    segments = []
    start_idx = 0 if contact_mask[0] else change_points[0]

    for i in range(len(change_points) - 1):
        if contact_mask[change_points[i] + 1]:
            start_idx = change_points[i]
        else:
            end_idx = change_points[i]
            if end_idx - start_idx >= min_samples:
                segments.append((start_idx, end_idx, True, []))

    # Handle last segment
    if contact_mask[-1]:
        end_idx = len(y)
        if end_idx - start_idx >= min_samples:
            segments.append((start_idx, end_idx, True, []))

    # Define merge function
    def merge_segments(segs, squeeze_samples):
        if len(segs) <= 1:
            return segs

        merged = []
        current_start = segs[0][0]
        current_end = segs[0][1]

        for i in range(1, len(segs)):
            if segs[i][0] - current_end <= squeeze_samples:
                current_end = segs[i][1]
            else:
                if current_end - current_start >= min_samples:
                    merged.append((current_start, current_end, True, []))
                current_start = segs[i][0]
                current_end = segs[i][1]

        if current_end - current_start >= min_samples:
            merged.append((current_start, current_end, True, []))

        return merged

    # Merge segments only if squeezing is enabled
    if enable_squeezing:
        prev_len = len(segments) + 1
        merged_segments = segments

        while len(merged_segments) < prev_len:
            prev_len = len(merged_segments)
            merged_segments = merge_segments(merged_segments, squeeze_samples)
    else:
        # Skip merging if squeezing is disabled
        merged_segments = segments

    # Now find non-contact segments between contact segments
    all_segments = []

    # Create a list to store filtered out segments for visualization
    filtered_segments = []

    # Helper function to find valid non-contact regions
    def extract_non_contact_region(start_idx, end_idx):
        # Get the envelope for this region
        region_envelope = envelope_smoothed[start_idx:end_idx]

        # If the entire region is below the non-contact threshold, use it all
        if not np.any(region_envelope > non_contact_threshold):
            return start_idx, end_idx

        # Find the longest continuous non-contact section
        non_contact_sections = []
        current_start = None

        for i in range(len(region_envelope)):
            if (
                region_envelope[i] <= non_contact_threshold
            ):  # Below non-contact threshold
                if current_start is None:
                    current_start = i
            elif (
                current_start is not None
            ):  # Above threshold and we were in a non-contact section
                non_contact_sections.append((current_start + start_idx, i + start_idx))
                current_start = None

        # Handle case where we end in a non-contact section
        if current_start is not None:
            non_contact_sections.append(
                (current_start + start_idx, len(region_envelope) + start_idx)
            )

        # Find the longest section
        if non_contact_sections:
            longest_section = max(non_contact_sections, key=lambda x: x[1] - x[0])
            if longest_section[1] - longest_section[0] >= min_samples:
                return longest_section

        # No valid non-contact section found
        return None, None

    # Add non-contact segment at the beginning if needed
    if merged_segments and merged_segments[0][0] > min_samples:
        start = 0
        end = merged_segments[0][0]
        valid_start, valid_end = extract_non_contact_region(start, end)

        if valid_start is not None and valid_end - valid_start >= min_samples:
            segment_audio = y[valid_start:valid_end]
            # Only add if segment is long enough for a full window
            if len(segment_audio) >= window_length_samples:
                # Use DOUBLED stride for non-contact segments
                non_contact_stride = window_stride_seconds * 2
                windows = create_windows(
                    segment_audio, sr, window_length_seconds, non_contact_stride
                )
                # Only add if we can create at least one window
                if windows:
                    all_segments.append((valid_start, valid_end, False, windows))
            else:
                print(f"Skipping initial non-contact segment: too short for a full window ({len(segment_audio)} samples < {window_length_samples} samples)")
                # Store for visualization
                filtered_segments.append((valid_start, valid_end, False))

    # Add all contact segments and find non-contact segments between them
    for i, (start, end, is_contact, _) in enumerate(merged_segments):
        # Add the contact segment only if it's long enough for a full window
        segment_audio = y[start:end]
        if len(segment_audio) >= window_length_samples:
            # Use normal stride for contact segments
            windows = create_windows(
                segment_audio, sr, window_length_seconds, window_stride_seconds
            )
            # Only add if we can create at least one window
            if windows:
                all_segments.append((start, end, True, windows))
            else:
                print(f"Skipping contact segment {i}: created no windows")
                # Store for visualization
                filtered_segments.append((start, end, True))
        else:
            print(f"Skipping contact segment {i}: too short for a full window ({len(segment_audio)} samples < {window_length_samples} samples)")
            # Store for visualization
            filtered_segments.append((start, end, True))

        # Add non-contact segment if there's a gap before the next contact segment
        if i < len(merged_segments) - 1:
            next_start = merged_segments[i + 1][0]
            if next_start - end >= min_samples:
                valid_start, valid_end = extract_non_contact_region(end, next_start)

                if valid_start is not None and valid_end - valid_start >= min_samples:
                    segment_audio = y[valid_start:valid_end]
                    # Only add if segment is long enough for a full window
                    if len(segment_audio) >= window_length_samples:
                        # Use DOUBLED stride for non-contact segments
                        non_contact_stride = window_stride_seconds * 2
                        windows = create_windows(
                            segment_audio, sr, window_length_seconds, non_contact_stride
                        )
                        # Only add if we can create at least one window
                        if windows:
                            all_segments.append((valid_start, valid_end, False, windows))
                        else:
                            print(f"Skipping non-contact segment between {i} and {i+1}: created no windows")
                            # Store for visualization
                            filtered_segments.append((valid_start, valid_end, False))
                    else:
                        print(f"Skipping non-contact segment between {i} and {i+1}: too short for a full window ({len(segment_audio)} samples < {window_length_samples} samples)")
                        # Store for visualization
                        filtered_segments.append((valid_start, valid_end, False))

    # Add non-contact segment at the end if needed
    if merged_segments and merged_segments[-1][1] < len(y) - min_samples:
        start = merged_segments[-1][1]
        end = len(y)
        valid_start, valid_end = extract_non_contact_region(start, end)

        if valid_start is not None and valid_end - valid_start >= min_samples:
            segment_audio = y[valid_start:valid_end]
            # Only add if segment is long enough for a full window
            if len(segment_audio) >= window_length_samples:
                # Use DOUBLED stride for non-contact segments
                non_contact_stride = window_stride_seconds * 2
                windows = create_windows(
                    segment_audio, sr, window_length_seconds, non_contact_stride
                )
                # Only add if we can create at least one window
                if windows:
                    all_segments.append((valid_start, valid_end, False, windows))
                else:
                    print(f"Skipping final non-contact segment: created no windows")
                    # Store for visualization
                    filtered_segments.append((valid_start, valid_end, False))
            else:
                print(f"Skipping final non-contact segment: too short for a full window ({len(segment_audio)} samples < {window_length_samples} samples)")
                # Store for visualization
                filtered_segments.append((valid_start, valid_end, False))

    print(f"Final segments after filtering by window size: {len(all_segments)}")
    print(f"Filtered out segments (too short): {len(filtered_segments)}")
    
    # Return based on the return_filtered_segments flag
    if return_filtered_segments:
        return all_segments, filtered_segments
    else:
        return all_segments


def map_to_category(bag_name, cfg=None):
    """Map bag name to a standard category."""
    # Define default categories
    default_categories = ['leaf', 'twig', 'trunk', 'ambient']
    
    # Use categories from config if available
    categories = default_categories
    if cfg and hasattr(cfg, 'output') and hasattr(cfg.output, 'categories'):
        categories = cfg.output.categories
    
    # Check for each category in the bag name
    bag_name_lower = bag_name.lower()
    for category in categories:
        if category.lower() in bag_name_lower:
            return category
    
    # Check for ambient/background/noncontact specially
    if 'ambient' in categories and any(term in bag_name_lower for term in ['ambient', 'background', 'noncontact']):
        return 'ambient'
    
    # Return None if no match
    return None


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Define default values for missing config parameters
    default_config = {
        "window_length_seconds": 1.0,
        "window_stride_seconds": 0.1,
        "non_contact_threshold_factor": 0.3,
        "enable_squeezing": True,
        "min_duration": 0.25,
        "dynamic_threshold_offset": 0.15,
        "squeeze_duration": 0.3
    }
    
    # Create a function to safely get config values with defaults
    def get_config(key, default_value):
        try:
            # First try to get from segmentation section
            if hasattr(cfg, 'segmentation') and hasattr(cfg.segmentation, key):
                return getattr(cfg.segmentation, key)
            # Then try from data section
            elif hasattr(cfg, 'data') and hasattr(cfg.data, key):
                return getattr(cfg.data, key)
            # Fall back to the provided default
            else:
                return default_value
        except (AttributeError, KeyError):
            return default_value
    
    # Print available categories from config
    if hasattr(cfg, 'output') and hasattr(cfg.output, 'categories'):
        print(f"Available categories from config: {cfg.output.categories}")
    else:
        print("No categories found in config, using defaults: ['leaf', 'twig', 'trunk', 'ambient']")
    
    # Test the category mapping function
    test_names = [
        "leaf_sample", "twig_test", "trunk_data", "ambient_background", 
        "random_name", "noncontact_sample"
    ]
    
    print("\nTesting category mapping:")
    for name in test_names:
        category = map_to_category(name, cfg)
        print(f"  '{name}' -> {category}")
    
    # Get original working directory
    try:
        original_cwd = hydra.utils.get_original_cwd()
    except:
        original_cwd = os.getcwd()
    
    # Use the correct full path to the audio file
    wav_file = os.path.join(original_cwd, "TestOutput/umass_tree1_robot_2_split_1_new_trunk/audio/umass_tree1_robot_2_split_1_new_trunk.wav")
    
    # Print out the full absolute path
    abs_wav_file = os.path.abspath(wav_file)
    print(f"Attempting to load audio file from:")
    print(f"  Path: {wav_file}")
    print(f"  Absolute path: {abs_wav_file}")
    print(f"  File exists: {os.path.exists(abs_wav_file)}")
    
    # Check if the file exists
    if not os.path.exists(abs_wav_file):
        print(f"Warning: File does not exist at {abs_wav_file}")
        # Try to find a sample file in the current directory
        sample_files = glob.glob("*.wav")
        if sample_files:
            print(f"Found sample files in current directory: {sample_files}")
            wav_file = sample_files[0]
            print(f"Using sample file: {wav_file}")
        else:
            print("No sample files found in current directory")
            print("Please specify a valid audio file path")
            return
    
    # We need to get the sample rate before segmentation for visualization
    # Load the audio file to get the sample rate
    try:
        audio_data, sr = librosa.load(wav_file, sr=None)
        print(f"Loaded audio with sample rate: {sr}Hz")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
        
    # Get segments using the updated segment_audio function with config
    segments_result = segment_audio(
        wav_file,
        window_length_seconds=get_config('window_length_seconds', default_config['window_length_seconds']),
        window_stride_seconds=get_config('window_stride_seconds', default_config['window_stride_seconds']),
        non_contact_threshold_factor=get_config('non_contact_threshold_factor', default_config['non_contact_threshold_factor']),
        enable_squeezing=get_config('enable_squeezing', default_config['enable_squeezing']),
        squeeze_factor_seconds=get_config('squeeze_duration', default_config['squeeze_duration']),
        min_segment_duration=get_config('min_duration', default_config['min_duration']),
        dynamic_threshold_offset=get_config('dynamic_threshold_offset', default_config['dynamic_threshold_offset']),
        cfg=cfg,
        is_robot=True,  # Assuming this is robot data for the example
        return_filtered_segments=True  # Request filtered segments for visualization
    )
    
    # Unpack the result - now we get both segments and filtered segments
    segments, filtered_segments = segments_result
    
    # Count total windows and calculate segment durations
    total_windows = 0
    windows_per_segment = []
    segment_durations = []
    window_durations = []
    contact_segments = 0
    non_contact_segments = 0

    # Calculate window duration in seconds
    window_length_seconds = get_config('window_length_seconds', default_config['window_length_seconds'])
    window_stride_seconds = get_config('window_stride_seconds', default_config['window_stride_seconds'])
    expected_window_duration = window_length_seconds
    expected_window_samples = int(expected_window_duration * sr)

    for i, (start, end, is_contact, windows) in enumerate(segments):
        segment_windows = len(windows)
        total_windows += segment_windows
        windows_per_segment.append(segment_windows)

        # Calculate segment duration in seconds
        duration = (end - start) / sr
        segment_durations.append(duration)

        # Count segment types
        if is_contact:
            contact_segments += 1
        else:
            non_contact_segments += 1

        segment_type = "Contact" if is_contact else "Non-contact"
        print(
            f"Segment {i+1} ({segment_type}): {segment_windows} windows, {duration:.2f} seconds"
        )

        # Verify window durations
        for j, window in enumerate(windows):
            window_duration = len(window) / sr
            window_durations.append(window_duration)
            if (
                abs(window_duration - expected_window_duration) > 0.001
            ):  # Allow small floating point differences
                print(
                    f"  Warning: Window {j+1} has unexpected duration: {window_duration:.4f}s (expected {expected_window_duration:.4f}s)"
                )

    # Display statistics
    print(
        f"\nTotal segments: {len(segments)} ({contact_segments} contact, {non_contact_segments} non-contact)"
    )
    print(f"Total windows: {total_windows}")

    if windows_per_segment:
        print(
            f"Average windows per segment: {sum(windows_per_segment)/len(windows_per_segment):.2f}"
        )
        print(f"Min windows in a segment: {min(windows_per_segment)}")
        print(f"Max windows in a segment: {max(windows_per_segment)}")

    if segment_durations:
        print(
            f"\nAverage segment duration: {sum(segment_durations)/len(segment_durations):.2f} seconds"
        )
        print(f"Min segment duration: {min(segment_durations):.2f} seconds")
        print(f"Max segment duration: {max(segment_durations):.2f} seconds")

    # Calculate separate window statistics for contact and non-contact segments
    if segments:
        contact_windows = []
        non_contact_windows = []
        
        for i, (start, end, is_contact, windows) in enumerate(segments):
            if is_contact:
                contact_windows.extend(windows)
            else:
                non_contact_windows.extend(windows)
        
        # Print window counts
        print(f"\nContact segments: {contact_segments} segments, {len(contact_windows)} windows")
        print(f"Non-contact segments: {non_contact_segments} segments, {len(non_contact_windows)} windows")
        
        # Calculate window density (windows per second) for each type
        if contact_segments > 0 and contact_windows:
            contact_seconds = sum((end - start) / sr for start, end, is_contact, _ in segments if is_contact)
            contact_density = len(contact_windows) / contact_seconds if contact_seconds > 0 else 0
            print(f"Contact window density: {contact_density:.2f} windows/second (stride: {window_stride_seconds:.2f}s)")
        
        if non_contact_segments > 0 and non_contact_windows:
            non_contact_seconds = sum((end - start) / sr for start, end, is_contact, _ in segments if not is_contact)
            non_contact_density = len(non_contact_windows) / non_contact_seconds if non_contact_seconds > 0 else 0
            print(f"Non-contact window density: {non_contact_density:.2f} windows/second (stride: {window_stride_seconds * 2:.2f}s)")
        
        # Calculate effective time coverage
        if contact_windows:
            contact_coverage = len(contact_windows) * window_stride_seconds
            print(f"Contact window temporal coverage: {contact_coverage:.2f} seconds")
        
        if non_contact_windows:
            # Non-contact uses double stride
            non_contact_coverage = len(non_contact_windows) * (window_stride_seconds * 2)
            print(f"Non-contact window temporal coverage: {non_contact_coverage:.2f} seconds")
        
        # Print data balance statistics
        print(f"\nData balance:")
        total_windows = len(contact_windows) + len(non_contact_windows)
        if total_windows > 0:
            contact_percentage = len(contact_windows) / total_windows * 100
            non_contact_percentage = len(non_contact_windows) / total_windows * 100
            print(f"Contact windows: {len(contact_windows)} ({contact_percentage:.1f}%)")
            print(f"Non-contact windows: {len(non_contact_windows)} ({non_contact_percentage:.1f}%)")

    # Window statistics
    if window_durations:
        print(f"\nWindow statistics:")
        print(
            f"Expected window length: {expected_window_duration:.4f} seconds ({expected_window_samples} samples)"
        )
        print(f"Actual window lengths:")
        print(f"  Average: {sum(window_durations)/len(window_durations):.4f} seconds")
        print(f"  Min: {min(window_durations):.4f} seconds")
        print(f"  Max: {max(window_durations):.4f} seconds")
        print(
            f"  All same length: {all(abs(d - expected_window_duration) < 0.001 for d in window_durations)}"
        )

    print(
        f"Window stride: {window_stride_seconds:.2f} seconds"
    )
    print(f"Total window time: {total_windows * expected_window_duration:.2f} seconds")
    print(
        f"Effective window time (accounting for overlap): {total_windows * window_stride_seconds:.2f} seconds"
    )

    # Add statistics about filtered segments
    filtered_contact = sum(1 for _, _, is_contact in filtered_segments if is_contact)
    filtered_non_contact = len(filtered_segments) - filtered_contact
    
    if filtered_segments:
        filtered_durations = [(end - start) / sr for start, end, _ in filtered_segments]
        print(f"\nFiltered out segments: {len(filtered_segments)} ({filtered_contact} contact, {filtered_non_contact} non-contact)")
        print(f"Average filtered segment duration: {sum(filtered_durations)/len(filtered_durations):.2f} seconds")
        print(f"Min filtered segment duration: {min(filtered_durations):.2f} seconds")
        print(f"Max filtered segment duration: {max(filtered_durations):.2f} seconds")

    # Create visualization
    # We need to use the same preprocessed audio that was used for segmentation
    # Let's load and preprocess it again to make sure we're visualizing the same data
    y, sr = preprocess_audio(wav_file, cfg=cfg, is_robot=True)
    time = np.arange(len(y)) / sr

    # Get envelope and threshold for visualization
    envelope = get_audio_envelope(y, frame_length=512, hop_length=128)
    envelope_smoothed = np.convolve(
        envelope, np.ones(int(0.02 * sr)) / int(0.02 * sr), mode="same"
    )

    # Calculate dynamic threshold for visualization
    noise_floor = np.percentile(envelope_smoothed, 10)
    signal_peak = np.percentile(envelope_smoothed, 90)
    
    # Get the dynamic threshold offset from config
    dynamic_threshold_offset = get_config('dynamic_threshold_offset', 0.15)
    
    dynamic_threshold = noise_floor + (signal_peak - noise_floor) * dynamic_threshold_offset
    non_contact_threshold = dynamic_threshold * get_config(
        "non_contact_threshold_factor", 0.5
    )

    # Calculate max amplitude for y-axis limits
    max_amplitude = max(abs(np.max(y)), abs(np.min(y)))

    # Debug output
    print(f"Audio signal min: {np.min(y)}, max: {np.max(y)}")
    print(
        f"Envelope min: {np.min(envelope_smoothed)}, max: {np.max(envelope_smoothed)}"
    )
    print(f"Setting y-axis limits to: [{-max_amplitude*1.2}, {max_amplitude*1.2}]")

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Plot audio data on primary y-axis
    ax1.plot(time, y, "b-", alpha=0.3, label="Audio Signal")
    ax1.plot(time, envelope_smoothed, "r-", alpha=0.8, label="Envelope")
    ax1.axhline(
        y=dynamic_threshold, color="g", linestyle="--", label="Contact Threshold"
    )
    ax1.axhline(
        y=non_contact_threshold,
        color="m",
        linestyle="--",
        label="Non-Contact Threshold",
    )
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Set y-axis limits based on max amplitude from audio
    ax1.set_ylim([-max_amplitude * 1.2, max_amplitude * 1.2])

    # Add min/max amplitude text to the plot
    ax1.text(
        0.02,
        0.98,
        f"Min amplitude: {np.min(y):.6f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax1.text(
        0.02,
        0.93,
        f"Max amplitude: {np.max(y):.6f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax1.text(
        0.02,
        0.88,
        f"Contact Threshold: {dynamic_threshold:.6f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax1.text(
        0.02,
        0.83,
        f"Non-Contact Threshold: {non_contact_threshold:.6f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Add squeezing info to the plot
    ax1.text(
        0.02,
        0.78,
        f"Segment squeezing: {'Enabled' if get_config('enable_squeezing', True) else 'Disabled'}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Add window configuration info to the plot
    ax1.text(
        0.02,
        0.73,
        f"Window length: {window_length_seconds:.2f}s",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax1.text(
        0.02,
        0.68,
        f"Contact window stride: {window_stride_seconds:.2f}s",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax1.text(
        0.02,
        0.63,
        f"Non-contact window stride: {window_stride_seconds * 2:.2f}s (doubled)",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Create secondary y-axis for force data
    ax2 = ax1.twinx()

    # Add vertical lines for segment and window boundaries
    if segments:
        # Get the actual y-limits after they've been set
        y_min, y_max = ax1.get_ylim()

        # First plot filtered segments as gray
        for i, (start, end, is_contact) in enumerate(filtered_segments):
            # Use gray color for filtered segments
            segment_color = "gray"
            segment_type = "Contact (filtered)" if is_contact else "Non-contact (filtered)"

            # Add segment markers with dashed lines
            ax1.axvline(
                x=start / sr, color=segment_color, linestyle="--", alpha=0.4, linewidth=1
            )
            ax1.axvline(
                x=end / sr, color=segment_color, linestyle="--", alpha=0.4, linewidth=1
            )

            # Add segment duration text
            duration = (end - start) / sr
            # Position filtered segment labels lower to avoid overlap
            ax1.text(
                start / sr,
                y_max * 0.75,
                f"F{i+1}_start",
                rotation=90,
                color=segment_color,
                verticalalignment="top",
                fontsize=7,
            )
            ax1.text(
                end / sr,
                y_max * 0.75,
                f"F{i+1}_end ({duration:.2f}s)",
                rotation=90,
                color=segment_color,
                verticalalignment="top",
                fontsize=7,
            )
            
            # Draw a semi-transparent rectangle to highlight the filtered region
            ax1.axvspan(start / sr, end / sr, alpha=0.1, color=segment_color)

        # Now plot the kept segments
        for i, (start, end, is_contact, windows) in enumerate(segments):
            # Choose color based on segment type
            segment_color = "r" if is_contact else "b"
            segment_type = "Contact" if is_contact else "Non-contact"

            # Add segment markers
            ax1.axvline(
                x=start / sr, color=segment_color, linestyle="-", alpha=0.5, linewidth=2
            )
            ax1.axvline(
                x=end / sr, color=segment_color, linestyle="-", alpha=0.5, linewidth=2
            )

            # Add segment duration text - position at the very top of the plot
            duration = (end - start) / sr
            ax1.text(
                start / sr,
                y_max * 0.95,
                f"S{i+1}_start ({segment_type})",
                rotation=90,
                color=segment_color,
                verticalalignment="top",
                fontweight="bold",
            )
            ax1.text(
                end / sr,
                y_max * 0.95,
                f"S{i+1}_end ({duration:.2f}s)",
                rotation=90,
                color=segment_color,
                verticalalignment="top",
                fontweight="bold",
            )

            # Add window markers (thinner lines)
            # Use appropriate stride based on segment type
            actual_stride = window_stride_seconds if is_contact else window_stride_seconds * 2
            
            for j, window in enumerate(windows):
                # Calculate correct window start position
                # Each window starts at the segment start plus j times the appropriate stride
                window_start = start / sr + j * actual_stride

                # Window end is simply window_start plus the window length
                window_end = window_start + window_length_seconds

                # Add start marker
                ax1.axvline(
                    x=window_start, color=segment_color, linestyle=":", alpha=0.3
                )
                ax1.text(
                    window_start,
                    y_max * 0.85,
                    f"W{j+1}_start",
                    rotation=90,
                    verticalalignment="top",
                    fontsize=8,
                    color=segment_color,
                )

                # Add end marker
                ax1.axvline(x=window_end, color=segment_color, linestyle=":", alpha=0.3)
                ax1.text(
                    window_end,
                    y_max * 0.85,
                    f"W{j+1}_end",
                    rotation=90,
                    verticalalignment="top",
                    fontsize=8,
                    color=segment_color,
                )
                
                # Add a small label to indicate the stride value for each window
                if j > 0:  # Only for windows after the first one
                    stride_label = f"{actual_stride:.2f}s"
                    ax1.text(
                        window_start - actual_stride/2,  # Position in the middle of the stride
                        y_max * 0.80,
                        stride_label,
                        horizontalalignment="center",
                        fontsize=7,
                        color=segment_color,
                    )

    # Add a legend entry for filtered segments
    if filtered_segments:
        ax1.plot([], [], color='gray', linestyle='--', label='Filtered Segments')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper right")

    # Update title to include filtered segments and stride information
    plt.title(
        f"Audio Analysis: {len(segments)} segments ({contact_segments} contact, {non_contact_segments} non-contact)\n"
        f"Windows: {len(contact_windows)} contact (stride: {window_stride_seconds:.2f}s), "
        f"{len(non_contact_windows)} non-contact (stride: {window_stride_seconds * 2:.2f}s)\n"
        f"Filtered: {len(filtered_segments)} segments ({filtered_contact} contact, {filtered_non_contact} non-contact)"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return segments, filtered_segments


if __name__ == "__main__":
    main()
