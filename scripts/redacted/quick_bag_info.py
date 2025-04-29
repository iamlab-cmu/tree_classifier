import rosbag
import argparse
import os
import glob
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import librosa
import tempfile
import soundfile as sf
import sys

# Import the segment_audio function from segment_audio.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segment_audio import segment_audio, get_audio_envelope

def get_bag_duration(bag_path):
    """
    Get the duration of the bag file in seconds.
    
    Args:
        bag_path (str): Path to the ROS bag file
        
    Returns:
        float: Duration in seconds
    """
    with rosbag.Bag(bag_path, 'r') as bag:
        start_time = None
        end_time = None
        
        # Get first message time
        for _, _, t in bag.read_messages():
            start_time = t.to_sec()
            break
        
        # Get last message time
        for _, _, t in bag.read_messages():
            end_time = t.to_sec()
        
        return end_time - start_time if start_time and end_time else 0

def extract_audio_from_bag(bag_path, save_to_file=None):
    """Extract audio data from a ROS bag file and save to a temporary WAV file."""
    try:
        # Find audio topic
        with rosbag.Bag(bag_path, 'r') as bag:
            # Get topic info
            topics_info = bag.get_type_and_topic_info()[1]
            
            # Find audio topic - priority order: /audio, AudioData types, then any with 'audio'
            audio_topic = '/audio' if '/audio' in topics_info else None
            
            if not audio_topic:
                for topic, info in topics_info.items():
                    if 'AudioData' in info.msg_type:
                        audio_topic = topic
                        break
            
            if not audio_topic:
                for topic in topics_info.keys():
                    if 'audio' in topic.lower() and 'info' not in topic.lower():
                        audio_topic = topic
                        break
            
            if not audio_topic:
                print(f"No audio topic found in {bag_path}")
                return None
                
            print(f"Selected audio topic: {audio_topic} with message type: {topics_info[audio_topic].msg_type}")
            
            # Collect audio data
            audio_data = []
            msg_count = 0
            
            for _, msg, _ in bag.read_messages(topics=[audio_topic]):
                try:
                    # Handle different audio message types
                    if hasattr(msg, 'data') and isinstance(msg.data, (list, tuple, np.ndarray)):
                        float_samples = np.array(msg.data, dtype=np.float32)
                        if len(float_samples.shape) > 1:
                            # If multi-channel, reshape
                            float_samples = float_samples.reshape(-1, 2)
                        audio_data.append(float_samples)
                        msg_count += 1
                except Exception as e:
                    print(f"Error processing audio message: {str(e)}")
            
            if not audio_data:
                print(f"No valid audio data found in {bag_path}")
                return None
                
            # Concatenate all audio data
            try:
                audio_array = np.concatenate(audio_data, axis=0)
                print(f"Collected {audio_array.size} audio samples from {msg_count} messages")
            except ValueError as e:
                print(f"Error concatenating audio data: {str(e)}")
                # Try to handle arrays of different shapes
                max_shape = max([a.shape[1] if len(a.shape) > 1 else 1 for a in audio_data])
                normalized_data = []
                for a in audio_data:
                    if len(a.shape) == 1:
                        a = a.reshape(-1, 1)
                    if a.shape[1] < max_shape:
                        a = np.pad(a, ((0, 0), (0, max_shape - a.shape[1])), 'constant')
                    normalized_data.append(a)
                audio_array = np.concatenate(normalized_data, axis=0)
                
            # Create temp file path if not provided
            temp_path = save_to_file if save_to_file else tempfile.mktemp(suffix='.wav')
            
            # Save to WAV file
            try:
                print(f"Saving audio data to {temp_path} with scipy (sample rate: 44100)")
                sf.write(temp_path, audio_array, 44100)
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"Failed to create audio file at {temp_path}")
                return temp_path
            except Exception as e:
                print(f"Error saving audio data to WAV: {str(e)}")
                return None
    
    except Exception as e:
        print(f"Error extracting audio from {bag_path}: {str(e)}")
        return None

def count_segments_from_bag(bag_path, window_length_seconds=1.0, window_stride_seconds=0.1):
    """
    Count segments by analyzing audio data directly from the bag file using the segment_audio function.
    
    Args:
        bag_path (str): Path to the ROS bag file
        window_length_seconds (float): Length of each window in seconds
        window_stride_seconds (float): Stride between windows in seconds
        
    Returns:
        tuple: (segment_count, segment_durations, segment_categories, bag_category, window_count, window_categories)
    """
    segment_count = 0
    segment_durations = []
    segment_categories = defaultdict(int)
    window_count = 0
    window_categories = defaultdict(int)
    bag_category = None
    
    # Try to determine category from bag filename
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    categories = ['leaf', 'twig', 'trunk']
    for category in categories:
        if category in bag_name.lower():
            bag_category = category
            break
    
    try:
        # Extract audio from bag
        temp_audio_path = extract_audio_from_bag(bag_path)
        
        if temp_audio_path:
            try:
                # Use segment_audio function to get segments - only pass window parameters
                segments, _ = segment_audio(
                    temp_audio_path, 
                    window_length_seconds=window_length_seconds,
                    window_stride_seconds=window_stride_seconds
                )
                
                # Count segments and calculate durations
                segment_count = len(segments)
                
                if segment_count > 0:
                    # Load audio to get sample rate
                    y, sr = librosa.load(temp_audio_path, sr=None)
                    
                    for start, end, is_contact, windows, _ in segments:
                        if is_contact:
                            duration = (end - start) / sr
                            segment_durations.append(duration)
                            
                            # Count windows in this segment
                            if bag_category:
                                segment_categories[bag_category] += 1
                                window_count += len(windows)
                                window_categories[bag_category] += len(windows)
                    
                # Clean up temporary file
                os.unlink(temp_audio_path)
                
            except Exception as e:
                print(f"Error segmenting audio from {bag_path}: {str(e)}")
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        else:
            # If no audio found, just count the bag as one segment
            if bag_category:
                segment_count = 1
                segment_duration = get_bag_duration(bag_path)
                segment_durations.append(segment_duration)
                segment_categories[bag_category] += 1
                # Estimate window count based on duration
                estimated_windows = max(1, int((segment_duration - window_length_seconds) / window_stride_seconds) + 1)
                window_count += estimated_windows
                window_categories[bag_category] += estimated_windows
    
    except Exception as e:
        print(f"Error analyzing audio data in {bag_path}: {str(e)}")
    
    return segment_count, segment_durations, segment_categories, bag_category, window_count, window_categories

def analyze_bags(directory, recursive=False, test_mode=False, window_length_seconds=1.0, window_stride_seconds=0.1):
    """
    Analyze all bag files in a directory.
    
    Args:
        directory (str): Directory containing bag files
        recursive (bool): Whether to search recursively
        test_mode (bool): If True, only process one random bag file
        window_length_seconds (float): Length of each window in seconds
        window_stride_seconds (float): Stride between windows in seconds
        
    Returns:
        dict: Statistics about the bag files
    """
    # Find all bag files
    if recursive:
        bag_files = glob.glob(os.path.join(directory, '**/*.bag'), recursive=True)
    else:
        bag_files = glob.glob(os.path.join(directory, '*.bag'))
    
    if not bag_files:
        print(f"No bag files found in {directory}")
        return None
    
    # If test mode is enabled, just pick one random bag file
    if test_mode and bag_files:
        import random
        bag_files = [random.choice(bag_files)]
        print(f"TEST MODE: Processing only one random bag file:")
        print(f"  - Full path: {bag_files[0]}")
        print(f"  - Filename: {os.path.basename(bag_files[0])}")
    
    # Collect statistics
    durations = []
    segment_counts = []
    all_segment_durations = []
    total_segments_by_category = defaultdict(int)
    bags_by_category = defaultdict(int)
    window_counts = []
    total_windows_by_category = defaultdict(int)
    
    print(f"Analyzing {len(bag_files)} bag files...")
    
    for bag_path in tqdm(bag_files):
        try:
            # Get bag duration
            duration = get_bag_duration(bag_path)
            durations.append(duration)
            
            # Get segment information by analyzing the bag directly
            segment_count, segment_durations, segment_categories, bag_category, window_count, window_categories = count_segments_from_bag(
                bag_path, 
                window_length_seconds=window_length_seconds,
                window_stride_seconds=window_stride_seconds
            )
            
            segment_counts.append(segment_count)
            all_segment_durations.extend(segment_durations)
            window_counts.append(window_count)
            
            # Update category counts
            for category, count in segment_categories.items():
                total_segments_by_category[category] += count
            
            # Update window category counts
            for category, count in window_categories.items():
                total_windows_by_category[category] += count
            
            # Update bag category count
            if bag_category:
                bags_by_category[bag_category] += 1
                
        except Exception as e:
            print(f"Error processing {bag_path}: {str(e)}")
    
    # Calculate statistics
    stats = {
        'total_bags': len(bag_files),
        'avg_duration': np.mean(durations) if durations else 0,
        'min_duration': np.min(durations) if durations else 0,
        'max_duration': np.max(durations) if durations else 0,
        'total_duration': np.sum(durations) if durations else 0,
        'avg_segments_per_bag': np.mean(segment_counts) if segment_counts else 0,
        'total_segments': sum(segment_counts),
        'avg_segment_duration': np.mean(all_segment_durations) if all_segment_durations else 0,
        'segment_categories': dict(total_segments_by_category),
        'bags_by_category': dict(bags_by_category),
        'total_windows': sum(window_counts),
        'avg_windows_per_bag': np.mean(window_counts) if window_counts else 0,
        'window_categories': dict(total_windows_by_category)
    }
    
    return stats

def main():
    """
    Main function using standard argparse.
    """
    parser = argparse.ArgumentParser(description='Get statistics about ROS bag files')
    parser.add_argument('directory', help='Directory containing ROS bag files')
    parser.add_argument('--recursive', '-r', action='store_true', 
                      help='Search for bag files recursively')
    parser.add_argument('--test', '-t', action='store_true',
                      help='Test mode: process only one random bag file')
    parser.add_argument('--window-length', '-w', type=float, default=1.0,
                      help='Window length in seconds')
    parser.add_argument('--window-stride', '-s', type=float, default=0.1,
                      help='Window stride in seconds')
    parser.add_argument('--extract-audio', '-e', action='store_true',
                      help='Extract audio from bag files to WAV files')
    args = parser.parse_args()
    
    print(f"Analyzing bag files in: {args.directory}")
    print(f"Recursive search: {args.recursive}")
    print(f"Test mode: {args.test}")
    print(f"Window length: {args.window_length} seconds")
    print(f"Window stride: {args.window_stride} seconds")
    
    # If extract audio option is enabled, extract audio from bags
    if args.extract_audio:
        if args.recursive:
            bag_files = glob.glob(os.path.join(args.directory, '**/*.bag'), recursive=True)
        else:
            bag_files = glob.glob(os.path.join(args.directory, '*.bag'))
        
        if args.test and bag_files:
            import random
            bag_files = [random.choice(bag_files)]
            print(f"TEST MODE: Extracting audio from one random bag file:")
            print(f"  - Full path: {bag_files[0]}")
            print(f"  - Filename: {os.path.basename(bag_files[0])}")
        
        for bag_path in tqdm(bag_files, desc="Extracting audio"):
            bag_name = os.path.splitext(os.path.basename(bag_path))[0]
            output_wav = f"{bag_name}_audio.wav"
            extract_audio_from_bag(bag_path, save_to_file=output_wav)
        
        print("Audio extraction complete.")
        return
    
    # Otherwise, analyze bags as usual
    stats = analyze_bags(
        args.directory, 
        args.recursive, 
        args.test, 
        window_length_seconds=args.window_length,
        window_stride_seconds=args.window_stride
    )
    
    if stats:
        print("\n=== Bag File Statistics ===")
        print(f"Total bag files: {stats['total_bags']}")
        print(f"Average duration: {stats['avg_duration']:.2f} seconds")
        print(f"Min duration: {stats['min_duration']:.2f} seconds")
        print(f"Max duration: {stats['max_duration']:.2f} seconds")
        print(f"Total duration: {stats['total_duration']:.2f} seconds ({stats['total_duration']/60:.2f} minutes)")
        
        if stats['bags_by_category']:
            print("\n=== Bags by Category ===")
            for category, count in stats['bags_by_category'].items():
                print(f"{category}: {count}")
        
        print("\n=== Segment Statistics ===")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Average segments per bag: {stats['avg_segments_per_bag']:.2f}")
        
        if stats['avg_segment_duration'] > 0:
            print(f"Average segment duration: {stats['avg_segment_duration']:.2f} seconds")
        
        print("\n=== Segment Categories ===")
        for category, count in stats['segment_categories'].items():
            print(f"{category}: {count}")
            
        print("\n=== Window Statistics ===")
        print(f"Total windows (samples): {stats['total_windows']}")
        print(f"Average windows per bag: {stats['avg_windows_per_bag']:.2f}")
        
        print("\n=== Windows by Category ===")
        for category, count in stats['window_categories'].items():
            print(f"{category}: {count}")

if __name__ == '__main__':
    main()
