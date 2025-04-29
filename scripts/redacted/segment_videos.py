import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import tempfile
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import hydra
from omegaconf import DictConfig
import sys
import glob
from segment_audio import segment_audio
import pandas as pd
import genpy

class BagVideoSegmenter:
    def __init__(self, bag_path, output_dir=None, cfg=None):
        """Initialize the segmenter with bag file path and optional output directory."""
        self.bag_path = bag_path
        self.bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        self.cfg = cfg  # Store the config
        
        if output_dir:
            self.output_dir = output_dir
        else:
            # Create output dir based on bag path structure
            bag_dir = os.path.dirname(bag_path)
            # Create output directory preserving the original path structure
            rel_path = os.path.relpath(bag_dir)
            self.output_dir = os.path.join(os.getcwd(), 'output', rel_path, self.bag_name)
        
        # Create all required directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.segments_dir = os.path.join(self.output_dir, 'segments')
        self.audio_dir = os.path.join(self.output_dir, 'audio')
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        
        # Initialize storage for audio segments and data
        self.audio_segments = None
        self.audio_data = None
        self.audio_sr = None
        self.force_data = None
        
        # Calculate FPS from the bag file
        self.fps = self._calculate_fps()
        
        # Store image messages with timestamps
        self.image_msgs = []

    def _calculate_fps(self):
        """Calculate actual FPS from the bag file."""
        with rosbag.Bag(self.bag_path, 'r') as bag:
            frame_times = []
            
            # Collect frame timestamps
            for topic, msg, t in bag.read_messages(topics=['/camera1/color/image_raw']):
                frame_times.append(t.to_sec())
            
            if frame_times:
                # Calculate FPS from frame intervals
                frame_intervals = np.diff(frame_times)
                fps = 1/np.mean(frame_intervals)
                print(f"Calculated FPS from bag: {fps:.1f}")
                return fps
            else:
                print("Warning: No frames found to calculate FPS, using default")
                return 30

    def extract_audio(self):
        """Extract audio from the bag file and save it to a WAV file."""
        # Find the audio topic
        with rosbag.Bag(self.bag_path, 'r') as bag:
            topics_info = bag.get_type_and_topic_info()
            audio_topic = None
            
            # First look specifically for /audio topic
            if '/audio' in topics_info[1]:
                audio_topic = '/audio'
            else:
                # Look for topics with AudioData in their type
                for topic, info in topics_info[1].items():
                    if 'AudioData' in info.msg_type:
                        audio_topic = topic
                        break
                
                # If still not found, fall back to any topic with 'audio' in name
                if not audio_topic:
                    for topic in topics_info[1]:
                        if 'audio' in topic.lower() and 'info' not in topic.lower():
                            audio_topic = topic
                            break
            
            if not audio_topic:
                print(f"No audio topic found in {self.bag_path}")
                return None
            
            print(f"Selected audio topic: {audio_topic}")
            
            # Extract audio data
            audio_data = []
            audio_msg_count = 0
            sample_rate = 44100  # Default sample rate
            
            for _, msg, _ in bag.read_messages(topics=[audio_topic]):
                audio_msg_count += 1
                
                # Try different ways to extract audio data based on message structure
                try:
                    if hasattr(msg, 'data'):
                        # Most common case - data is in msg.data
                        if isinstance(msg.data, (list, tuple)):
                            # Data is already a list/tuple of values
                            audio_data.append(np.array(msg.data, dtype=np.float32))
                        elif isinstance(msg.data, bytes) or isinstance(msg.data, bytearray):
                            # Try to infer format from message attributes
                            if hasattr(msg, 'format') and msg.format == 'S16LE':
                                # 16-bit PCM
                                data_array = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
                            else:
                                # Default to float32 (most common in ROS audio)
                                data_array = np.frombuffer(msg.data, dtype=np.float32)
                            audio_data.append(data_array)
                        else:
                            # Already a numpy array or similar
                            audio_data.append(np.array(msg.data, dtype=np.float32))
                    elif hasattr(msg, 'audio_data'):
                        # Some messages store audio in audio_data field
                        audio_data.append(np.array(msg.audio_data, dtype=np.float32))
                except Exception as e:
                    print(f"Warning: Could not process audio message {audio_msg_count}: {e}")
                    continue
                
                # Try to get sample rate from the message
                if hasattr(msg, 'sample_rate'):
                    sample_rate = msg.sample_rate
                elif hasattr(msg, 'sr'):
                    sample_rate = msg.sr
                elif hasattr(msg, 'rate'):
                    sample_rate = msg.rate
                elif hasattr(msg, 'sampling_rate'):
                    sample_rate = msg.sampling_rate
        
        if audio_msg_count == 0:
            print(f"No audio messages found on topic {audio_topic}")
            return None
            
        print(f"Collected {sum(len(d) for d in audio_data if isinstance(d, np.ndarray))} audio samples from {audio_msg_count} messages")
        
        # Concatenate all audio data
        if audio_data:
            try:
                # Filter out empty arrays and check for valid arrays
                valid_audio_data = [d for d in audio_data if isinstance(d, np.ndarray) and d.size > 0]
                
                if not valid_audio_data:
                    print("No valid audio data found to concatenate")
                    return None
                
                all_audio = np.concatenate(valid_audio_data)
                
                # Ensure audio data is valid
                if not np.isfinite(all_audio).all():
                    print("Warning: Audio contains NaN or Inf values, replacing with zeros")
                    all_audio = np.nan_to_num(all_audio)
                
                # Apply RMS normalization
                if len(all_audio) > 0:
                    # Normalize audio
                    rms = np.sqrt(np.mean(all_audio**2))
                    target_rms = 0.1  # Target RMS level
                    gain = target_rms / (rms + 1e-9)  # Avoid division by zero
                    normalized_audio = all_audio * gain
                    
                    print(f"Applied RMS normalization: original RMS={rms:.6f}, target RMS={target_rms:.6f}")
                    
                    # Apply noise reduction if noise file is specified in config
                    if hasattr(self, 'cfg') and 'preprocessing' in self.cfg and 'noise_file' in self.cfg.preprocessing:
                        noise_file = self.cfg.preprocessing.noise_file
                        if os.path.exists(noise_file):
                            normalized_audio = self._apply_noise_reduction(normalized_audio, sample_rate, noise_file)
                            print(f"Applied noise reduction using {noise_file}")
                    
                    # Save the normalized audio
                    audio_path = os.path.join(self.audio_dir, f"{self.bag_name}_full.wav")
                    sf.write(audio_path, normalized_audio, sample_rate)
                    print(f"Saved audio file: {audio_path}")
                    
                    # Store audio data and sample rate for later use
                    self.audio_data = normalized_audio
                    self.audio_sr = sample_rate
                    
                    return audio_path
                
            except Exception as e:
                print(f"Error processing audio data: {str(e)}")
                return None
        else:
            print("No audio data was collected")
            return None

    def _apply_noise_reduction(self, audio_data, sr, noise_file):
        """Apply noise reduction to the audio data."""
        try:
            import noisereduce as nr
            
            # Load the noise sample
            noise_y, noise_sr = librosa.load(noise_file, sr=None)
            
            # Ensure both audio files have the same sample rate
            if sr != noise_sr:
                noise_y = librosa.resample(noise_y, orig_sr=noise_sr, target_sr=sr)
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sr,
                y_noise=noise_y,
                prop_decrease=1.0,
                stationary=False
            )
            
            return reduced_noise
            
        except Exception as e:
            print(f"Error applying noise reduction: {str(e)}")
            return audio_data

    def collect_image_messages(self):
        """Collect all image messages from the bag file."""
        print("Collecting image messages...")
        
        with rosbag.Bag(self.bag_path, 'r') as bag:
            # Find image topics
            topics_info = bag.get_type_and_topic_info()
            image_topics = []
            
            for topic, info in topics_info[1].items():
                if 'sensor_msgs/Image' in info.msg_type and 'camera1/color/image_raw' in topic:
                    image_topics.append(topic)
            
            if not image_topics:
                print(f"No camera1 image topics found in {self.bag_path}")
                return False
                
            print(f"Found image topics: {image_topics}")
            
            # Get the first message timestamp to use as reference
            first_timestamp = None
            # Use the first message without specifying a start time
            for _, _, t in bag.read_messages(topics=image_topics):
                first_timestamp = t.to_sec()
                break
            
            if first_timestamp is None:
                print("No messages found in image topics")
                return False
            
            print(f"First message timestamp: {first_timestamp}")
            
            # Collect image messages with timestamps
            for topic, msg, t in tqdm(bag.read_messages(topics=image_topics), 
                                     total=bag.get_message_count(image_topics[0])):
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    # Store both the absolute timestamp and the relative timestamp
                    self.image_msgs.append({
                        'timestamp': t.to_sec(),
                        'image': cv_img,
                        'header_stamp': msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()
                    })
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
        
        # Sort messages by timestamp
        self.image_msgs.sort(key=lambda x: x['timestamp'])
        
        print(f"Collected {len(self.image_msgs)} image messages")
        return len(self.image_msgs) > 0

    def segment_audio(self, cfg):
        """Segment the audio using the segment_audio function."""
        # Find the audio file
        audio_path = os.path.join(self.audio_dir, f"{self.bag_name}_full.wav")
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
        
        print(f"Segmenting audio from: {audio_path}")
        
        try:
            # Define default values for segmentation parameters
            default_config = {
                'window_length_seconds': 1.0,
                'window_stride_seconds': 0.1,
                'non_contact_threshold_factor': 0.3,
                'enable_squeezing': True,
                'min_segment_duration': 0.05,
                'max_segment_duration': 1.0,
                'min_amplitude_threshold': None,
                'max_amplitude_threshold': None,
                'dynamic_threshold_percentile': 90,
                'dynamic_threshold_offset': 0.15
            }
            
            # Define the get_config function
            def get_config_for_segment(key):
                try:
                    # First try to get from segmentation section
                    if hasattr(cfg, 'segmentation') and hasattr(cfg.segmentation, key):
                        return getattr(cfg.segmentation, key)
                    # Then try from data section
                    elif hasattr(cfg.data, key):
                        return getattr(cfg.data, key)
                    # Fall back to default
                    else:
                        return default_config[key]
                except (AttributeError, KeyError):
                    return default_config[key]
            
            # Get segments from segment_audio function with all parameters
            segments_result = segment_audio(
                audio_path, 
                window_length_seconds=get_config_for_segment('window_length_seconds'),
                window_stride_seconds=get_config_for_segment('window_stride_seconds'),
                non_contact_threshold_factor=get_config_for_segment('non_contact_threshold_factor'),
                enable_squeezing=get_config_for_segment('enable_squeezing'),
                min_segment_duration=get_config_for_segment('min_segment_duration'),
                max_segment_duration=get_config_for_segment('max_segment_duration'),
                min_amplitude_threshold=get_config_for_segment('min_amplitude_threshold'),
                max_amplitude_threshold=get_config_for_segment('max_amplitude_threshold'),
                dynamic_threshold_percentile=get_config_for_segment('dynamic_threshold_percentile'),
                dynamic_threshold_offset=get_config_for_segment('dynamic_threshold_offset')
            )
            
            # Handle different return types from segment_audio
            if isinstance(segments_result, tuple):
                self.audio_segments, self.force_data = segments_result
                print(f"Received segments and force data. Segment count: {len(self.audio_segments)}")
            else:
                self.audio_segments = segments_result
                self.force_data = None
                print(f"Received only segments. Segment count: {len(self.audio_segments)}")
            
            # Debug: Print segment structure
            if self.audio_segments and len(self.audio_segments) > 0:
                print(f"First segment structure: {type(self.audio_segments[0])}, length: {len(self.audio_segments[0])}")
                print(f"First segment content: {self.audio_segments[0]}")
                
                # Check if windows are present
                has_windows = False
                if len(self.audio_segments[0]) >= 4:
                    windows = self.audio_segments[0][3]
                    if windows is not None:
                        has_windows = True
                        print(f"Windows found in first segment: {len(windows)}")
                        if len(windows) > 0:
                            print(f"First window: {windows[0]}")
                
                if not has_windows:
                    print("No windows found in segments. Will generate windows from segments.")
                    # Generate windows for each segment
                    self._generate_windows_from_segments(cfg)
            
            # Count contact and non-contact segments
            if len(self.audio_segments) > 0 and len(self.audio_segments[0]) >= 3:
                contact_segments = sum(1 for seg in self.audio_segments if seg[2])  # is_contact is at index 2
                non_contact_segments = len(self.audio_segments) - contact_segments
                
                print(f"Contact segments: {contact_segments}")
                print(f"Non-contact segments: {non_contact_segments}")
            
            return len(self.audio_segments) > 0
            
        except Exception as e:
            print(f"Error segmenting audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_windows_from_segments(self, cfg):
        """Generate windows for segments that don't have them."""
        print("Generating windows from segments...")
        
        # Load audio to get sample rate
        audio_path = os.path.join(self.audio_dir, f"{self.bag_name}_full.wav")
        _, sr = librosa.load(audio_path, sr=None)
        
        # Get window parameters from config
        window_length_seconds = cfg.data.get('window_length_seconds', 1.0)
        window_stride_seconds = cfg.data.get('window_stride_seconds', 0.1)
        
        # Convert to samples
        window_length_samples = int(window_length_seconds * sr)
        window_stride_samples = int(window_stride_seconds * sr)
        
        print(f"Using window length: {window_length_seconds}s ({window_length_samples} samples)")
        print(f"Using window stride: {window_stride_seconds}s ({window_stride_samples} samples)")
        
        # Process each segment
        for i, segment in enumerate(self.audio_segments):
            start, end, is_contact = segment[:3]
            
            # Calculate number of windows
            segment_length = end - start
            num_windows = max(1, int((segment_length - window_length_samples) / window_stride_samples) + 1)
            
            # Generate windows
            windows = []
            for j in range(num_windows):
                window_start = start + j * window_stride_samples
                window_end = window_start + window_length_samples
                
                # Stop if the window would extend beyond the segment
                if window_end > end:
                    break
                    
                windows.append((window_start, window_end))
            
            # Add windows to the segment
            if len(segment) >= 4:
                segment[3] = windows
            else:
                # Extend the segment tuple to include windows
                self.audio_segments[i] = segment + (windows,)
            
            print(f"Generated {len(windows)} windows for segment {i}")

    def create_window_videos(self, include_non_contact=False):
        """Create video clips for each window within the audio segments."""
        if not self.audio_segments:
            print("No audio segments available. Run segment_audio first.")
            return False
        
        if not self.image_msgs:
            print("No image messages available. Run collect_image_messages first.")
            return False
        
        # Load the full audio file for extracting segments
        audio_path = os.path.join(self.audio_dir, f"{self.bag_name}_full.wav")
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        # Create a temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        
        # Create a metadata list for the dataset
        metadata = []
        
        # Get the start time of the bag file from the first image message
        bag_start_time = self.image_msgs[0]['timestamp'] if self.image_msgs else 0
        
        # Count total windows for progress bar
        total_windows = 0
        for segment in self.audio_segments:
            if len(segment) >= 4:  # Check if segment has windows
                windows = segment[3]
                if windows:
                    total_windows += len(windows)
        
        if total_windows == 0:
            print("No windows found in any segments. Check segment_audio output.")
            return False
        
        print(f"Found {total_windows} windows across {len(self.audio_segments)} segments")
        
        try:
            # Process each segment
            successful_videos = []
            window_counter = 0
            
            with tqdm(total=total_windows, desc="Creating window videos") as pbar:
                for i, segment in enumerate(self.audio_segments):
                    # Unpack the segment tuple - format depends on segment_audio implementation
                    if len(segment) >= 4:  # Newer format with windows
                        start, end, is_contact, windows = segment[:4]
                    else:  # Older format without windows
                        print(f"Segment {i} does not have window information, skipping")
                        continue
                    
                    # Skip non-contact segments if not included
                    if not is_contact and not include_non_contact:
                        continue
                    
                    # Skip if no windows
                    if not windows:
                        print(f"Segment {i} has no windows, skipping")
                        continue
                    
                    # Get segment type label for filenames
                    segment_type = "contact" if is_contact else "ambient"
                    
                    # Calculate window size based on the first window
                    if isinstance(windows[0], np.ndarray):
                        # Windows are audio data arrays, not indices
                        print(f"Windows in segment {i} are audio data arrays, not index tuples")
                        window_size = len(windows[0])
                        window_stride = window_size // 2  # Assume 50% overlap as default
                        
                        # Process each window in the segment
                        for j, window_data in enumerate(windows):
                            # Calculate window indices within the segment
                            window_start = start + (j * window_stride)
                            window_end = min(window_start + window_size, end)
                            
                            # Calculate window duration in seconds
                            window_start_time = window_start / sr
                            window_end_time = window_end / sr
                            window_duration = window_end_time - window_start_time
                            
                            # Calculate absolute timestamps for the window
                            abs_start_time = bag_start_time + window_start_time
                            abs_end_time = bag_start_time + window_end_time
                            
                            # Save the audio window
                            window_audio_filename = f"{self.bag_name}_segment_{i}_window_{j}_{segment_type}.wav"
                            window_audio_path = os.path.join(self.audio_dir, window_audio_filename)
                            sf.write(window_audio_path, window_data, sr)
                            
                            # Find images that fall within this window's time range
                            window_images = []
                            for msg in self.image_msgs:
                                if abs_start_time <= msg['timestamp'] <= abs_end_time:
                                    window_images.append(msg)
                            
                            if not window_images:
                                # Try using relative timing from the start of the bag
                                rel_start_time = window_start_time
                                rel_end_time = window_end_time
                                
                                for msg in self.image_msgs:
                                    rel_time = msg['timestamp'] - bag_start_time
                                    if rel_start_time <= rel_time <= rel_end_time:
                                        window_images.append(msg)
                                
                                if not window_images:
                                    # FALLBACK: Use the last available images
                                    num_frames_needed = int(window_duration * self.fps)
                                    last_images = self.image_msgs[-num_frames_needed:] if len(self.image_msgs) >= num_frames_needed else self.image_msgs[-1:]
                                    
                                    if len(last_images) < num_frames_needed:
                                        # If we don't have enough images, duplicate the last one
                                        last_image = last_images[-1]
                                        while len(last_images) < num_frames_needed:
                                            last_images.append(last_image)
                                    
                                    window_images = last_images
                                    print(f"Using {len(window_images)} fallback images for window {j} in segment {i}")
                            
                            # Create a temporary video file for this window
                            temp_video_path = os.path.join(temp_dir, f"temp_window_{i}_{j}.mp4")
                            
                            # Get frame dimensions from the first image
                            height, width, _ = window_images[0]['image'].shape
                            
                            # Create video writer
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
                            
                            # Write all frames to the video
                            for msg in window_images:
                                video_writer.write(msg['image'])
                            
                            video_writer.release()
                            
                            # Create the final video with audio using moviepy
                            window_video_filename = f"{self.bag_name}_segment_{i}_window_{j}_{segment_type}.mp4"
                            window_video_path = os.path.join(self.segments_dir, window_video_filename)
                            
                            # Create the video clip with audio
                            try:
                                video_clip = VideoFileClip(temp_video_path)
                                audio_clip = AudioFileClip(window_audio_path)
                                
                                # Make sure audio and video have the same duration
                                if audio_clip.duration > video_clip.duration:
                                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                                elif video_clip.duration > audio_clip.duration:
                                    video_clip = video_clip.subclip(0, audio_clip.duration)
                                
                                # Set the audio for the video
                                final_clip = video_clip.set_audio(audio_clip)
                                
                                # Write the final video
                                final_clip.write_videofile(window_video_path, codec='libx264', audio_codec='aac')
                                
                                # Close the clips
                                video_clip.close()
                                audio_clip.close()
                                
                                # Print the absolute path to make it very clear where the file is saved
                                abs_video_path = os.path.abspath(window_video_path)
                                print(f"Created window video: {abs_video_path}")
                                successful_videos.append(abs_video_path)
                                
                                # Add to metadata
                                category = "ambient" if not is_contact else map_to_category(self.bag_name)
                                metadata.append({
                                    'video_file': os.path.join("segments", window_video_filename),
                                    'audio_file': os.path.join("audio", window_audio_filename),
                                    'category': category,
                                    'duration': window_duration,
                                    'is_contact': is_contact,
                                    'segment_id': i,
                                    'window_id': j
                                })
                                
                            except Exception as e:
                                print(f"Error creating video for window {j} in segment {i}: {str(e)}")
                                
                            # Update progress bar
                            pbar.update(1)
                            window_counter += 1
                    else:
                        # Windows are tuples of (start, end) indices
                        # Process each window in the segment
                        for j, window in enumerate(windows):
                            window_start, window_end = window
                            
                            # Calculate window duration in seconds
                            window_start_time = window_start / sr
                            window_end_time = window_end / sr
                            window_duration = window_end_time - window_start_time
                            
                            # Calculate absolute timestamps for the window
                            abs_start_time = bag_start_time + window_start_time
                            abs_end_time = bag_start_time + window_end_time
                            
                            # Extract the audio window
                            window_audio = audio_data[window_start:window_end]
                            
                            # Save the audio window
                            window_audio_filename = f"{self.bag_name}_segment_{i}_window_{j}_{segment_type}.wav"
                            window_audio_path = os.path.join(self.audio_dir, window_audio_filename)
                            sf.write(window_audio_path, window_audio, sr)
                            
                            # Find images that fall within this window's time range
                            window_images = []
                            for msg in self.image_msgs:
                                if abs_start_time <= msg['timestamp'] <= abs_end_time:
                                    window_images.append(msg)
                            
                            if not window_images:
                                # Try using relative timing from the start of the bag
                                rel_start_time = window_start_time
                                rel_end_time = window_end_time
                                
                                for msg in self.image_msgs:
                                    rel_time = msg['timestamp'] - bag_start_time
                                    if rel_start_time <= rel_time <= rel_end_time:
                                        window_images.append(msg)
                                
                                if not window_images:
                                    # FALLBACK: Use the last available images
                                    num_frames_needed = int(window_duration * self.fps)
                                    last_images = self.image_msgs[-num_frames_needed:] if len(self.image_msgs) >= num_frames_needed else self.image_msgs[-1:]
                                    
                                    if len(last_images) < num_frames_needed:
                                        # If we don't have enough images, duplicate the last one
                                        last_image = last_images[-1]
                                        while len(last_images) < num_frames_needed:
                                            last_images.append(last_image)
                                    
                                    window_images = last_images
                            
                            # Create a temporary video file for this window
                            temp_video_path = os.path.join(temp_dir, f"temp_window_{i}_{j}.mp4")
                            
                            # Get frame dimensions from the first image
                            height, width, _ = window_images[0]['image'].shape
                            
                            # Create video writer
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
                            
                            # Write all frames to the video
                            for msg in window_images:
                                video_writer.write(msg['image'])
                            
                            video_writer.release()
                            
                            # Create the final video with audio using moviepy
                            window_video_filename = f"{self.bag_name}_segment_{i}_window_{j}_{segment_type}.mp4"
                            window_video_path = os.path.join(self.segments_dir, window_video_filename)
                            
                            # Create the video clip with audio
                            try:
                                video_clip = VideoFileClip(temp_video_path)
                                audio_clip = AudioFileClip(window_audio_path)
                                
                                # Make sure audio and video have the same duration
                                if audio_clip.duration > video_clip.duration:
                                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                                elif video_clip.duration > audio_clip.duration:
                                    video_clip = video_clip.subclip(0, audio_clip.duration)
                                
                                # Set the audio for the video
                                final_clip = video_clip.set_audio(audio_clip)
                                
                                # Write the final video
                                final_clip.write_videofile(window_video_path, codec='libx264', audio_codec='aac')
                                
                                # Close the clips
                                video_clip.close()
                                audio_clip.close()
                                
                                # Print the absolute path to make it very clear where the file is saved
                                abs_video_path = os.path.abspath(window_video_path)
                                print(f"Created window video: {abs_video_path}")
                                successful_videos.append(abs_video_path)
                                
                                # Add to metadata
                                category = "ambient" if not is_contact else map_to_category(self.bag_name)
                                metadata.append({
                                    'video_file': os.path.join("segments", window_video_filename),
                                    'audio_file': os.path.join("audio", window_audio_filename),
                                    'category': category,
                                    'duration': window_duration,
                                    'is_contact': is_contact,
                                    'segment_id': i,
                                    'window_id': j
                                })
                                
                            except Exception as e:
                                print(f"Error creating video for window {j} in segment {i}: {str(e)}")
                            
                            # Update progress bar
                            pbar.update(1)
                            window_counter += 1
            
            # Create a metadata CSV file
            if metadata:
                metadata_path = os.path.join(self.output_dir, 'windows_metadata.csv')
                pd.DataFrame(metadata).to_csv(metadata_path, index=False)
                abs_metadata_path = os.path.abspath(metadata_path)
                print(f"Created metadata file: {abs_metadata_path}")
            
            # Print a summary of all created files
            print("\n===== SUMMARY =====")
            print(f"Output directory: {os.path.abspath(self.output_dir)}")
            print(f"Created {len(successful_videos)} window videos from {window_counter} windows")
            print("==================\n")
            
            return True
            
        except Exception as e:
            print(f"Error creating window videos: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

def map_to_category(bag_name):
    """Map bag name to a standard category."""
    if 'leaf' in bag_name.lower():
        return 'leaf'
    elif 'twig' in bag_name.lower():
        return 'twig'
    elif 'trunk' in bag_name.lower():
        return 'trunk'
    else:
        return 'unknown'

@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Process bag files and create segment videos."""
    # Get the original working directory
    original_cwd = hydra.utils.get_original_cwd()
    
    # Get bag directory
    bag_dir = cfg.data.get('bag_dir', '')
    if not bag_dir:
        print("Error: No bag directory specified in config")
        return
    
    # Make path absolute
    if not os.path.isabs(bag_dir):
        bag_dir = os.path.join(original_cwd, bag_dir)
    
    # Get output directory
    output_dir = cfg.data.get('output_dir', 'segmented_videos')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(original_cwd, output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for bag files in: {os.path.abspath(bag_dir)}")
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")
    
    # Find bag files
    recursive = cfg.data.get('recursive', False)
    if recursive:
        bag_files = glob.glob(os.path.join(bag_dir, '**/*.bag'), recursive=True)
        # Also try uppercase extension
        bag_files.extend(glob.glob(os.path.join(bag_dir, '**/*.BAG'), recursive=True))
    else:
        bag_files = glob.glob(os.path.join(bag_dir, '*.bag'))
        # Also try uppercase extension
        bag_files.extend(glob.glob(os.path.join(bag_dir, '*.BAG')))
    
    if not bag_files:
        print(f"No bag files found in {bag_dir}")
        return
    
    print(f"Found {len(bag_files)} bag files to process:")
    for bag_file in bag_files:
        print(f"  {os.path.basename(bag_file)}")
    
    # Process each bag file
    for bag_path in tqdm(bag_files, desc="Processing bags"):
        try:
            # Create a unique output directory for this bag
            bag_name = os.path.splitext(os.path.basename(bag_path))[0]
            bag_output_dir = os.path.join(output_dir, bag_name)
            
            # Create the segmenter
            segmenter = BagVideoSegmenter(bag_path, bag_output_dir, cfg)
            
            # Extract audio
            audio_path = segmenter.extract_audio()
            if not audio_path:
                print(f"Failed to extract audio from {bag_path}, skipping")
                continue
            
            # Collect image messages
            if not segmenter.collect_image_messages():
                print(f"Failed to collect image messages from {bag_path}, skipping")
                continue
            
            # Segment audio
            if not segmenter.segment_audio(cfg):
                print(f"Failed to segment audio from {bag_path}, skipping")
                continue
            
            # Create window videos instead of segment videos
            include_non_contact = cfg.data.get('include_non_contact', False)
            
            # Check if we should use window videos or segment videos
            use_windows = cfg.data.get('use_windows', True)
            if use_windows:
                print("Creating window videos...")
                segmenter.create_window_videos(include_non_contact=include_non_contact)
            else:
                print("Creating segment videos...")
                segmenter.create_segment_videos(include_non_contact=include_non_contact)
            
        except Exception as e:
            print(f"Error processing bag {bag_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # After processing all bags, consolidate the dataset
    consolidate = cfg.data.get('consolidate_dataset', True)
    if consolidate:
        consolidate_dataset(cfg, original_cwd)

def consolidate_dataset(cfg, original_cwd):
    """Consolidate all segment videos into a single dataset folder."""
    # Get the output directory where individual bag results are stored
    output_dir = cfg.data.get('output_dir', 'segmented_videos')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(original_cwd, output_dir)
    
    # Create a dataset directory using the name from config
    if 'output' in cfg and 'dataset_dir' in cfg.output:
        dataset_dir = cfg.output.dataset_dir
    else:
        dataset_dir = cfg.data.get('dataset_dir', 'audio_visual_dataset')
    
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(original_cwd, dataset_dir)
    
    # Create subdirectories for videos only (no separate audio directory)
    videos_dir = os.path.join(dataset_dir, 'videos')
    os.makedirs(videos_dir, exist_ok=True)
    
    print(f"\nConsolidating dataset into: {os.path.abspath(dataset_dir)}")
    
    # Get minimum duration from config
    min_duration = cfg.data.get('window_length_seconds', 1.0)
    print(f"Using minimum video duration: {min_duration} seconds")
    
    # Find all bag output directories
    bag_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    # Collect simplified metadata (just video path and category)
    simplified_metadata = []
    video_count = 0
    skipped_count = 0
    
    # Process each bag directory
    for bag_dir in tqdm(bag_dirs, desc="Consolidating dataset"):
        bag_path = os.path.join(output_dir, bag_dir)
        
        # Find metadata file
        metadata_file = os.path.join(bag_path, 'windows_metadata.csv')
        if not os.path.exists(metadata_file):
            metadata_file = os.path.join(bag_path, 'segments_metadata.csv')
            
        if os.path.exists(metadata_file):
            # Load metadata
            metadata = pd.read_csv(metadata_file)
            
            # Process each entry
            for _, row in metadata.iterrows():
                # Get source paths
                video_src = os.path.join(bag_path, row['video_file'])
                
                # Skip if video doesn't exist
                if not os.path.exists(video_src):
                    continue
                
                # Check video duration
                duration = row.get('duration', 0)
                if duration < min_duration:
                    skipped_count += 1
                    continue
                
                # Get category
                category = row['category']
                
                # Create a unique identifier
                segment_id = row.get('segment_id', 0)
                window_id = row.get('window_id', 0)
                unique_id = f"{bag_dir}_seg{segment_id}_win{window_id}"
                
                # Create new filename
                video_filename = f"{category}_{unique_id}.mp4"
                
                # Create destination path
                video_dst = os.path.join(videos_dir, video_filename)
                
                # Copy video file
                shutil.copy2(video_src, video_dst)
                
                # Add simplified metadata entry
                simplified_metadata.append({
                    'video': os.path.join('videos', video_filename),
                    'category': category
                })
                
                video_count += 1
    
    # Create simplified metadata file
    if simplified_metadata:
        df = pd.DataFrame(simplified_metadata)
        csv_path = os.path.join(dataset_dir, 'dataset.csv')
        df.to_csv(csv_path, index=False)
        
        # Print statistics
        print(f"\nDataset consolidated successfully!")
        print(f"Total videos included: {video_count}")
        print(f"Videos skipped (too short): {skipped_count}")
        print(f"Dataset CSV: {os.path.abspath(csv_path)}")
        
        # Print category statistics
        if 'category' in df.columns:
            print("\nVideos per category:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count}")
    else:
        print("No metadata found to consolidate")

if __name__ == "__main__":
    main() 