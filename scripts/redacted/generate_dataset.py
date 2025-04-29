    
# import rosbag
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# import os
# import wave
# import array
# from rosbag.bag import BagMessage
# from tqdm import tqdm
# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
# import tempfile
# import shutil
# import matplotlib.pyplot as plt
# import librosa
# import soundfile as sf
# from collections import namedtuple
#
# class RosBagExtractor:
#     def __init__(self, bag_path, output_dir=None):
#         """Initialize the extractor with bag file path and optional output directory."""
#         self.bag_path = bag_path
#
#         if output_dir:
#             self.output_dir = output_dir
#         else:
#             # Create output dir based on bag path structure
#             bag_dir = os.path.dirname(bag_path)
#             bag_name = os.path.splitext(os.path.basename(bag_path))[0]
#
#             # Create output directory preserving the original path structure
#             rel_path = os.path.relpath(bag_dir)
#             self.output_dir = os.path.join(os.getcwd(), 'output', rel_path, bag_name)
#
#         self.bridge = CvBridge()
#
#         # Create all required directories
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.image_dir = os.path.join(self.output_dir, 'images')
#         self.audio_dir = os.path.join(self.output_dir, 'audio')
#         os.makedirs(self.image_dir, exist_ok=True)
#         os.makedirs(self.audio_dir, exist_ok=True)
#
#         # Add camera info storage
#         self.camera_info = None
#         self._load_camera_info()
#
#     def _load_camera_info(self):
#         """Load camera calibration info from the bag file."""
#         camera_info_topic = '/camera1/color/camera_info'
#
#         with rosbag.Bag(self.bag_path, 'r') as bag:
#             # Get the first camera info message
#             for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
#                 self.camera_info = {
#                     'height': msg.height,
#                     'width': msg.width,
#                     'K': msg.K,
#                     'D': msg.D,
#                     'R': msg.R,
#                     'P': msg.P,
#                     'distortion_model': msg.distortion_model
#                 }
#                 break  # Only need the first message since they're all identical
#
#         if not self.camera_info:
#             print(f"Warning: No camera info found on topic {camera_info_topic}")
#
#     def _get_topic_by_type(self, type_filter):
#         """Get the first topic that matches the given message type filter."""
#         with rosbag.Bag(self.bag_path, 'r') as bag:
#             topics = bag.get_type_and_topic_info()[1]
#             for topic_name, topic_info in topics.items():
#                 if type_filter in topic_info.msg_type.lower():
#                     return topic_name
#         return None
#
#     def extract_images(self):
#         """Extract images from the first image topic found."""
#         image_topic = self._get_topic_by_type('image')
#         if not image_topic:
#             print("No image topics found in bag file")
#             return
#
#         print(f"Found image topic: {image_topic}")
#         print("Extracting images...")
#
#         if self.camera_info:
#             print(f"Processing images with camera calibration:")
#             print(f"Resolution: {self.camera_info['width']}x{self.camera_info['height']}")
#             print(f"Distortion model: {self.camera_info['distortion_model']}")
#
#         with rosbag.Bag(self.bag_path, 'r') as bag:
#             total_msgs = bag.get_message_count(image_topic)
#
#             for i, data in enumerate(tqdm(bag.read_messages(topics=[image_topic]), total=total_msgs)):
#                 try:
#                     cv_img = self.bridge.imgmsg_to_cv2(data.message, desired_encoding='bgr8')
#                     timestamp = str(data.timestamp.to_nsec())
#                     filename = os.path.join(self.image_dir, f'frame_{timestamp}.jpg')
#                     cv2.imwrite(filename, cv_img)
#                 except Exception as e:
#                     print(f"Error processing image message {i}: {str(e)}")
#
#     def _analyze_audio_quality(self, audio_data):
#         """Analyze audio quality and return a grade."""
#         if not audio_data:
#             return "F - No audio data found"
#
#         # Convert to numpy array for analysis
#         audio_array = np.array(audio_data)
#
#         # Calculate basic statistics
#         peak_amplitude = np.max(np.abs(audio_array))
#         rms = np.sqrt(np.mean(np.square(audio_array)))
#         crest_factor = peak_amplitude / rms if rms > 0 else 0
#
#         # Grade based on various factors
#         grade = "A"
#         reasons = []
#
#         # Check for clipping
#         if peak_amplitude > 30000:  # Close to int16 max (32767)
#             grade = "B"
#             reasons.append("Some audio clipping detected")
#
#         # Check for very low volume
#         if rms < 1000:
#             grade = "C"
#             reasons.append("Low audio levels")
#
#         # Check for poor dynamic range
#         if crest_factor < 3:
#             grade = "C"
#             reasons.append("Limited dynamic range")
#
#         # Check for silence or near-silence
#         if rms < 100:
#             grade = "F"
#             reasons.append("Extremely low or no audio signal")
#
#         return f"{grade} - {'; '.join(reasons)}" if reasons else f"{grade} - Good audio quality"
#
#     def extract_audio(self):
#         """Extract audio from the first audio topic found using librosa."""
#         audio_topic = self._get_topic_by_type('audio')
#         if not audio_topic:
#             print("No audio topics found in bag file")
#             return
#
#         print(f"Found audio topic: {audio_topic}")
#         print("Extracting audio...")
#
#         audio_data = []
#         timestamp = None
#
#         with rosbag.Bag(self.bag_path, 'r') as bag:
#             total_msgs = bag.get_message_count(audio_topic)
#
#             for data in tqdm(bag.read_messages(topics=[audio_topic]), total=total_msgs):
#                 try:
#                     if type(data) != BagMessage:
#                         continue
#
#                     # Get audio data and reshape to stereo channels
#                     float_samples = np.array(data.message.data, dtype=np.float32)
#                     float_samples = float_samples.reshape(-1, 2)
#
#                     audio_data.append(float_samples)
#
#                     if timestamp is None:
#                         timestamp = data.timestamp.to_nsec()
#
#                 except Exception as e:
#                     print(f"Error processing audio message: {str(e)}")
#
#         if audio_data and timestamp:
#             # Concatenate all audio chunks, maintaining stereo channels
#             audio_array = np.concatenate(audio_data, axis=0)
#
#             # Save audio file using soundfile
#             wav_filename = os.path.join(self.audio_dir, f'audio_{timestamp}.wav')
#             sf.write(wav_filename, audio_array, 44100, 'FLOAT')
#
#             print(f"Saved audio file: {wav_filename}")
#
#             # Create waveform and spectrogram visualization using matplotlib
#             plt.figure(figsize=(15, 10))
#
#             # Plot waveform for both channels
#             plt.subplot(3, 1, 1)
#             time = np.arange(len(audio_array)) / 44100
#             plt.plot(time, audio_array[:, 0], alpha=0.5, label='Left')
#             plt.plot(time, audio_array[:, 1], alpha=0.5, label='Right')
#             plt.title('Stereo Waveform')
#             plt.legend()
#             plt.xlabel('Time (s)')
#             plt.ylabel('Amplitude')
#
#             # Plot spectrogram for left channel
#             plt.subplot(3, 1, 2)
#             D = librosa.stft(audio_array[:, 0])
#             S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#             plt.imshow(S_db, aspect='auto', origin='lower')
#             plt.colorbar(format='%+2.0f dB')
#             plt.title('Left Channel Spectrogram')
#             plt.ylabel('Frequency Bin')
#
#             # Plot spectrogram for right channel
#             plt.subplot(3, 1, 3)
#             D = librosa.stft(audio_array[:, 1])
#             S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#             plt.imshow(S_db, aspect='auto', origin='lower')
#             plt.colorbar(format='%+2.0f dB')
#             plt.title('Right Channel Spectrogram')
#             plt.xlabel('Time Frame')
#             plt.ylabel('Frequency Bin')
#
#             plt.tight_layout()
#             plt.savefig(os.path.join(self.audio_dir, f'audio_analysis_{timestamp}.png'))
#             plt.close()
#
#     def draw_3d_vector(self, frame, force, origin=(500, 70), scale=15):
#         """Draw 3D force vector visualization"""
#         # Define 3D to 2D projection matrix (more tilted for better 3D perspective)
#         proj_matrix = np.array([
#             [1, -0.3, 0.3],    # x-axis
#             [0, -0.5, -1],     # y-axis
#             [-0.5, -0.8, 0.3]  # z-axis
#         ]) * scale
#
#         # Colors (BGR format)
#         red = (0, 0, 255)    # x-axis
#         green = (0, 255, 0)  # y-axis
#         blue = (255, 0, 0)   # z-axis
#         yellow = (0, 255, 255)  # force vector
#
#         # Draw coordinate system axes
#         axis_length = 20
#         for i, color in enumerate([(red, 'X'), (green, 'Y'), (blue, 'Z')]):
#             axis = np.zeros(3)
#             axis[i] = axis_length
#             projected = np.dot(proj_matrix, axis)
#             end_point = (
#                 int(origin[0] + projected[0]),
#                 int(origin[1] + projected[1])
#             )
#             cv2.line(frame, origin, end_point, color[0], 2)
#             # Add axis labels with slight offset for better visibility
#             label_offset = 5
#             label_pos = (end_point[0] + label_offset, end_point[1] + label_offset)
#             cv2.putText(frame, color[1], label_pos, 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[0], 1)
#
#         # Draw force vector
#         force_vec = np.array([force.x, force.y, force.z])
#         # Normalize force vector for visualization
#         magnitude = np.linalg.norm(force_vec)
#         if magnitude > 0:
#             force_vec = force_vec / magnitude * min(magnitude, axis_length)
#
#         projected_force = np.dot(proj_matrix, force_vec)
#         force_end = (
#             int(origin[0] + projected_force[0]),
#             int(origin[1] + projected_force[1])
#         )
#
#         # Draw force vector
#         cv2.line(frame, origin, force_end, yellow, 2)
#
#         # Add magnitude text with better positioning
#         cv2.putText(frame, f"{magnitude:.1f}N", 
#                     (origin[0] - 30, origin[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1)
#
#         return frame
#
#     def draw_ft_data(self, frame, force, torque, camera2_frame=None, audio_data=None):
#         """Helper function to draw force/torque data and audio visualization on frame"""
#         # Get frame dimensions
#         height, width = frame.shape[:2]
#
#         # If camera2 frame exists, overlay it in bottom right
#         if camera2_frame is not None:
#             # Calculate size for picture-in-picture (1/4 of original size)
#             pip_height = height // 4
#             pip_width = width // 4
#
#             # Resize camera2 frame
#             pip_frame = cv2.resize(camera2_frame, (pip_width, pip_height))
#
#             # Calculate position for bottom right corner
#             x_offset = width - pip_width - 10  # 10 pixels padding
#             y_offset = height - pip_height - 10  # 10 pixels padding
#
#             # Create a semi-transparent overlay for PIP background
#             cv2.rectangle(frame, 
#                          (x_offset-2, y_offset-2), 
#                          (x_offset+pip_width+2, y_offset+pip_height+2), 
#                          (0, 0, 0), 
#                          2)  # Border
#
#             # Overlay the PIP frame
#             frame[y_offset:y_offset+pip_height, x_offset:x_offset+pip_width] = pip_frame
#
#         # Create a semi-transparent overlay for left side data
#         overlay = frame.copy()
#         # Draw background rectangle for left side data
#         cv2.rectangle(overlay, (10, 10), (220, 140), (0, 0, 0), -1)
#         # Draw background rectangle for 3D visualization (top-right)
#         cv2.rectangle(overlay, (400, 10), (600, 140), (0, 0, 0), -1)
#         # Add transparency
#         frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
#
#         # Colors for values (BGR format)
#         red = (0, 0, 255)
#         green = (0, 255, 0)
#         blue = (255, 0, 0)
#         white = (255, 255, 255)
#
#         # Add text with smaller font size
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.4
#
#         # Force line
#         y_offset = 25
#         cv2.putText(frame, "F:", (15, y_offset), font, font_scale, white, 1)
#         cv2.putText(frame, f"x:{force.x:6.2f}", (35, y_offset), font, font_scale, red, 1)
#         cv2.putText(frame, f"y:{force.y:6.2f}", (95, y_offset), font, font_scale, green, 1)
#         cv2.putText(frame, f"z:{force.z:6.2f}", (155, y_offset), font, font_scale, blue, 1)
#
#         # Torque line
#         y_offset = 45
#         cv2.putText(frame, "T:", (15, y_offset), font, font_scale, white, 1)
#         cv2.putText(frame, f"x:{torque.x:6.3f}", (35, y_offset), font, font_scale, red, 1)
#         cv2.putText(frame, f"y:{torque.y:6.3f}", (95, y_offset), font, font_scale, green, 1)
#         cv2.putText(frame, f"z:{torque.z:6.3f}", (155, y_offset), font, font_scale, blue, 1)
#
#         # Draw 3D visualization
#         frame = self.draw_3d_vector(frame, force)
#
#         # Audio visualization (if provided)
#         if audio_data is not None:
#             y_offset = 110
#             cv2.putText(frame, "A:", (15, y_offset), font, font_scale, white, 1)
#
#             # Draw audio magnitude bar
#             bar_start = 35
#             bar_width = 175
#             bar_height = 15
#
#             # Calculate RMS energy using librosa (handle 1D array)
#             audio_data = np.asarray(audio_data).flatten()  # Ensure 1D array
#             magnitude = librosa.feature.rms(y=audio_data, frame_length=2048)[0][-1]
#             normalized_magnitude = min(1.0, magnitude * 4)  # Adjust scaling factor as needed
#
#             # Draw background bar
#             cv2.rectangle(frame, 
#                          (bar_start, y_offset - bar_height + 5),
#                          (bar_start + bar_width, y_offset - 2),
#                          (50, 50, 50),
#                          -1)
#
#             # Draw magnitude bar
#             bar_length = int(normalized_magnitude * bar_width)
#             if bar_length > 0:
#                 if normalized_magnitude < 0.5:
#                     color = (0, 255, 0)
#                 elif normalized_magnitude < 0.8:
#                     color = (0, 255, 255)
#                 else:
#                     color = (0, 0, 255)
#
#                 cv2.rectangle(frame, 
#                              (bar_start, y_offset - bar_height + 5),
#                              (bar_start + bar_length, y_offset - 2),
#                              color,
#                              -1)
#
#         return frame
#
#     def create_video(self, fps=30):
#         """Create a video from extracted images and audio."""
#         print("Creating video...")
#
#         bag_name = os.path.splitext(os.path.basename(self.bag_path))[0]
#         temp_dir = tempfile.mkdtemp()
#
#         try:
#             temp_video = os.path.join(temp_dir, 'temp_video.mp4')
#             final_video_path = os.path.join(self.output_dir, f'{bag_name}_video.mp4')
#
#             with rosbag.Bag(self.bag_path, 'r') as bag:
#                 print("Collecting messages...")
#                 image_msgs = []
#                 camera2_msgs = {}
#                 ft_msgs = {}
#                 audio_msgs = {}
#
#                 for topic, msg, t in bag.read_messages():
#                     timestamp = t.to_nsec()
#
#                     if topic == '/camera1/color/image_raw':
#                         try:
#                             cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#                             image_msgs.append({
#                                 'timestamp': timestamp,
#                                 'image': cv_img,
#                                 'header_stamp': msg.header.stamp.to_nsec()
#                             })
#                         except Exception as e:
#                             print(f"Error processing image: {str(e)}")
#
#                     elif topic == '/camera2/color/image_raw':
#                         try:
#                             cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#                             camera2_msgs[msg.header.stamp.to_nsec()] = cv_img
#                         except Exception as e:
#                             print(f"Error processing camera2: {str(e)}")
#
#                     elif topic == '/ft_sensor':
#                         # Store force/torque data with bag timestamp
#                         ft_msgs[timestamp] = msg
#
#                     elif topic == '/audio':  # Use the correct audio data topic
#                         try:
#                             # Store audio data with its timestamp
#                             audio_msgs[timestamp] = msg.data  # AudioData message has a 'data' field
#                         except Exception as e:
#                             print(f"Error processing audio: {str(e)}")
#
#                 if not image_msgs:
#                     print("No valid images found to create video")
#                     return
#
#                 # Sort messages by timestamp
#                 image_msgs.sort(key=lambda x: x['timestamp'])
#                 ft_timestamps = sorted(ft_msgs.keys())
#                 audio_timestamps = sorted(audio_msgs.keys())
#
#                 print(f"Collected messages:")
#                 print(f"Images: {len(image_msgs)}")
#                 print(f"Force/Torque: {len(ft_msgs)}")
#                 print(f"Camera2: {len(camera2_msgs)}")
#                 print(f"Audio: {len(audio_msgs)}")
#
#                 # Get frame dimensions
#                 height, width, layers = image_msgs[0]['image'].shape
#
#                 # Create video writer
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
#
#                 print("Writing frames to video...")
#                 max_time_diff = int(0.1 * 1e9)  # 100ms in nanoseconds
#
#                 for frame_data in tqdm(image_msgs):
#                     frame = frame_data['image']
#                     frame_timestamp = frame_data['timestamp']
#
#                     # Find closest force/torque data
#                     ft_data = None
#                     if ft_timestamps:
#                         closest_ft = min(ft_timestamps, key=lambda x: abs(x - frame_timestamp))
#                         ft_data = ft_msgs[closest_ft]  # Always use closest force data
#                     else:
#                         # Create empty force/torque data if no force data available at all
#                         Vector3 = namedtuple('Vector3', ['x', 'y', 'z'])
#                         ft_data = namedtuple('FTData', ['force', 'torque'])(
#                             force=Vector3(0.0, 0.0, 0.0),
#                             torque=Vector3(0.0, 0.0, 0.0)
#                         )
#
#                     # Find closest camera2 frame
#                     camera2_frame = None
#                     if camera2_msgs:
#                         closest_camera2 = min(camera2_msgs.keys(), 
#                                             key=lambda x: abs(x - frame_timestamp))
#                         if abs(closest_camera2 - frame_timestamp) < max_time_diff:
#                             camera2_frame = camera2_msgs[closest_camera2]
#
#                     # Find closest audio data
#                     audio_data = None
#                     if audio_timestamps:
#                         closest_audio = min(audio_timestamps, 
#                                           key=lambda x: abs(x - frame_timestamp))
#                         if abs(closest_audio - frame_timestamp) < max_time_diff:
#                             audio_data = audio_msgs[closest_audio]
#
#                     # Draw all data on frame (now always including force widget)
#                     annotated_frame = self.draw_ft_data(frame, ft_data.force, ft_data.torque, 
#                                                       camera2_frame, audio_data)
#                     video_writer.write(annotated_frame)
#
#                 video_writer.release()
#
#                 # Ensure video file exists before proceeding
#                 if not os.path.exists(temp_video):
#                     print(f"Error: Temporary video file {temp_video} was not created")
#                     return
#
#                 # Check for audio messages instead of audio topic
#                 if audio_msgs:
#                     print("Adding audio to video...")
#                     self.extract_audio()
#
#                     # Find the extracted audio file
#                     audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
#                     if audio_files:
#                         audio_path = os.path.join(self.audio_dir, audio_files[0])
#
#                         video_clip = VideoFileClip(temp_video)
#                         audio_clip = AudioFileClip(audio_path)
#
#                         if audio_clip.duration > video_clip.duration:
#                             audio_clip = audio_clip.subclip(0, video_clip.duration)
#
#                         final_clip = video_clip.set_audio(audio_clip)
#                         final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
#
#                         video_clip.close()
#                         audio_clip.close()
#                     else:
#                         print("No audio file found to add to video")
#                         shutil.copy2(temp_video, final_video_path)
#                 else:
#                     print("No audio messages found in bag file")
#                     shutil.copy2(temp_video, final_video_path)
#
#                 print(f"Video created successfully at: {final_video_path}")
#         finally:
#             shutil.rmtree(temp_dir)
#---------------------------
import argparse
import os
import glob
import shutil
from tqdm import tqdm
from extract import rosbagextractor

def find_bags_recursive(directory):
    """find all .bag files recursively in the given directory."""
    return glob.glob(os.path.join(directory, '**/*.bag'), recursive=true)

def process_bag(bag_path, output_dir=none, fps=30):
    # get bag file name without extension
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    
    # create bag-specific output directory
    if output_dir:
        bag_output_dir = os.path.join(output_dir, bag_name)
    else:
        bag_output_dir = os.path.join('output', bag_name)
    
    # create extractor instance with bag-specific output directory
    extractor = rosbagextractor(bag_path, bag_output_dir)

    # extract images
    extractor.extract_images()

    # extract audio 
    extractor.extract_audio()

    # create video with audio
    extractor.create_video(fps=fps)

def get_unique_filename(dest_dir, filename):
    """generate a unique filename by adding a number if the file already exists."""
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(os.path.join(dest_dir, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
        
    return new_filename

def collect_bags(source_dir, destination_dir, folder_name):
    """collect all bag files from source directory and copy to destination/folder_name."""
    # create destination directory with specified folder name
    dest_path = os.path.join(destination_dir, folder_name)
    os.makedirs(dest_path, exist_ok=true)

    # find all bag files
    bag_files = find_bags_recursive(source_dir)
    
    if not bag_files:
        print(f"no .bag files found in {source_dir}")
        return

    print(f"found {len(bag_files)} bag files. copying to {dest_path}...")
    
    # copy each bag file
    for bag_path in tqdm(bag_files):
        original_name = os.path.basename(bag_path)
        unique_name = get_unique_filename(dest_path, original_name)
        final_dest = os.path.join(dest_path, unique_name)
        
        if unique_name != original_name:
            print(f"Renamed '{original_name}' to '{unique_name}' to avoid duplicate")
            
        shutil.copy2(bag_path, final_dest)

    print(f"Successfully copied {len(bag_files)} bag files to {dest_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract data from ROS bags or collect bag files')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract data from ROS bags')
    extract_parser.add_argument('path', help='Path to ROS bag file or directory containing ROS bags')
    extract_parser.add_argument('-o', '--output', help='Output directory for extracted files', default='output')
    extract_parser.add_argument('--fps', type=int, default=30, help='FPS for video creation')
    extract_parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search for bag files in subdirectories')

    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect bag files recursively')
    collect_parser.add_argument('source', help='Source directory to search for bag files')
    collect_parser.add_argument('destination', help='Destination directory for collected bag files')
    collect_parser.add_argument('folder_name', help='Name of the folder to store bag files')

    args = parser.parse_args()

    if args.command == 'extract':
        if os.path.isfile(args.path):
            # Process single bag file
            process_bag(args.path, args.output, args.fps)
        elif os.path.isdir(args.path):
            if args.recursive:
                # Find all bag files recursively
                bag_files = find_bags_recursive(args.path)
            else:
                # Only find bag files in the current directory
                bag_files = [os.path.join(args.path, f) for f in os.listdir(args.path) 
                           if f.endswith('.bag')]
            
            if not bag_files:
                print(f"No .bag files found in {args.path}")
                return

            # Process all found bag files
            for bag_path in bag_files:
                print(f"\nProcessing bag file: {os.path.basename(bag_path)}")
                process_bag(bag_path, args.output, args.fps)
        else:
            print(f"Error: {args.path} is not a valid file or directory")
    
    elif args.command == 'collect':
        collect_bags(args.source, args.destination, args.folder_name)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

import argparse

import os
import rosbag
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import glob

def extract_bag_data(bag_path, label=None):
    """Extract data from a single bag file."""
    data = defaultdict(list)
    
    print(f"Processing: {os.path.basename(bag_path)}")
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/ft_sensor']):
            # Store timestamp and label
            data['timestamp'].append(t.to_nsec())
            if label is not None:
                data['label'].append(label)
            
            # Store force data
            data['force_x'].append(msg.force.x)
            data['force_y'].append(msg.force.y)
            data['force_z'].append(msg.force.z)
            
            # Store torque data
            data['torque_x'].append(msg.torque.x)
            data['torque_y'].append(msg.torque.y)
            data['torque_z'].append(msg.torque.z)
    
    return pd.DataFrame(data)

def find_bags_recursive(directory):
    """Find all .bag files recursively in the given directory."""
    return glob.glob(os.path.join(directory, '**/*.bag'), recursive=True)

def main():
    parser = argparse.ArgumentParser(description='Generate dataset from ROS bags')
    parser.add_argument('source', help='Source directory containing ROS bags')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--label', help='Optional label for the data')
    parser.add_argument('--append', action='store_true', help='Append to existing CSV file')
    
    args = parser.parse_args()
    
    # Find all bag files
    bag_files = find_bags_recursive(args.source)
    
    if not bag_files:
        print(f"No .bag files found in {args.source}")
        return
    
    print(f"Found {len(bag_files)} bag files")
    
    # Process each bag file
    all_data = []
    for bag_path in tqdm(bag_files):
        df = extract_bag_data(bag_path, args.label)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("No data extracted from bag files")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')
    
    # Save to CSV
    mode = 'a' if args.append else 'w'
    header = not args.append or not os.path.exists(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(args.output, mode=mode, header=header, index=False)
    
    print(f"Dataset saved to: {args.output}")
    print(f"Total samples: {len(combined_df)}")

if __name__ == '__main__':
    main()
