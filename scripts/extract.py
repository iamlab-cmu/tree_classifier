import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import wave
import array
from rosbag.bag import BagMessage
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import tempfile
import shutil
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from collections import namedtuple
import sys
import pandas as pd
from omegaconf import OmegaConf
import noisereduce as nr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from segment_audio import segment_audio


class RosBagExtractor:
    def __init__(self, bag_path, output_dir=None):
        """Initialize the extractor with bag file path and optional output directory."""
        self.bag_path = bag_path

        if output_dir:
            self.output_dir = output_dir
        else:
            bag_dir = os.path.dirname(bag_path)
            bag_name = os.path.splitext(os.path.basename(bag_path))[0]

            rel_path = os.path.relpath(bag_dir)
            self.output_dir = os.path.join(os.getcwd(), "output", rel_path, bag_name)

        self.bridge = CvBridge()

        os.makedirs(self.output_dir, exist_ok=True)
        self.image_dir = os.path.join(self.output_dir, "images")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.force_dir = os.path.join(self.output_dir, "force")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.force_dir, exist_ok=True)

        self.camera_info = None
        self._load_camera_info()

        self.audio_segments = None
        self.audio_data = None
        self.audio_sr = None

        self.fps = None
        self._calculate_fps()

    def _load_camera_info(self):
        """Load camera calibration info from the bag file."""
        camera_info_topic = "/camera1/color/camera_info"

        with rosbag.Bag(self.bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
                self.camera_info = {
                    "height": msg.height,
                    "width": msg.width,
                    "K": msg.K,
                    "D": msg.D,
                    "R": msg.R,
                    "P": msg.P,
                    "distortion_model": msg.distortion_model,
                }
                break  # Only need the first message since they're all identical

        if not self.camera_info:
            print(f"Warning: No camera info found on topic {camera_info_topic}")

    def _get_topic_by_type(self, type_filter):
        """Get the first topic that matches the given message type filter."""
        with rosbag.Bag(self.bag_path, "r") as bag:
            topics = bag.get_type_and_topic_info()[1]
            for topic_name, topic_info in topics.items():
                if type_filter in topic_info.msg_type.lower():
                    return topic_name
        return None

    def extract_images(self):
        """Extract images from the first image topic found."""
        image_topic = self._get_topic_by_type("image")
        if not image_topic:
            print("No image topics found in bag file")
            return

        print(f"Found image topic: {image_topic}")
        print("Extracting images...")

        if self.camera_info:
            print(f"Processing images with camera calibration:")
            print(
                f"Resolution: {self.camera_info['width']}x{self.camera_info['height']}"
            )
            print(f"Distortion model: {self.camera_info['distortion_model']}")

        with rosbag.Bag(self.bag_path, "r") as bag:
            total_msgs = bag.get_message_count(image_topic)

            for i, data in enumerate(
                tqdm(bag.read_messages(topics=[image_topic]), total=total_msgs)
            ):
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(
                        data.message, desired_encoding="bgr8"
                    )
                    timestamp = str(data.timestamp.to_nsec())
                    filename = os.path.join(self.image_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(filename, cv_img)
                except Exception as e:
                    print(f"Error processing image message {i}: {str(e)}")

    def _analyze_audio_quality(self, audio_data):
        """Analyze audio quality and return a grade."""
        if not audio_data:
            return "F - No audio data found"

        audio_array = np.array(audio_data)

        peak_amplitude = np.max(np.abs(audio_array))
        rms = np.sqrt(np.mean(np.square(audio_array)))
        crest_factor = peak_amplitude / rms if rms > 0 else 0

        grade = "A"
        reasons = []

        if peak_amplitude > 30000:  # Close to int16 max (32767)
            grade = "B"
            reasons.append("Some audio clipping detected")

        if rms < 1000:
            grade = "C"
            reasons.append("Low audio levels")

        if crest_factor < 3:
            grade = "C"
            reasons.append("Limited dynamic range")

        if rms < 100:
            grade = "F"
            reasons.append("Extremely low or no audio signal")

        return (
            f"{grade} - {'; '.join(reasons)}"
            if reasons
            else f"{grade} - Good audio quality"
        )

    def extract_audio(self):
        """Extract audio from the first audio topic found using librosa."""
        audio_topic = self._get_topic_by_type("audio")
        if not audio_topic:
            raise Exception("No audio topics found in bag file")

        audio_data = []
        timestamp = None

        with rosbag.Bag(self.bag_path, "r") as bag:
            total_msgs = bag.get_message_count(audio_topic)

            for data in tqdm(bag.read_messages(topics=[audio_topic]), total=total_msgs):
                try:
                    if type(data) != BagMessage:
                        continue

                    float_samples = np.array(data.message.data, dtype=np.float32)
                    float_samples = float_samples.reshape(-1, 2)

                    audio_data.append(float_samples)

                    if timestamp is None:
                        timestamp = data.timestamp.to_nsec()

                except Exception as e:
                    raise Exception(f"Error processing audio message: {str(e)}")

        if not audio_data:
            raise Exception("No audio data found in the bag file")

        audio_array = np.concatenate(audio_data, axis=0)
        os.makedirs(self.audio_dir, exist_ok=True)
        temp_path = os.path.join(self.audio_dir, "temp.wav")
        sf.write(temp_path, audio_array, 44100)  # Use temporary sample rate

        _, sample_rate = librosa.load(temp_path, sr=None)
        os.remove(temp_path)  # Clean up temp file

        duration = len(audio_array) / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")

        audio_data = preprocess_audio(audio_data, sample_rate)

        if audio_data and timestamp:
            audio_array = np.concatenate(audio_data, axis=0)

            bag_name = os.path.splitext(os.path.basename(self.bag_path))[0]
            wav_filename = os.path.join(self.audio_dir, f"{bag_name}.wav")
            sf.write(wav_filename, audio_array, sample_rate, "FLOAT")

            print(f"Saved audio file: {wav_filename}")

            plt.figure(figsize=(15, 5))
            time = np.arange(len(audio_array)) / sample_rate
            combined_audio = np.mean(audio_array, axis=1)
            plt.plot(time, combined_audio, color="purple")
            plt.title("Audio Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.savefig(os.path.join(self.audio_dir, f"{bag_name}_waveform.png"))
            plt.close()

            plt.figure(figsize=(15, 5))
            D = librosa.stft(combined_audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            plt.imshow(S_db, aspect="auto", origin="lower")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Audio Spectrogram")
            plt.xlabel("Time Frame")
            plt.ylabel("Frequency Bin")
            plt.tight_layout()
            plt.savefig(os.path.join(self.audio_dir, f"{bag_name}_spectrogram.png"))
            plt.close()

    def draw_3d_vector(self, frame, force, origin=(500, 70), scale=15):
        """Draw 3D force vector visualization"""
        proj_matrix = (
            np.array(
                [
                    [1, -0.3, 0.3],  # x-axis
                    [0, -0.5, -1],  # y-axis
                    [-0.5, -0.8, 0.3],  # z-axis
                ]
            )
            * scale
        )

        red = (0, 0, 255)  # x-axis
        green = (0, 255, 0)  # y-axis
        blue = (255, 0, 0)  # z-axis
        yellow = (0, 255, 255)  # force vector

        axis_length = 20
        for i, color in enumerate([(red, "X"), (green, "Y"), (blue, "Z")]):
            axis = np.zeros(3)
            axis[i] = axis_length
            projected = np.dot(proj_matrix, axis)
            end_point = (int(origin[0] + projected[0]), int(origin[1] + projected[1]))
            cv2.line(frame, origin, end_point, color[0], 2)
            label_offset = 5
            label_pos = (end_point[0] + label_offset, end_point[1] + label_offset)
            cv2.putText(
                frame, color[1], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[0], 1
            )

        force_vec = np.array([force.x, force.y, force.z])
        magnitude = np.linalg.norm(force_vec)
        if magnitude > 0:
            force_vec = force_vec / magnitude * min(magnitude, axis_length)

        projected_force = np.dot(proj_matrix, force_vec)
        force_end = (
            int(origin[0] + projected_force[0]),
            int(origin[1] + projected_force[1]),
        )

        cv2.line(frame, origin, force_end, yellow, 2)

        cv2.putText(
            frame,
            f"{magnitude:.1f}N",
            (origin[0] - 30, origin[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            yellow,
            1,
        )

        return frame

    def process_audio_segments(self):
        """Process audio segments before video creation"""
        try:
            audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".wav")]
            if not audio_files:
                print("No audio file found for segmentation")
                return

            audio_path = os.path.join(self.audio_dir, audio_files[0])
            print(f"Processing audio segments from: {audio_path}")

            self.audio_data, self.audio_sr = librosa.load(
                audio_path, sr=None, mono=False
            )  # Use original sample rate, keep stereo

            if len(self.audio_data.shape) > 1 and self.audio_data.shape[0] == 2:
                self.audio_data_mono = np.mean(self.audio_data, axis=0)
            else:
                self.audio_data_mono = self.audio_data

            cfg = OmegaConf.load("scripts/config/config.yaml")

            segments_and_force = segment_audio(
                audio_file=audio_path,
                window_length_seconds=cfg.data.window_length_seconds,
                window_stride_seconds=cfg.data.window_stride_seconds,
                force_dir=os.path.join(self.output_dir, "force"),
            )

            if isinstance(segments_and_force, tuple):
                self.audio_segments, self.force_data = segments_and_force
                print(
                    f"Received segments and force data. Segment count: {len(self.audio_segments)}"
                )

                contact_segments = sum(
                    1 for _, _, is_contact, _, _ in self.audio_segments if is_contact
                )
                non_contact_segments = len(self.audio_segments) - contact_segments

                print(f"Contact segments: {contact_segments}")
                print(f"Non-contact segments: {non_contact_segments}")

                contact_duration = sum(
                    (end - start) / self.audio_sr
                    for start, end, is_contact, _, _ in self.audio_segments
                    if is_contact
                )
                non_contact_duration = sum(
                    (end - start) / self.audio_sr
                    for start, end, is_contact, _, _ in self.audio_segments
                    if not is_contact
                )

                print(f"Contact duration: {contact_duration:.2f}s")
                print(f"Non-contact duration: {non_contact_duration:.2f}s")

                for i, (start, end, is_contact, windows, _) in enumerate(
                    self.audio_segments
                ):
                    segment_type = "Contact" if is_contact else "Non-contact"
                    print(
                        f"Segment {i + 1}: {segment_type}, start={start / self.audio_sr:.2f}s, end={end / self.audio_sr:.2f}s, duration={(end - start) / self.audio_sr:.2f}s"
                    )
            else:
                self.audio_segments = segments_and_force
                self.force_data = None
                print(
                    f"Received only segments. Segment count: {len(self.audio_segments)}"
                )

                contact_segments = sum(
                    1 for _, _, is_contact, _ in self.audio_segments if is_contact
                )
                non_contact_segments = len(self.audio_segments) - contact_segments

                print(f"Contact segments: {contact_segments}")
                print(f"Non-contact segments: {non_contact_segments}")

                contact_duration = sum(
                    (end - start) / self.audio_sr
                    for start, end, is_contact, _ in self.audio_segments
                    if is_contact
                )
                non_contact_duration = sum(
                    (end - start) / self.audio_sr
                    for start, end, is_contact, _ in self.audio_segments
                    if not is_contact
                )

                print(f"Contact duration: {contact_duration:.2f}s")
                print(f"Non-contact duration: {non_contact_duration:.2f}s")

                for i, (start, end, is_contact, windows) in enumerate(
                    self.audio_segments
                ):
                    segment_type = "Contact" if is_contact else "Non-contact"
                    print(
                        f"Segment {i + 1}: {segment_type}, start={start / self.audio_sr:.2f}s, end={end / self.audio_sr:.2f}s, duration={(end - start) / self.audio_sr:.2f}s"
                    )

            if self.audio_segments:
                print(f"Found {len(self.audio_segments)} audio segments")
            else:
                print("No audio segments found")

        except ImportError as e:
            print(f"Error importing segment_audio: {str(e)}")
            self.audio_segments = None
        except Exception as e:
            print(f"Error processing audio segments: {str(e)}")
            self.audio_segments = None

    def _calculate_fps(self):
        """Calculate actual FPS from the bag file."""
        with rosbag.Bag(self.bag_path, "r") as bag:
            frame_times = []

            for topic, msg, t in bag.read_messages(topics=["/camera1/color/image_raw"]):
                frame_times.append(t.to_sec())

            if frame_times:
                frame_intervals = np.diff(frame_times)
                self.fps = 1 / np.mean(frame_intervals)
                print(f"Calculated FPS from bag: {self.fps:.1f}")
            else:
                print("Warning: No frames found to calculate FPS, using default")
                self.fps = 30

    def create_video(self, fps=None):
        """Create a video from extracted images and audio."""
        if fps is None:
            fps = self.fps

        print(f"Creating video with FPS: {fps:.1f}")

        self.process_audio_segments()

        print("Creating video...")

        bag_name = os.path.splitext(os.path.basename(self.bag_path))[0]
        temp_dir = tempfile.mkdtemp()

        try:
            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            final_video_path = os.path.join(self.output_dir, f"{bag_name}_video.mp4")

            with rosbag.Bag(self.bag_path, "r") as bag:
                print("Collecting messages...")
                image_msgs = []
                camera2_msgs = {}
                audio_msgs = {}

                for topic, msg, t in bag.read_messages():
                    timestamp = t.to_nsec()

                    if topic == "/camera1/color/image_raw":
                        try:
                            cv_img = self.bridge.imgmsg_to_cv2(
                                msg, desired_encoding="bgr8"
                            )
                            image_msgs.append(
                                {
                                    "timestamp": timestamp,
                                    "image": cv_img,
                                    "header_stamp": msg.header.stamp.to_nsec(),
                                }
                            )
                        except Exception as e:
                            print(f"Error processing image: {str(e)}")

                    elif topic == "/camera2/color/image_raw":
                        try:
                            cv_img = self.bridge.imgmsg_to_cv2(
                                msg, desired_encoding="bgr8"
                            )
                            camera2_msgs[msg.header.stamp.to_nsec()] = cv_img
                        except Exception as e:
                            print(f"Error processing camera2: {str(e)}")

                    elif topic == "/audio":  # Use the correct audio data topic
                        try:
                            audio_msgs[timestamp] = (
                                msg.data
                            )  # AudioData message has a 'data' field
                        except Exception as e:
                            print(f"Error processing audio: {str(e)}")

                if not image_msgs:
                    print("No valid images found to create video")
                    return

                image_msgs.sort(key=lambda x: x["timestamp"])
                audio_timestamps = sorted(audio_msgs.keys())

                print(f"Collected messages:")
                print(f"Images: {len(image_msgs)}")
                print(f"Camera2: {len(camera2_msgs)}")
                print(f"Audio: {len(audio_msgs)}")

                start_time = image_msgs[0]["timestamp"] / 1e9  # Convert to seconds
                end_time = image_msgs[-1]["timestamp"] / 1e9
                duration = end_time - start_time
                total_frames = int(duration * fps)

                print(f"Video duration: {duration:.2f} seconds")
                print(f"Total frames to write: {total_frames}")

                height, width, layers = image_msgs[0]["image"].shape

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

                print("Writing frames to video...")
                max_time_diff = int(0.1 * 1e9)  # 100ms in nanoseconds

                target_times = np.linspace(start_time, end_time, total_frames)

                for frame_time in tqdm(target_times):
                    frame_time_ns = int(frame_time * 1e9)  # Convert to nanoseconds

                    closest_msg = min(
                        image_msgs, key=lambda x: abs(x["timestamp"] - frame_time_ns)
                    )
                    frame = closest_msg["image"]
                    frame_timestamp = closest_msg["timestamp"]

                    camera2_frame = None
                    if camera2_msgs:
                        closest_camera2 = min(
                            camera2_msgs.keys(), key=lambda x: abs(x - frame_timestamp)
                        )
                        if abs(closest_camera2 - frame_timestamp) < max_time_diff:
                            camera2_frame = camera2_msgs[closest_camera2]

                    audio_data = None
                    if audio_timestamps:
                        closest_audio = min(
                            audio_timestamps, key=lambda x: abs(x - frame_timestamp)
                        )
                        if abs(closest_audio - frame_timestamp) < max_time_diff:
                            audio_data = audio_msgs[closest_audio]

                video_writer.release()

                if not os.path.exists(temp_video):
                    print(f"Error: Temporary video file {temp_video} was not created")
                    return

                if audio_msgs:
                    print("Adding audio to video...")
                    self.extract_audio()

                    audio_files = [
                        f for f in os.listdir(self.audio_dir) if f.endswith(".wav")
                    ]
                    if audio_files:
                        audio_path = os.path.join(self.audio_dir, audio_files[0])

                        video_clip = VideoFileClip(temp_video)
                        audio_clip = AudioFileClip(audio_path)

                        if audio_clip.duration > video_clip.duration:
                            audio_clip = audio_clip.subclip(0, video_clip.duration)

                        final_clip = video_clip.set_audio(audio_clip)
                        final_clip.write_videofile(
                            final_video_path, codec="libx264", audio_codec="aac"
                        )

                        video_clip.close()
                        audio_clip.close()
                    else:
                        print("No audio file found to add to video")
                        shutil.copy2(temp_video, final_video_path)
                else:
                    print("No audio messages found in bag file")
                    shutil.copy2(temp_video, final_video_path)

                print(f"Video created successfully at: {final_video_path}")
        finally:
            shutil.rmtree(temp_dir)


def preprocess_audio(audio_data, sample_rate):
    """Preprocess audio data using noise reduction."""
    cfg = OmegaConf.load("scripts/config/config.yaml")
    noise_file = cfg.preprocessing.noise_file if hasattr(cfg, "preprocessing") else None

    if not noise_file or not os.path.exists(noise_file):
        raise Exception(
            f"Noise file not found. Please ensure the noise file exists at the path specified in config.yaml"
        )

    try:
        audio_array = np.concatenate(audio_data, axis=0)

        y = np.mean(audio_array, axis=1)

        noise_y, noise_sr = librosa.load(noise_file)

        if sample_rate != noise_sr:
            noise_y = librosa.resample(noise_y, orig_sr=noise_sr, target_sr=sample_rate)

        reduced_noise = nr.reduce_noise(
            y=y, sr=sample_rate, y_noise=noise_y, prop_decrease=1.0, stationary=False
        )

        cleaned_stereo = np.stack([reduced_noise, reduced_noise], axis=1)

        return [cleaned_stereo]

    except Exception as e:
        raise Exception(f"Error during audio preprocessing: {str(e)}")
