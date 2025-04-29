import os
import sys
import argparse
import cv2
import glob
import numpy as np
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm
from contextlib import contextmanager
from moviepy.editor import ImageSequenceClip, AudioFileClip
import soundfile as sf


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout


def extract_images_from_bag(bag_path, output_dir):
    """
    Extract images from a bag file.

    Args:
        bag_path (str): Path to the bag file
        output_dir (str): Directory to save the images

    Returns:
        float: The calculated FPS
    """
    print("Extracting images...", end=" ", flush=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    existing_images = glob.glob(os.path.join(image_dir, "*.jpg"))
    if existing_images:
        print(
            f"Removing {len(existing_images)} existing images...", end=" ", flush=True
        )
        for img in existing_images:
            os.remove(img)

    bridge = CvBridge()
    fps = 30.0  # Default FPS

    image_topic = None
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, info in bag.get_type_and_topic_info().topics.items():
            if "image" in info.msg_type.lower():
                image_topic = topic
                break

    if not image_topic:
        print("No image topics found in bag file")
        return fps

    print(f"Found image topic: {image_topic}")

    with rosbag.Bag(bag_path, "r") as bag:
        frame_times = []
        for _, _, t in bag.read_messages(topics=[image_topic]):
            frame_times.append(t.to_sec())

        if len(frame_times) > 1:
            frame_intervals = np.diff(frame_times)
            fps = 1.0 / np.mean(frame_intervals)
            print(f"Calculated FPS: {fps:.1f}")

        total_msgs = bag.get_message_count(image_topic)
        processed_timestamps = set()  # Track processed timestamps to avoid duplicates
        frame_counter = 0

        for i, data in enumerate(
            tqdm(bag.read_messages(topics=[image_topic]), total=total_msgs)
        ):
            try:
                timestamp_ns = data.timestamp.to_nsec()
                if timestamp_ns in processed_timestamps:
                    continue

                processed_timestamps.add(timestamp_ns)

                cv_img = bridge.imgmsg_to_cv2(data.message, desired_encoding="bgr8")
                timestamp = str(timestamp_ns)
                filename = os.path.join(image_dir, f"frame_{frame_counter:06d}.jpg")
                cv2.imwrite(filename, cv_img)
                frame_counter += 1
            except Exception as e:
                print(f"Error processing image message {i}: {str(e)}")

    print(f"Done. Extracted {frame_counter} unique frames.")
    return fps


def extract_audio_from_bag(bag_path, output_dir):
    """
    Extract audio from a bag file.

    Args:
        bag_path (str): Path to the bag file
        output_dir (str): Directory to save the audio

    Returns:
        str: Path to the saved audio file
    """
    print("Extracting audio...", end=" ", flush=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    audio_file = os.path.join(audio_dir, f"{bag_name}.wav")

    audio_topic = None
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, info in bag.get_type_and_topic_info().topics.items():
            if "audio" in info.msg_type.lower():
                audio_topic = topic
                break

    if not audio_topic:
        print("No audio topics found in bag file")
        return None

    print(f"Found audio topic: {audio_topic}")

    audio_data = []
    with rosbag.Bag(bag_path, "r") as bag:
        total_msgs = bag.get_message_count(audio_topic)
        for data in tqdm(bag.read_messages(topics=[audio_topic]), total=total_msgs):
            try:
                float_samples = np.array(data.message.data, dtype=np.float32)
                float_samples = float_samples.reshape(-1, 2)
                audio_data.append(float_samples)
            except Exception as e:
                print(f"Error processing audio message: {str(e)}")

    if not audio_data:
        print("No audio data found in the bag file")
        return None

    audio_array = np.concatenate(audio_data, axis=0)

    sample_rate = 44100
    sf.write(audio_file, audio_array, sample_rate, "FLOAT")

    print("Done")
    return audio_file


def create_video_from_bag(bag_path, output_dir=None, fps=None):
    """
    Create a simple video from a bag file without any preprocessing or normalization.

    Args:
        bag_path (str): Path to the bag file
        output_dir (str): Directory to save the output (default: None, will use 'output/bag_name')
        fps (float): Frames per second for the video (default: None, will calculate from bag)

    Returns:
        str: Path to the created video file
    """
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]

    if output_dir:
        bag_output_dir = os.path.join(output_dir, bag_name)
    else:
        bag_output_dir = os.path.join("output", bag_name)

    os.makedirs(bag_output_dir, exist_ok=True)

    print(f"Processing {os.path.basename(bag_path)}...")

    calculated_fps = extract_images_from_bag(bag_path, bag_output_dir)

    audio_file = extract_audio_from_bag(bag_path, bag_output_dir)

    if fps is not None:
        video_fps = fps
    elif calculated_fps is not None and calculated_fps > 0:
        video_fps = calculated_fps
    else:
        video_fps = 30.0

    print(f"Using FPS: {video_fps:.1f}")

    image_dir = os.path.join(bag_output_dir, "images")
    output_video = os.path.join(bag_output_dir, f"{bag_name}_video.mp4")

    if not os.path.exists(image_dir):
        print(f"Error: Missing images for {bag_name}")
        return None

    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )

    if not image_files:
        print(f"Error: No image files found in {image_dir}")
        return None

    print(f"Creating video from {len(image_files)} images...", end=" ", flush=True)

    try:
        duration = len(image_files) / video_fps

        video_clip = ImageSequenceClip(image_files, fps=video_fps)

        video_clip = video_clip.set_duration(duration)

        if audio_file and os.path.exists(audio_file):
            audio_clip = AudioFileClip(audio_file)

            if audio_clip.duration > duration:
                audio_clip = audio_clip.subclip(0, duration)

            video_clip = video_clip.set_audio(audio_clip)
        elif audio_file is None:
            print("Warning: No audio data found, creating video without audio")
        else:
            print(
                f"Warning: Audio file {audio_file} not found, creating video without audio"
            )

        with suppress_stdout():
            if os.path.exists(output_video):
                os.remove(output_video)

            video_clip.write_videofile(
                output_video,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(bag_output_dir, "temp-audio.m4a"),
                remove_temp=True,
                verbose=False,
                logger=None,
                fps=video_fps,
            )
        print("Done")
        print(f"Video saved to: {output_video}")
        return output_video

    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return None


def find_bags_recursive(directory, recursive=False):
    """
    Find all .bag files in the given directory.

    Args:
        directory (str): Directory to search
        recursive (bool): Whether to search recursively in subdirectories

    Returns:
        list: List of paths to bag files
    """
    if recursive:
        return glob.glob(os.path.join(directory, "**/*.bag"), recursive=True)
    else:
        return glob.glob(os.path.join(directory, "*.bag"))


def main():
    parser = argparse.ArgumentParser(
        description="Create video from ROS bag file(s) without preprocessing"
    )
    parser.add_argument(
        "path", help="Path to ROS bag file or directory containing bag files"
    )
    parser.add_argument(
        "--output", "-o", help="Output directory (default: output/bag_name)"
    )
    parser.add_argument("--fps", type=float, help="Override FPS for the output video")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search recursively for bag files in subdirectories",
    )

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path not found: {args.path}")
        return 1

    if os.path.isfile(args.path):
        if not args.path.endswith(".bag"):
            print(f"Error: File is not a bag file: {args.path}")
            return 1

        result = create_video_from_bag(args.path, args.output, args.fps)

        if result:
            print("Video creation successful!")
            return 0
        else:
            print("Video creation failed.")
            return 1

    elif os.path.isdir(args.path):
        bag_files = find_bags_recursive(args.path, args.recursive)

        if not bag_files:
            print(f"No bag files found in {args.path}")
            return 1

        print(f"Found {len(bag_files)} bag files")

        success_count = 0
        fail_count = 0

        for i, bag_path in enumerate(bag_files):
            print(
                f"\nProcessing bag {i + 1}/{len(bag_files)}: {os.path.basename(bag_path)}"
            )

            try:
                result = create_video_from_bag(bag_path, args.output, args.fps)

                if result:
                    print(
                        f"Successfully created video from {os.path.basename(bag_path)}"
                    )
                    success_count += 1
                else:
                    print(f"Failed to create video from {os.path.basename(bag_path)}")
                    fail_count += 1
            except Exception as e:
                print(f"Error processing bag file {bag_path}: {str(e)}")
                fail_count += 1

        print("\nProcessing summary:")
        print(f"Total bag files: {len(bag_files)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {fail_count}")

        return 0 if fail_count == 0 else 1

    else:
        print(f"Error: Path is neither a file nor a directory: {args.path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
