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
from moviepy.editor import (
    ImageSequenceClip,
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
)
from omegaconf import OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from extract import RosBagExtractor
from segment_audio import segment_audio, get_audio_envelope


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout


def find_bags_recursive(directory):
    """Find all .bag files recursively in the given directory."""
    return glob.glob(os.path.join(directory, "**/*.bag"), recursive=True)


default_config = {
    "window_length_seconds": 1.0,
    "window_stride_seconds": 0.1,
    "non_contact_threshold_factor": 0.5,
    "enable_squeezing": True,
    "dynamic_threshold_offset": 0.15,
    "squeeze_duration": 0.3,
    "min_duration": 0.25,  # Add minimum segment duration
}


def get_config(cfg, key, default):
    try:
        if hasattr(cfg, "data") and hasattr(cfg.data, key):
            return getattr(cfg.data, key)
        elif hasattr(cfg, "segmentation") and hasattr(cfg.segmentation, key):
            return getattr(cfg.segmentation, key)
        else:
            return default
    except (AttributeError, KeyError):
        return default


def process_bag(
    bag_path, output_dir=None, fps=None, cfg=None, is_robot=True, no_visuals=False
):
    """Process a single bag file."""
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]

    if output_dir:
        bag_output_dir = os.path.join(output_dir, bag_name)
    else:
        bag_output_dir = os.path.join("output", bag_name)

    print(f"Processing {os.path.basename(bag_path)}...", end=" ", flush=True)

    extractor = RosBagExtractor(bag_path, bag_output_dir)

    try:
        with suppress_stdout():
            extractor.extract_images()
    except Exception as e:
        print(f"\nSkipping {os.path.basename(bag_path)}: No image data")
        return

    try:
        with suppress_stdout():
            extractor.extract_audio()
    except Exception as e:
        print(f"\nSkipping {os.path.basename(bag_path)}: No audio data")
        return

    calculated_fps = getattr(extractor, "fps", None)

    if calculated_fps is not None and calculated_fps > 0:
        fps = calculated_fps
    else:
        fps = 30.0

    print(f"(Using FPS: {fps:.1f}) ", end="", flush=True)

    print("Creating video...", end=" ", flush=True)

    image_dir = os.path.join(bag_output_dir, "images")
    audio_file = os.path.join(bag_output_dir, "audio", f"{bag_name}.wav")
    output_video = os.path.join(bag_output_dir, f"{bag_name}_video.mp4")

    if not os.path.exists(image_dir) or not os.path.exists(audio_file):
        print(f"\nError: Missing images or audio for {bag_name}")
        return

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_files:
        print(f"\nError: No image files found in {image_dir}")
        return

    try:
        if cfg is None:
            config_path = os.path.join(script_dir, "config", "config.yaml")
            if os.path.exists(config_path):
                cfg = OmegaConf.load(config_path)
            else:
                cfg = OmegaConf.create({"data": {}, "segmentation": {}})

        sample_img = cv2.imread(image_files[0])
        img_height, img_width = sample_img.shape[:2]

        audio_data, sr = librosa.load(audio_file, sr=None)

        window_length_seconds = get_config(
            cfg, "window_length_seconds", default_config["window_length_seconds"]
        )
        window_stride_seconds = get_config(
            cfg, "window_stride_seconds", default_config["window_stride_seconds"]
        )

        stderr_fd = sys.stderr.fileno()
        with os.fdopen(os.dup(stderr_fd), "wb") as copied:
            sys.stderr.flush()
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), stderr_fd)
                segments = segment_audio(
                    audio_file,
                    window_length_seconds=get_config(
                        cfg,
                        "window_length_seconds",
                        default_config["window_length_seconds"],
                    ),
                    window_stride_seconds=get_config(
                        cfg,
                        "window_stride_seconds",
                        default_config["window_stride_seconds"],
                    ),
                    non_contact_threshold_factor=get_config(
                        cfg,
                        "non_contact_threshold_factor",
                        default_config["non_contact_threshold_factor"],
                    ),
                    enable_squeezing=get_config(
                        cfg, "enable_squeezing", default_config["enable_squeezing"]
                    ),
                    squeeze_factor_seconds=get_config(
                        cfg, "squeeze_duration", default_config["squeeze_duration"]
                    ),
                    min_segment_duration=get_config(
                        cfg, "min_duration", default_config["min_duration"]
                    ),
                    dynamic_threshold_offset=get_config(
                        cfg,
                        "dynamic_threshold_offset",
                        default_config["dynamic_threshold_offset"],
                    ),
                    cfg=cfg,
                    is_robot=is_robot,
                )
            finally:
                sys.stderr.flush()
                os.dup2(copied.fileno(), stderr_fd)

        envelope = get_audio_envelope(audio_data, frame_length=512, hop_length=128)
        window_size = int(0.02 * sr)
        envelope_smoothed = np.convolve(
            envelope, np.ones(window_size) / window_size, mode="same"
        )

        noise_floor = np.percentile(envelope_smoothed, 10)
        signal_peak = np.percentile(envelope_smoothed, 90)
        dynamic_threshold = noise_floor + (signal_peak - noise_floor) * get_config(
            cfg, "dynamic_threshold_offset", default_config["dynamic_threshold_offset"]
        )
        non_contact_threshold = dynamic_threshold * get_config(
            cfg,
            "non_contact_threshold_factor",
            default_config["non_contact_threshold_factor"],
        )

        duration = len(image_files) / fps

        video_clip = ImageSequenceClip(image_files, fps=fps)

        video_clip = video_clip.set_duration(duration)

        audio_clip = AudioFileClip(audio_file)

        if audio_clip.duration > duration:
            audio_clip = audio_clip.subclip(0, duration)

        video_clip = video_clip.set_audio(audio_clip)

        if no_visuals:
            final_clip = video_clip
        else:
            waveform_height = 100  # Height of the waveform visualization
            total_frames = len(image_files)

            def make_waveform_frame(t):
                frame = np.ones((waveform_height, img_width, 3), dtype=np.uint8) * 255

                frame_idx = int(t * fps)

                max_amplitude = (
                    np.max(np.abs(audio_data)) if len(audio_data) > 0 else 1.0
                )

                for i in range(img_width):
                    audio_pos = int((i / img_width) * len(audio_data))
                    if audio_pos < len(audio_data):
                        normalized_amp = (
                            audio_data[audio_pos] / max_amplitude
                        )  # Now in range [-1, 1]
                        amplitude = int(
                            (0.5 - 0.5 * normalized_amp) * waveform_height
                        )  # Map to [0, waveform_height]

                        cv2.line(
                            frame,
                            (i, waveform_height // 2),  # Center point
                            (i, amplitude),  # Amplitude point
                            (0, 0, 255),  # Red color
                            1,
                        )  # Line thickness

                for i in range(img_width):
                    env_pos = int((i / img_width) * len(envelope_smoothed))
                    if env_pos < len(envelope_smoothed):
                        env_val = envelope_smoothed[env_pos] / max_amplitude
                        env_y = int((1.0 - env_val) * waveform_height)
                        env_y = max(0, min(waveform_height - 1, env_y))
                        cv2.circle(frame, (i, env_y), 1, (0, 128, 0), -1)

                threshold_y = int(
                    (1.0 - dynamic_threshold / max_amplitude) * waveform_height
                )
                threshold_y = max(0, min(waveform_height - 1, threshold_y))
                cv2.line(
                    frame, (0, threshold_y), (img_width, threshold_y), (0, 255, 0), 1
                )

                non_contact_y = int(
                    (1.0 - non_contact_threshold / max_amplitude) * waveform_height
                )
                non_contact_y = max(0, min(waveform_height - 1, non_contact_y))
                cv2.line(
                    frame,
                    (0, non_contact_y),
                    (img_width, non_contact_y),
                    (255, 0, 255),
                    1,
                )

                current_time = t
                current_sample = int(current_time * sr)

                for i, (start, end, is_contact, _) in enumerate(segments):
                    start_x = int((start / len(audio_data)) * img_width)
                    end_x = int((end / len(audio_data)) * img_width)

                    color = (
                        (255, 0, 0) if is_contact else (0, 0, 255)
                    )  # Red for contact, Blue for non-contact

                    cv2.line(frame, (start_x, 0), (start_x, waveform_height), color, 2)
                    cv2.line(frame, (end_x, 0), (end_x, waveform_height), color, 2)

                    label = f"S{i + 1}"
                    cv2.putText(
                        frame,
                        label,
                        (start_x + 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

                    if start <= current_sample <= end:
                        overlay = frame.copy()
                        cv2.rectangle(
                            overlay, (start_x, 0), (end_x, waveform_height), color, -1
                        )  # Filled rectangle
                        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                        cv2.putText(
                            frame,
                            "CURRENT",
                            (start_x + 5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )

                progress = t / duration if duration > 0 else 0
                progress_x = int(progress * img_width)
                cv2.line(
                    frame,
                    (progress_x, 0),
                    (progress_x, waveform_height),
                    (0, 255, 0),  # Green color
                    2,
                )  # Line thickness

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (img_width - 80, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )

                return frame

            waveform_clip = ColorClip(
                size=(img_width, waveform_height), color=(0, 0, 0), duration=duration
            )
            waveform_clip = waveform_clip.set_make_frame(make_waveform_frame)

            final_clip = CompositeVideoClip(
                [video_clip, waveform_clip.set_position(("center", img_height))],
                size=(img_width, img_height + waveform_height),
            )

        with suppress_stdout():
            final_clip.write_videofile(
                output_video,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(bag_output_dir, "temp-audio.m4a"),
                remove_temp=True,
                verbose=False,
                logger=None,
                fps=fps,  # Explicitly set the FPS for the output video
            )
        print("Done")

    except Exception as e:
        print(f"\nError creating video: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Extract data from ROS bags")
    parser.add_argument(
        "path", help="Path to ROS bag file or directory containing ROS bags"
    )
    parser.add_argument(
        "-o", "--output", help="Output directory for extracted files", default="output"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively search for bag files in subdirectories",
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Specify if processing robot data (default: true)",
        default=True,
    )
    parser.add_argument(
        "--probe", action="store_true", help="Specify if processing probe data"
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Generate video without waveform visualizations",
    )

    args = parser.parse_args()

    is_robot = not args.probe

    config_path = os.path.join(script_dir, "config", "config.yaml")
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({"data": {}, "segmentation": {}})

    if os.path.isfile(args.path):
        process_bag(
            args.path,
            args.output,
            cfg=cfg,
            is_robot=is_robot,
            no_visuals=args.no_visuals,
        )
    elif os.path.isdir(args.path):
        if args.recursive:
            bag_files = find_bags_recursive(args.path)
        else:
            bag_files = [
                os.path.join(args.path, f)
                for f in os.listdir(args.path)
                if f.endswith(".bag")
            ]

        if not bag_files:
            print(f"No bag files found in {args.path}")
            return

        for bag_path in bag_files:
            process_bag(
                bag_path,
                args.output,
                cfg=cfg,
                is_robot=is_robot,
                no_visuals=args.no_visuals,
            )
    else:
        print(f"Error: Path not found: {args.path}")


if __name__ == "__main__":
    main()

