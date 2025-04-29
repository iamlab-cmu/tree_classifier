import rosbag
import librosa
import numpy as np
import os
import shutil
from segment_audio import segment_audio, get_audio_envelope
import soundfile as sf
import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import tempfile
from cv_bridge import CvBridge
import cv2
import glob
import noisereduce as nr
import sys
import json
import matplotlib.pyplot as plt


def extract_audio_from_bag(bag_path, save_to_file=None, cfg=None):
    """
    Extract audio data from a bag file and save it to a temporary WAV file.

    Args:
        bag_path (str): Path to the ROS bag file
        save_to_file (str): Optional path to save the extracted audio to
        cfg (DictConfig): Configuration with preprocessing options

    Returns:
        str: Path to the temporary WAV file, or None if no audio found
    """
    try:
        with rosbag.Bag(bag_path, "r") as bag:
            topics_info = bag.get_type_and_topic_info()
            topics = topics_info[1].keys()

            audio_topic = "/audio" if "/audio" in topics else None

            if not audio_topic:
                for topic, info in topics_info[1].items():
                    if "AudioData" in info.msg_type:
                        audio_topic = topic
                        break

            if not audio_topic:
                for topic in topics:
                    if "audio" in topic.lower() and "info" not in topic.lower():
                        audio_topic = topic
                        break

            if not audio_topic:
                print(f"No audio topic found in {bag_path}")
                return None

            print(
                f"Selected audio topic: {audio_topic} with message type: {topics_info[1][audio_topic].msg_type}"
            )

            audio_data = []
            audio_msg_count = 0
            sample_rate = 44100  # Default sample rate

            for _, msg, _ in bag.read_messages(topics=[audio_topic]):
                audio_msg_count += 1

                if audio_msg_count == 1:
                    print(f"Audio message type: {type(msg).__name__}")
                    relevant_attrs = [
                        attr
                        for attr in dir(msg)
                        if not attr.startswith("_")
                        and attr
                        not in [
                            "serialize",
                            "deserialize",
                            "serialize_numpy",
                            "deserialize_numpy",
                        ]
                    ]
                    print(f"Relevant attributes: {relevant_attrs}")
                else:
                    try:
                        if hasattr(msg, "data"):
                            if isinstance(msg.data, (list, tuple)):
                                audio_data.append(np.array(msg.data, dtype=np.float32))
                            elif isinstance(msg.data, bytes) or isinstance(
                                msg.data, bytearray
                            ):
                                if hasattr(msg, "format") and msg.format == "S16LE":
                                    data_array = (
                                        np.frombuffer(msg.data, dtype=np.int16).astype(
                                            np.float32
                                        )
                                        / 32768.0
                                    )
                                else:
                                    data_array = np.frombuffer(
                                        msg.data, dtype=np.float32
                                    )
                                audio_data.append(data_array)
                            else:
                                audio_data.append(np.array(msg.data, dtype=np.float32))
                        elif hasattr(msg, "audio_data"):
                            audio_data.append(
                                np.array(msg.audio_data, dtype=np.float32)
                            )
                    except Exception as e:
                        print(
                            f"Warning: Could not process audio message {audio_msg_count}: {e}"
                        )
                        continue

                    if hasattr(msg, "sample_rate"):
                        sample_rate = msg.sample_rate
                    elif hasattr(msg, "sr"):
                        sample_rate = msg.sr
                    elif hasattr(msg, "rate"):
                        sample_rate = msg.rate
                    elif hasattr(msg, "sampling_rate"):
                        sample_rate = msg.sampling_rate

            if audio_msg_count == 0:
                print(f"No audio messages found on topic {audio_topic}")
                return None

            print(
                f"Collected {sum(len(d) for d in audio_data if isinstance(d, np.ndarray))} audio samples from {audio_msg_count} messages"
            )

            if audio_data:
                try:
                    valid_audio_data = [
                        d
                        for d in audio_data
                        if isinstance(d, np.ndarray) and d.size > 0
                    ]

                    if not valid_audio_data:
                        print("No valid audio data found to concatenate")
                        return None

                    all_audio = np.concatenate(valid_audio_data)

                    if not np.isfinite(all_audio).all():
                        print(
                            "Warning: Audio contains NaN or Inf values, replacing with zeros"
                        )
                        all_audio = np.nan_to_num(all_audio)

                    do_norm = True  # Default to True for backward compatibility
                    if (
                        cfg
                        and hasattr(cfg, "preprocessing")
                        and hasattr(cfg.preprocessing, "do_norm")
                    ):
                        do_norm = cfg.preprocessing.do_norm
                        print(
                            f"Audio normalization is {'enabled' if do_norm else 'disabled'} as per configuration"
                        )
                    else:
                        print(
                            "No do_norm setting found in config, defaulting to enabled"
                        )

                    if do_norm:
                        rms_level = np.sqrt(np.mean(all_audio**2))
                        if rms_level > 0:  # Avoid division by zero
                            target_rms = 0.1  # Target RMS level (adjust as needed)
                            all_audio = all_audio * (target_rms / rms_level)
                            print(
                                f"Applied RMS normalization: original RMS={rms_level:.6f}, target RMS={target_rms:.6f}"
                            )
                        else:
                            print(
                                "Warning: Audio has zero RMS level, skipping normalization"
                            )
                    else:
                        print(
                            "Skipping audio normalization as it's disabled in the configuration"
                        )

                    if save_to_file:
                        temp_file = save_to_file
                    else:
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False
                        ).name

                    try:
                        from scipy.io import wavfile

                        print(
                            f"Saving audio data to {temp_file} with scipy (sample rate: {sample_rate})"
                        )
                        wavfile.write(temp_file, sample_rate, all_audio)

                        if os.path.isfile(temp_file) and os.path.getsize(temp_file) > 0:
                            try:
                                y, sr = librosa.load(temp_file, sr=None)
                                print(
                                    f"Successfully extracted audio: {len(y)} samples at {sr}Hz"
                                )
                                print(
                                    f"Audio statistics: min={np.min(y):.6f}, max={np.max(y):.6f}, mean={np.mean(y):.6f}"
                                )

                                if np.max(np.abs(y)) < 0.01:
                                    print(
                                        "WARNING: Extracted audio has very low amplitude - contact detection may fail"
                                    )
                            except Exception as e:
                                print(f"Error analyzing extracted audio: {e}")
                            return temp_file
                        else:
                            print(f"Error: Failed to save audio file or file is empty")
                            return None

                    except Exception as e:
                        print(f"Error saving with scipy: {str(e)}")

                        try:
                            print(f"Trying with soundfile instead")
                            sf.write(temp_file, all_audio, sample_rate)
                            return temp_file
                        except Exception as e2:
                            print(f"Error saving with soundfile: {str(e2)}")
                            return None

                except Exception as e:
                    print(f"Error processing audio data: {str(e)}")
                    print(
                        f"Audio data shapes: {[d.shape if isinstance(d, np.ndarray) else type(d) for d in audio_data]}"
                    )
                    return None
            else:
                print("No audio data was collected")
                return None

    except Exception as e:
        print(f"Error extracting audio from bag {bag_path}: {str(e)}")
        return None


def find_closest_image(bag, target_time, max_diff_secs=0.1):
    """
    Find image closest to the target timestamp.

    Args:
        bag: Open rosbag.Bag instance
        target_time: Target timestamp in seconds
        max_diff_secs: Maximum time difference allowed (in seconds)

    Returns:
        tuple: (image_msg, actual_timestamp, time_diff) or (None, None, None) if no match
    """
    image_topics = ["/camera1/color/image_raw"]

    best_diff = float("inf")
    closest_msg = None
    closest_time = None

    for topic in image_topics:
        for _, msg, t in bag.read_messages(topics=[topic]):
            time_diff = abs(t.to_sec() - target_time)
            if time_diff < best_diff:
                best_diff = time_diff
                closest_msg = msg
                closest_time = t.to_sec()

    if best_diff <= max_diff_secs and closest_msg is not None:
        return closest_msg, closest_time, best_diff

    return None, None, None


def process_bag(
    bag_path,
    output_dir,
    window_length_seconds=1.0,
    window_stride_seconds=0.1,
    cfg=None,
    is_robot=False,
):
    """Process a single bag file and extract audio-visual segments."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        audio_dir = os.path.join(output_dir, "audio")
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        metadata = []

        if not os.path.exists(bag_path):
            print(f"ERROR: Bag file does not exist: {bag_path}")
            return []

        bag_name = os.path.splitext(os.path.basename(bag_path))[0]

        standard_category = map_to_category(bag_name, cfg)

        if standard_category is None:
            print(f"Skipping {bag_name} as no category could be determined")
            print(
                f"Make sure the bag name contains one of the categories defined in config.yaml"
            )
            return []

        print(f"Mapped {bag_name} to category: {standard_category}")

        temp_audio_path = extract_audio_from_bag(bag_path, cfg=cfg)

        if not temp_audio_path:
            print(f"Failed to extract audio from {bag_path}")
            return []

        data, sr = librosa.load(temp_audio_path, sr=None)
        print(f"Loaded audio file with {len(data)} samples at {sr}Hz")
        print(
            f"Audio stats: min={np.min(data):.6f}, max={np.max(data):.6f}, mean={np.mean(data):.6f}"
        )

        noise_data = None
        noise_sr = None

        enable_denoising = True  # Default to True for backward compatibility
        if (
            cfg
            and hasattr(cfg, "preprocessing")
            and hasattr(cfg.preprocessing, "enable_denoising")
        ):
            enable_denoising = cfg.preprocessing.enable_denoising
            print(
                f"Denoising is {'enabled' if enable_denoising else 'disabled'} as per configuration"
            )
        else:
            print("No enable_denoising setting found in config, defaulting to enabled")

        if not enable_denoising:
            print("Skipping denoising as it's disabled in the configuration")
            segmentation_audio_data = data
        else:
            noise_file = None
            if cfg and hasattr(cfg, "preprocessing"):
                if is_robot and hasattr(cfg.preprocessing, "robot_noise_file"):
                    noise_file = cfg.preprocessing.robot_noise_file
                    print(f"Using robot noise file from config: {noise_file}")
                elif not is_robot and hasattr(cfg.preprocessing, "probe_noise_file"):
                    noise_file = cfg.preprocessing.probe_noise_file
                    print(f"Using probe noise file from config: {noise_file}")
                elif hasattr(cfg.preprocessing, "noise_file"):
                    noise_file = cfg.preprocessing.noise_file
                    print(f"Using generic noise file from config: {noise_file}")

                if noise_file and not os.path.isabs(noise_file):
                    if "hydra" in sys.modules:
                        original_cwd = hydra.utils.get_original_cwd()
                        absolute_noise_file = os.path.join(original_cwd, noise_file)
                        if os.path.exists(absolute_noise_file):
                            noise_file = absolute_noise_file
                            print(f"Found noise file at absolute path: {noise_file}")
                        else:
                            print(
                                f"Warning: Noise file not found at {absolute_noise_file}"
                            )

                            if os.path.exists(noise_file):
                                print(
                                    f"Found noise file in current directory: {noise_file}"
                                )
                            else:
                                print(
                                    f"Warning: Noise file not found in current directory either: {noise_file}"
                                )

                                if is_robot and hasattr(
                                    cfg.preprocessing, "probe_noise_file"
                                ):
                                    fallback_file = cfg.preprocessing.probe_noise_file
                                    if not os.path.isabs(fallback_file):
                                        fallback_file = os.path.join(
                                            original_cwd, fallback_file
                                        )
                                    if os.path.exists(fallback_file):
                                        noise_file = fallback_file
                                        print(
                                            f"Falling back to probe noise file: {noise_file}"
                                        )
                                elif not is_robot and hasattr(
                                    cfg.preprocessing, "robot_noise_file"
                                ):
                                    fallback_file = cfg.preprocessing.robot_noise_file
                                    if not os.path.isabs(fallback_file):
                                        fallback_file = os.path.join(
                                            original_cwd, fallback_file
                                        )
                                    if os.path.exists(fallback_file):
                                        noise_file = fallback_file
                                        print(
                                            f"Falling back to robot noise file: {noise_file}"
                                        )
            else:
                default_file = (
                    "./robot_humming.wav" if is_robot else "./probe_humming.wav"
                )
                noise_file = default_file
                print(f"Using default noise file: {noise_file}")

            use_denoising = False
            if noise_file and os.path.exists(noise_file):
                try:
                    noise_data, noise_sr = librosa.load(noise_file, sr=None)
                    print(f"Successfully loaded noise file: {noise_file}")
                    use_denoising = True
                except Exception as e:
                    print(f"Error loading noise file {noise_file}: {e}")
                    print("Will skip denoising due to noise file loading error")
                    noise_data = None
                    noise_sr = None
            else:
                print(f"Noise file not found or not accessible: {noise_file}")
                print("Will skip denoising")

            if use_denoising and noise_data is not None and noise_sr is not None:
                try:
                    if sr != noise_sr:
                        noise_data = librosa.resample(
                            noise_data, orig_sr=noise_sr, target_sr=sr
                        )

                    print("Applying noise reduction...")
                    denoised_data = nr.reduce_noise(
                        y=data,
                        sr=sr,
                        y_noise=noise_data,
                        prop_decrease=1.0,
                        stationary=False,
                    )

                    segmentation_audio_data = denoised_data
                except Exception as e:
                    print(f"Error during denoising: {e}")
                    segmentation_audio_data = data
            else:
                print(
                    "No valid noise file available for denoising, using original audio"
                )
                segmentation_audio_data = data

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_segmentation_file:
            temp_segmentation_path = temp_segmentation_file.name
            sf.write(temp_segmentation_path, segmentation_audio_data, sr)

        bridge = CvBridge()

        image_msgs = {}
        with rosbag.Bag(bag_path, "r") as bag:
            topics_info = bag.get_type_and_topic_info()

            image_topics = []
            for topic, info in topics_info[1].items():
                if (
                    "sensor_msgs/Image" in info.msg_type
                    and "camera1/color/image_raw" in topic
                ):
                    image_topics.append(topic)

            if not image_topics:
                print(f"No camera1 image topics found in {bag_path}")
                os.unlink(temp_audio_path)
                return []

            print(f"Found image topics: {image_topics}")

            for topic, msg, t in bag.read_messages(topics=image_topics):
                timestamp = t.to_sec()
                if topic not in image_msgs:
                    image_msgs[topic] = []
                image_msgs[topic].append((timestamp, msg))

        default_config = {
            "window_length_seconds": 1.0,
            "window_stride_seconds": 0.1,
            "non_contact_threshold_factor": 0.3,
            "enable_squeezing": True,
            "dynamic_threshold_offset": 0.15,
        }

        try:

            def get_config_for_segment(key):
                try:
                    if hasattr(cfg, "segmentation") and hasattr(cfg.segmentation, key):
                        return getattr(cfg.segmentation, key)
                    elif hasattr(cfg.data, key):
                        return getattr(cfg.data, key)
                    else:
                        return default_config[key]
                except (AttributeError, KeyError):
                    return default_config[key]

            print("\n=== PARAMETERS BEING PASSED TO SEGMENT_AUDIO ===")
            print(f"window_length_seconds: {window_length_seconds}")
            print(f"window_stride_seconds: {window_stride_seconds}")
            print(
                f"non_contact_threshold_factor: {get_config_for_segment('non_contact_threshold_factor')}"
            )
            print(f"enable_squeezing: {get_config_for_segment('enable_squeezing')}")
            print(f"squeeze_factor_seconds: 0.3")
            print(
                f"dynamic_threshold_offset: {get_config_for_segment('dynamic_threshold_offset')}"
            )
            print(f"is_robot: {is_robot}")
            print("================================================\n")

            print("Calling segment_audio function...")
            segments = segment_audio(
                temp_segmentation_path,
                window_length_seconds=window_length_seconds,
                window_stride_seconds=window_stride_seconds,
                non_contact_threshold_factor=get_config_for_segment(
                    "non_contact_threshold_factor"
                ),
                enable_squeezing=get_config_for_segment("enable_squeezing"),
                squeeze_factor_seconds=0.3,
                dynamic_threshold_offset=get_config_for_segment(
                    "dynamic_threshold_offset"
                ),
                cfg=cfg,
                is_robot=is_robot,
            )

            print("\n=== SEGMENTATION RESULTS ===")
            print(f"Number of segments returned: {len(segments)}")
            contact_window_total = 0
            for i, (start, end, is_contact, windows) in enumerate(segments):
                segment_type = "Contact" if is_contact else "Non-contact"
                duration = (end - start) / sr
                print(
                    f"Segment {i + 1} ({segment_type}): {len(windows)} windows, {duration:.2f} seconds"
                )
                if is_contact:
                    contact_window_total += len(windows)
            print(f"Total contact windows found: {contact_window_total}")
            print("===========================\n")

        except Exception as e:
            print(f"Error segmenting audio from {bag_path}: {str(e)}")
            print(
                f"Audio stats: min={np.min(segmentation_audio_data)}, max={np.max(segmentation_audio_data)}, mean={np.mean(segmentation_audio_data)}, shape={segmentation_audio_data.shape}, sr={sr}"
            )
            os.unlink(temp_audio_path)
            os.unlink(temp_segmentation_path)
            return []

        if not check_segments(segments, segmentation_audio_data, sr):
            print(f"Invalid segments returned for {bag_path}")
            os.unlink(temp_audio_path)
            os.unlink(temp_segmentation_path)
            return []

        if not segments:
            print(f"No segments found in {bag_path}")
            os.unlink(temp_audio_path)
            os.unlink(temp_segmentation_path)
            return []

        contact_segments = sum(1 for _, _, is_contact, _ in segments if is_contact)
        non_contact_segments = len(segments) - contact_segments

        print(
            f"Found {len(segments)} segments in {bag_path} ({contact_segments} contact, {non_contact_segments} non-contact)"
        )

        for i, segment in enumerate(segments):
            start, end, is_contact, windows = segment

            duration = end - start

            segment_type = "contact" if is_contact else "ambient"

            if not windows or len(windows) == 0:
                print(
                    f"Warning: Segment {i + 1} ({segment_type}) has no windows. Skipping."
                )
                continue

            print(
                f"Processing segment {i + 1} ({segment_type}): {len(windows)} windows"
            )

            for j, window_data in enumerate(windows):
                window_filename = (
                    f"{bag_name}_segment_{i}_window_{j}_{segment_type}.wav"
                )
                window_path = os.path.join(audio_dir, window_filename)
                sf.write(window_path, window_data, sr)

                window_middle_time = (
                    start
                    + j
                    * (
                        window_stride_seconds
                        if is_contact
                        else window_stride_seconds * 2
                    )
                    * sr
                    + len(window_data) / 2
                ) / sr

                try:
                    closest_image = None
                    min_time_diff = float("inf")

                    for topic in image_msgs:
                        for timestamp, img_msg in image_msgs[topic]:
                            time_diff = abs(timestamp - window_middle_time)
                            if time_diff < min_time_diff:
                                min_time_diff = time_diff
                                closest_image = img_msg

                    if closest_image is None:
                        print(f"No image found for window {i}_{j}")
                        continue

                    cv_image = bridge.imgmsg_to_cv2(closest_image, "bgr8")

                    img_filename = (
                        f"{bag_name}_segment_{i}_window_{j}_{segment_type}.jpg"
                    )
                    img_path = os.path.join(image_dir, img_filename)
                    cv2.imwrite(img_path, cv_image)

                    metadata.append(
                        {
                            "audio_file": os.path.join("audio", window_filename),
                            "image_file": os.path.join("images", img_filename),
                            "category": "ambient"
                            if not is_contact
                            else standard_category,
                        }
                    )

                except Exception as e:
                    print(f"Error processing image: {str(e)}")

        os.unlink(temp_audio_path)
        os.unlink(temp_segmentation_path)

        if (
            hasattr(cfg, "debug")
            and hasattr(cfg.debug, "enabled")
            and cfg.debug.enabled
        ):
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)

            orig_audio_path = os.path.join(debug_dir, f"{bag_name}_original.wav")
            sf.write(orig_audio_path, data, sr)
            print(f"Saved original audio to: {orig_audio_path}")

            denoised_audio_path = os.path.join(debug_dir, f"{bag_name}_denoised.wav")
            sf.write(denoised_audio_path, segmentation_audio_data, sr)
            print(f"Saved denoised audio to: {denoised_audio_path}")

            print("\n=== RUNNING SEGMENT_AUDIO ON ORIGINAL AUDIO ===")
            orig_segments = segment_audio(
                orig_audio_path,
                window_length_seconds=window_length_seconds,
                window_stride_seconds=window_stride_seconds,
                non_contact_threshold_factor=get_config_for_segment(
                    "non_contact_threshold_factor"
                ),
                enable_squeezing=get_config_for_segment("enable_squeezing"),
                squeeze_factor_seconds=0.3,
                dynamic_threshold_offset=get_config_for_segment(
                    "dynamic_threshold_offset"
                ),
                cfg=cfg,
                is_robot=is_robot,
            )
            print(f"Original audio segments: {len(orig_segments)}")
            contact_segments_orig = sum(
                1 for _, _, is_contact, _ in orig_segments if is_contact
            )
            print(
                f"Contact segments: {contact_segments_orig}, Non-contact segments: {len(orig_segments) - contact_segments_orig}"
            )

        print("\n=== FINAL METADATA ANALYSIS ===")
        contact_files = sum(1 for item in metadata if item["category"] != "ambient")
        non_contact_files = sum(1 for item in metadata if item["category"] == "ambient")
        print(f"Total files created: {len(metadata)}")
        print(f"Contact files: {contact_files}")
        print(f"Non-contact files: {non_contact_files}")
        print("===========================\n")

        if contact_files != contact_window_total:
            print(
                f"WARNING: Mismatch between detected contact windows ({contact_window_total}) and processed files ({contact_files})"
            )
            print("This suggests the window handling code is not working as expected")

        return metadata
    except Exception as e:
        print(f"Error processing bag {bag_path}: {str(e)}")
        if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if "temp_segmentation_path" in locals() and os.path.exists(
            temp_segmentation_path
        ):
            os.unlink(temp_segmentation_path)
        return []


@hydra.main(version_base=None, config_path="config", config_name="config")
def process_bags(cfg: DictConfig):
    """Process both probe and robot bag files and create separate datasets."""
    debug_mode = False
    if hasattr(cfg, "debug") and hasattr(cfg.debug, "enabled"):
        debug_mode = cfg.debug.enabled

    original_cwd = hydra.utils.get_original_cwd()

    probe_output_dir = os.path.join(original_cwd, cfg.output.probe_dataset_dir)
    robot_output_dir = os.path.join(original_cwd, cfg.output.robot_dataset_dir)

    os.makedirs(probe_output_dir, exist_ok=True)
    os.makedirs(robot_output_dir, exist_ok=True)

    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg.startswith("preprocessing.enable_denoising="):
                denoising_value = arg.split("=")[1].lower()
                if denoising_value in ["true", "false"]:
                    if not hasattr(cfg, "preprocessing"):
                        cfg.preprocessing = {}
                    cfg.preprocessing.enable_denoising = denoising_value == "true"
                    print(
                        f"Denoising {'enabled' if cfg.preprocessing.enable_denoising else 'disabled'} via command-line argument"
                    )

    probe_bag_dir = os.path.join(original_cwd, cfg.data.probe_bag_dir)
    print(f"Looking for probe bags in: {probe_bag_dir}")

    if not os.path.isdir(probe_bag_dir):
        print(f"ERROR: Probe bag directory not found: {probe_bag_dir}")
        print("Please check that the 'probe_bag_dir' path in config.yaml is correct.")
        print(f"Current setting: {cfg.data.probe_bag_dir}")
    else:
        probe_bags = glob.glob(
            os.path.join(probe_bag_dir, "**/*.bag"), recursive=True
        ) + glob.glob(os.path.join(probe_bag_dir, "**/*.BAG"), recursive=True)

        if not probe_bags:
            print(
                f"WARNING: No probe bag files (*.bag, *.BAG) found in {probe_bag_dir}"
            )
        else:
            print(f"Found {len(probe_bags)} probe bag files")
            probe_metadata = []
            for bag_path in tqdm(probe_bags, desc="Processing probe bags"):
                metadata = process_bag(
                    bag_path,
                    probe_output_dir,
                    window_length_seconds=cfg.data.window_length_seconds,
                    window_stride_seconds=cfg.data.window_stride_seconds,
                    cfg=cfg,
                    is_robot=False,  # Specify this is not robot data
                )
                probe_metadata.extend(metadata)

            if probe_metadata:
                df = pd.DataFrame(probe_metadata)
                csv_path = os.path.join(probe_output_dir, "dataset.csv")
                df.to_csv(csv_path, index=False)
                print(f"Created probe dataset with {len(probe_metadata)} examples")
            else:
                print(
                    "WARNING: No metadata collected from probe bags, dataset not created"
                )
                print("Check for issues in the process_bag function or bag contents")

    robot_bag_dir = os.path.join(original_cwd, cfg.data.robot_bag_dir)
    print(f"Looking for robot bags in: {robot_bag_dir}")

    if not os.path.isdir(robot_bag_dir):
        print(f"ERROR: Robot bag directory not found: {robot_bag_dir}")
        print("Please check that the 'robot_bag_dir' path in config.yaml is correct.")
        print(f"Current setting: {cfg.data.robot_bag_dir}")
        print("Common issues include typos or incorrect folder names.")
    else:
        robot_bags = glob.glob(
            os.path.join(robot_bag_dir, "**/*.bag"), recursive=True
        ) + glob.glob(os.path.join(robot_bag_dir, "**/*.BAG"), recursive=True)

        if not robot_bags:
            print(
                f"WARNING: No robot bag files (*.bag, *.BAG) found in {robot_bag_dir}"
            )
        else:
            print(f"Found {len(robot_bags)} robot bag files")
            robot_metadata = []
            for bag_path in tqdm(
                robot_bags, desc="Processing robot bags", disable=debug_mode
            ):
                if debug_mode and hasattr(cfg.debug, "target_bag"):
                    if cfg.debug.target_bag not in bag_path:
                        continue

                print(f"Processing bag: {bag_path}")
                metadata = process_bag(
                    bag_path,
                    robot_output_dir,
                    window_length_seconds=cfg.data.window_length_seconds,
                    window_stride_seconds=cfg.data.window_stride_seconds,
                    cfg=cfg,
                    is_robot=True,
                )
                robot_metadata.extend(metadata)

            if robot_metadata:
                df = pd.DataFrame(robot_metadata)
                csv_path = os.path.join(robot_output_dir, "dataset.csv")
                df.to_csv(csv_path, index=False)
                print(f"Created robot dataset with {len(robot_metadata)} examples")
            else:
                print(
                    "WARNING: No metadata collected from robot bags, dataset not created"
                )
                print("Check for issues in the process_bag function or bag contents")


def map_to_category(bag_name, cfg=None):
    """Map bag name to a standard category."""
    default_categories = ["leaf", "twig", "trunk", "ambient"]
    categories = default_categories
    if cfg and hasattr(cfg, "output") and hasattr(cfg.output, "categories"):
        categories = cfg.output.categories

    bag_name_lower = bag_name.lower()

    for category in categories:
        if category.lower() in bag_name_lower:
            return category

    print(f"WARNING: Could not determine category for bag: {bag_name}")
    print(f"  Available categories: {categories}")
    print(
        "  Bags must have a category name in their filename (e.g., 'leaf_sample.bag')"
    )
    print("  If this is intentional, you can ignore this warning.")

    ambient_keywords = ["ambient", "background", "noncontact"]
    if "ambient" in categories and any(
        keyword in bag_name_lower for keyword in ambient_keywords
    ):
        print(
            f"  Assigning to 'ambient' category based on keywords: {ambient_keywords}"
        )
        return "ambient"

    return None


def check_segments(segments, audio_data, sr):
    """Validate that segments are properly formatted."""
    if not segments or not isinstance(segments, list):
        print("Error: segments is not a valid list")
        return False

    for segment in segments:
        if not isinstance(segment, tuple) or len(segment) < 3:
            print(f"Error: segment {segment} is not a valid tuple")
            return False

        start, end, is_contact = segment[0], segment[1], segment[2]

        if not isinstance(start, (int, np.integer)) or not isinstance(
            end, (int, np.integer)
        ):
            print(f"Error: segment start/end indices are not integers: {start}, {end}")
            return False

        if start < 0 or end > len(audio_data) or start >= end:
            print(
                f"Error: segment indices out of bounds: {start}, {end}, audio length: {len(audio_data)}"
            )
            return False

    return True


def preprocess_audio(audio_data, sr, noise_data=None, noise_sr=None):
    """
    Preprocess audio by removing noise.

    Args:
        audio_data (np.ndarray): Audio data to process
        sr (int): Sample rate
        noise_data (np.ndarray): Pre-loaded noise data
        noise_sr (int): Sample rate of noise data

    Returns:
        np.ndarray: Preprocessed audio array
    """
    if noise_data is not None and noise_sr is not None:
        try:
            if sr != noise_sr:
                noise_y = librosa.resample(noise_data, orig_sr=noise_sr, target_sr=sr)
            else:
                noise_y = noise_data

            processed_audio = nr.reduce_noise(
                y=audio_data,
                sr=sr,
                y_noise=noise_y,
                prop_decrease=1.0,
                stationary=False,
            )

            return processed_audio
        except Exception as e:
            print(f"Error applying noise reduction: {e}")
            return audio_data
    else:
        return audio_data


def visualize_segmentation(audio_data, sr, envelope, thresholds, segments, output_path):
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    process_bags()

