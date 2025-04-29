"""
Script to create different dataset variations by toggling window duration parameters.
"""

import os
import subprocess
import argparse
import datetime
from pathlib import Path
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate datasets with different window durations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for saving the generated datasets",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dataset generation if output directories already exist",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    base_output_dir = args.output_dir or f"window_duration_ablation_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    window_configs = []
    for i in range(1, 5):  # Modified range from (1, 5) to get 0.1, 0.2, 0.3, 0.4
        min_duration = round(i * 0.1, 1)  # 0.1, 0.2, 0.3, 0.4
        stride = 0.2  # Set stride to 0.2 for all configs

        window_configs.append(
            {
                "name": f"window_{int(min_duration * 10)}",
                "window_length_seconds": min_duration,
                "window_stride_seconds": stride,
                "description": f"Window duration: {min_duration}s, stride: {stride}s (fixed stride)",
            }
        )

    print(f"=== Generating {len(window_configs)} dataset variations ===")
    for config in window_configs:
        print(f"- {config['name']}: {config['description']}")

    for config in window_configs:
        probe_dir = os.path.join(
            base_output_dir, f"audio_visual_dataset_{config['name']}"
        )
        robot_dir = os.path.join(
            base_output_dir, f"audio_visual_dataset_robo_{config['name']}"
        )

        if (
            args.skip_existing
            and os.path.exists(probe_dir)
            and os.path.exists(robot_dir)
        ):
            print(
                f"\nSkipping {config['name']} configuration (directories already exist)"
            )
            continue

        print(f"\n=== Processing {config['name']} configuration ===")
        print(f"Description: {config['description']}")
        print(f"window_length_seconds: {config['window_length_seconds']}")
        print(f"window_stride_seconds: {config['window_stride_seconds']}")
        print(f"normalization: enabled")
        print(f"denoising: enabled")

        python_executable = (
            sys.executable
        )  # Gets the path to the current Python interpreter
        command = [
            python_executable,
            "scripts/bag_audio_visual_segmentation.py",
            f"data.window_length_seconds={config['window_length_seconds']}",
            f"data.window_stride_seconds={config['window_stride_seconds']}",
            "preprocessing.do_norm=true",
            "preprocessing.enable_denoising=true",
            f"output.probe_dataset_dir={probe_dir}",
            f"output.robot_dataset_dir={robot_dir}",
        ]

        print(f"Running command: {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            for line in process.stdout:
                print(line, end="")

            return_code = process.wait()

            if return_code == 0:
                print(
                    f"\n✅ Successfully generated datasets for {config['name']} configuration"
                )

                probe_audio_dir = os.path.join(probe_dir, "audio")
                probe_image_dir = os.path.join(probe_dir, "images")
                robot_audio_dir = os.path.join(robot_dir, "audio")
                robot_image_dir = os.path.join(robot_dir, "images")

                if os.path.exists(probe_audio_dir):
                    probe_audio_count = len(
                        [f for f in os.listdir(probe_audio_dir) if f.endswith(".wav")]
                    )
                    print(f"Probe audio files: {probe_audio_count}")

                if os.path.exists(probe_image_dir):
                    probe_image_count = len(
                        [
                            f
                            for f in os.listdir(probe_image_dir)
                            if f.endswith((".jpg", ".png"))
                        ]
                    )
                    print(f"Probe image files: {probe_image_count}")

                if os.path.exists(robot_audio_dir):
                    robot_audio_count = len(
                        [f for f in os.listdir(robot_audio_dir) if f.endswith(".wav")]
                    )
                    print(f"Robot audio files: {robot_audio_count}")

                if os.path.exists(robot_image_dir):
                    robot_image_count = len(
                        [
                            f
                            for f in os.listdir(robot_image_dir)
                            if f.endswith((".jpg", ".png"))
                        ]
                    )
                    print(f"Robot image files: {robot_image_count}")
            else:
                print(
                    f"\n❌ Failed to generate datasets for {config['name']} configuration (return code {return_code})"
                )

        except Exception as e:
            print(
                f"\n❌ Error generating datasets for {config['name']} configuration: {e}"
            )

    readme_path = os.path.join(base_output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Window Duration Ablation Study\n\n")
        f.write(
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("## Dataset Variations\n\n")
        f.write("All dataset variations include:\n")
        f.write("- **Normalization**: Enabled\n")
        f.write("- **Denoising**: Enabled\n\n")

        for config in window_configs:
            f.write(f"### {config['name']}\n")
            f.write(f"- **Description**: {config['description']}\n")
            f.write(f"- **Window Length**: {config['window_length_seconds']} seconds\n")
            f.write(
                f"- **Window Stride**: {config['window_stride_seconds']} seconds (fixed stride)\n"
            )
            f.write(
                f"- **Probe dataset directory**: `audio_visual_dataset_{config['name']}`\n"
            )
            f.write(
                f"- **Robot dataset directory**: `audio_visual_dataset_robo_{config['name']}`\n\n"
            )

        f.write("## Usage with Training\n\n")
        f.write("To train on a specific dataset variation, run:\n\n")
        f.write("```bash\n")
        f.write("python learning/train.py \\\n")
        f.write("    data.train_base_path=path/to/audio_visual_dataset_window_X \\\n")
        f.write("    data.test_base_path=path/to/audio_visual_dataset_robo_window_X\n")
        f.write("```\n\n")
        f.write(
            "Replace `X` with one of the window duration values (1, 2, 3, ..., 10).\n"
        )

    print(f"\n=== Window Duration Ablation Complete ===")
    print(f"Generated datasets are available in: {base_output_dir}")
    print(
        f"A README.md file with dataset descriptions has been created at: {readme_path}"
    )


if __name__ == "__main__":
    main()
