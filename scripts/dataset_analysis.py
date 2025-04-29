import os
import sys
import glob
import pandas as pd
from collections import Counter

def main():
    # Allow CSV path as a command-line argument (with wildcard support)
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = "learning/audio_visual_dataset_robo_window*"  # Default pattern

    # Use glob to expand the pattern
    matching_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not matching_dirs:
        print(f"No directories found matching: {pattern}")
        return

    # Look for dataset.csv in each matched directory
    found_csvs = []
    for d in matching_dirs:
        csv_path = os.path.join(d, "dataset.csv")
        if os.path.isfile(csv_path):
            found_csvs.append(csv_path)

    if not found_csvs:
        print(f"No dataset.csv files found in matched directories: {matching_dirs}")
        return

    # Update/add window_stride_seconds for each CSV
    for csv_path in found_csvs:
        print(f"\nProcessing dataset CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Add or update the window_stride_seconds column
        df['window_stride_seconds'] = 0.2

        # Save the updated CSV (overwrite)
        df.to_csv(csv_path, index=False)
        print(f"Set window_stride_seconds=0.2 for all rows in {csv_path}")

        # Optionally, print class distribution
        if 'category' in df.columns:
            class_counts = Counter(df['category'])
            print("Data distribution (class: count):")
            for class_name, count in class_counts.most_common():
                print(f"{class_name}: {count}")

if __name__ == "__main__":
    main()
