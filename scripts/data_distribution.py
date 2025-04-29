import os
import pandas as pd
from collections import Counter
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import sys

def get_config():
    """Load the configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'config.yaml')
    if os.path.exists(config_path):
        return OmegaConf.load(config_path)
    else:
        print(f"Warning: Config file not found at {config_path}")
        return OmegaConf.create({"output": {"categories": ["leaf", "twig", "trunk", "ambient"]}})

def count_categories(dataset_path):
    """
    Count the number of examples in each category in the dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        Counter: Counter object with category counts
    """
    csv_path = os.path.join(dataset_path, 'dataset.csv')
    if not os.path.exists(csv_path):
        print(f"Dataset CSV file not found at {csv_path}")
        return Counter()
    
    try:
        df = pd.read_csv(csv_path)
        if 'category' not in df.columns:
            print(f"Error: 'category' column not found in {csv_path}")
            return Counter()
        
        return Counter(df['category'])
    except Exception as e:
        print(f"Error reading dataset CSV file: {e}")
        return Counter()

def plot_distribution(probe_counts, robot_counts, categories):
    """Create a bar chart of the category distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the bar positions
    bar_width = 0.35
    index = range(len(categories))
    
    # Create the bars
    probe_bars = ax.bar([i - bar_width/2 for i in index], 
                        [probe_counts.get(c, 0) for c in categories], 
                        bar_width, label='Probe Data')
    
    robot_bars = ax.bar([i + bar_width/2 for i in index], 
                        [robot_counts.get(c, 0) for c in categories], 
                        bar_width, label='Robot Data')
    
    # Add labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Examples')
    ax.set_title('Distribution of Examples by Category')
    ax.set_xticks(index)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add count labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    add_labels(probe_bars)
    add_labels(robot_bars)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('category_distribution.png')
    print(f"Distribution plot saved as 'category_distribution.png'")
    
    # Show the plot if not running in a headless environment
    try:
        plt.show()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Display category distribution in audio-visual datasets')
    parser.add_argument('--probe-dir', type=str, help='Path to probe dataset directory',
                        default='audio_visual_dataset')
    parser.add_argument('--robot-dir', type=str, help='Path to robot dataset directory',
                        default='audio_visual_dataset_robo')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plot')
    args = parser.parse_args()
    
    # Load config to get categories
    cfg = get_config()
    categories = cfg.output.categories if 'output' in cfg and 'categories' in cfg.output else ['leaf', 'twig', 'trunk', 'ambient']
    
    # Add current directory to search paths
    probe_paths = [
        args.probe_dir,
        os.path.join(os.getcwd(), args.probe_dir)
    ]
    
    robot_paths = [
        args.robot_dir,
        os.path.join(os.getcwd(), args.robot_dir)
    ]
    
    # Find existing dataset paths
    probe_path = next((p for p in probe_paths if os.path.exists(p)), None)
    robot_path = next((p for p in robot_paths if os.path.exists(p)), None)
    
    if not probe_path and not robot_path:
        print(f"Error: Neither probe dataset ({args.probe_dir}) nor robot dataset ({args.robot_dir}) found")
        sys.exit(1)
    
    # Count categories
    probe_counts = count_categories(probe_path) if probe_path else Counter()
    robot_counts = count_categories(robot_path) if robot_path else Counter()
    
    # Print results
    print("\n===== Dataset Distribution =====")
    print("\nProbe Dataset:")
    if probe_path:
        total_probe = sum(probe_counts.values())
        if total_probe > 0:
            for category in categories:
                count = probe_counts.get(category, 0)
                percentage = (count / total_probe) * 100 if total_probe > 0 else 0
                print(f"  {category}: {count} ({percentage:.1f}%)")
            print(f"  Total: {total_probe}")
        else:
            print("  No examples found")
    else:
        print("  Dataset not found")
    
    print("\nRobot Dataset:")
    if robot_path:
        total_robot = sum(robot_counts.values())
        if total_robot > 0:
            for category in categories:
                count = robot_counts.get(category, 0)
                percentage = (count / total_robot) * 100 if total_robot > 0 else 0
                print(f"  {category}: {count} ({percentage:.1f}%)")
            print(f"  Total: {total_robot}")
        else:
            print("  No examples found")
    else:
        print("  Dataset not found")
    
    # Create combined total
    total_counts = probe_counts + robot_counts
    total_examples = sum(total_counts.values())
    
    print("\nCombined Total:")
    if total_examples > 0:
        for category in categories:
            count = total_counts.get(category, 0)
            percentage = (count / total_examples) * 100 if total_examples > 0 else 0
            print(f"  {category}: {count} ({percentage:.1f}%)")
        print(f"  Total: {total_examples}")
    else:
        print("  No examples found")
    
    # Create plot if there's data and not disabled
    if not args.no_plot and (sum(probe_counts.values()) > 0 or sum(robot_counts.values()) > 0):
        try:
            plot_distribution(probe_counts, robot_counts, categories)
        except Exception as e:
            print(f"Warning: Could not create distribution plot: {e}")

if __name__ == "__main__":
    main()
