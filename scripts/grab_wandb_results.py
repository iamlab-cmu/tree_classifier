#!/usr/bin/env python
"""
Script to extract test accuracy data from Weights & Biases runs.
"""
import os
import sys
import wandb
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract test accuracy data from W&B runs")
    parser.add_argument('--project', type=str, 
                        default="rspears-carnegie-mellon-university/contact-sound-classification-augmentations-v3",
                        help="W&B project path")
    parser.add_argument('--output-dir', type=str, default="wandb_results",
                        help="Directory for saving the extracted data")
    parser.add_argument('--filter-tag', type=str, default=None,
                        help="Filter runs by tag (e.g., 'window_ablation')")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    parser.add_argument('--max-runs', type=int, default=None,
                        help="Maximum number of runs to process (None means all runs)")
    return parser.parse_args()

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        debug = args.debug
        
        if debug:
            print("Debug mode enabled")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        if debug:
            print(f"Created output directory: {args.output_dir}")
        
        # Initialize W&B API
        api = wandb.Api()
        if debug:
            print("Initialized W&B API")
        
        # Get the project path
        project_path = args.project
        print(f"Fetching runs from W&B project: {project_path}")
        
        # Get all runs from the project
        try:
            # Try to get all runs at once (may be limited by W&B API)
            runs = api.runs(project_path, per_page=1000)  # Increase per_page to get more runs
            print(f"Retrieved {len(runs)} runs from W&B")
            
            # If you're using a newer W&B API that doesn't return all runs at once,
            # you might need a more complex approach to handle pagination
            if debug and len(runs) > 0:
                print(f"Note: Showing {len(runs)} runs, which may not be all runs if your project has more than 1000 runs")
            
        except TypeError:
            # Older API version may not support per_page parameter
            runs = api.runs(project_path)
            print(f"Retrieved {len(runs)} runs from W&B (using legacy API)")
        
        if debug:
            print(f"Retrieved {len(runs)} runs in total from W&B")
        
        # Filter runs by tag if specified
        if args.filter_tag:
            runs = [run for run in runs if args.filter_tag in run.tags]
            print(f"Filtered to {len(runs)} runs with tag '{args.filter_tag}'")
        
        # Limit number of runs if specified
        if args.max_runs is not None and len(runs) > args.max_runs:
            print(f"Limiting to {args.max_runs} runs (out of {len(runs)} total)")
            runs = runs[:args.max_runs]
        
        # Prepare data storage
        run_data = []
        
        # Extract data from each run
        for i, run in enumerate(runs):
            if debug and i % 10 == 0:
                print(f"Processing run {i+1}/{len(runs)}: {run.name}")
            
            # Basic run info
            run_info = {
                "run_id": run.id,
                "run_name": run.name,
                "tags": ",".join(run.tags),
                "state": run.state,
            }
            
            # Debug: print summary keys to see available metrics
            if debug and i == 0:
                print("Available metrics in first run:")
                for key in run.summary._json_dict:
                    print(f"  {key}: {run.summary._json_dict[key]}")
            
            # Extract config parameters of interest
            config = run.config
            for key in config:
                # Add key to run_info with prefix to avoid collisions
                run_info[f"config/{key}"] = config[key]
            
            # Special handling for window duration if it exists
            if "window_length_seconds" in config.get("data", {}):
                run_info["window_duration"] = config["data"]["window_length_seconds"]
            
            # Try different possible metric names for test accuracy
            accuracy_found = False
            for accuracy_key in ["test/accuracy", "test_accuracy", "accuracy", "test/Accuracy", "Test/Accuracy"]:
                if run.summary._json_dict.get(accuracy_key) is not None:
                    run_info["test_accuracy"] = run.summary._json_dict[accuracy_key]
                    if debug:
                        print(f"  Found test_accuracy as '{accuracy_key}': {run_info['test_accuracy']}")
                    accuracy_found = True
                    break
            
            if not accuracy_found and debug:
                print(f"  No test accuracy found for run {run.name}")
            
            # Try different possible metric names for binary accuracy
            binary_found = False
            for binary_key in ["test/binary_accuracy", "test_binary_accuracy", "binary_accuracy", 
                             "test/BinaryAccuracy", "Test/BinaryAccuracy"]:
                if run.summary._json_dict.get(binary_key) is not None:
                    run_info["test_binary_accuracy"] = run.summary._json_dict[binary_key]
                    if debug:
                        print(f"  Found test_binary_accuracy as '{binary_key}': {run_info['test_binary_accuracy']}")
                    binary_found = True
                    break
            
            if not binary_found and debug:
                print(f"  No binary accuracy found for run {run.name}")
            
            # Also look for loss under different names
            for loss_key in ["test/loss", "test_loss", "loss", "Test/Loss"]:
                if run.summary._json_dict.get(loss_key) is not None:
                    run_info["test_loss"] = run.summary._json_dict[loss_key]
                    break
            
            # Add to collection
            run_data.append(run_info)
        
        # Check if we have any data
        if not run_data:
            print("Warning: No run data was collected!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(run_data)
        if debug:
            print(f"Created DataFrame with shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Save to CSV
        csv_path = os.path.join(args.output_dir, "wandb_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
        # Create a simplified CSV with just the run names and accuracy metrics
        if len(run_data) > 0:
            # Create a simplified DataFrame with just run names and accuracy values
            simple_data = []
            for run in run_data:
                simple_info = {
                    "Run Name": run["run_name"]
                }
                
                if "test_accuracy" in run:
                    simple_info["Test Accuracy"] = run["test_accuracy"]
                
                if "test_binary_accuracy" in run:
                    simple_info["Test Binary Accuracy"] = run["test_binary_accuracy"]
                    
                simple_data.append(simple_info)
            
            simple_df = pd.DataFrame(simple_data)
            if debug:
                print(f"Created simplified DataFrame with shape: {simple_df.shape}")
                print(f"Simplified DataFrame columns: {simple_df.columns.tolist()}")
            
            simple_csv_path = os.path.join(args.output_dir, "wandb_test_accuracy.csv")
            simple_df.to_csv(simple_csv_path, index=False)
            print(f"Saved simplified accuracy results to {simple_csv_path}")
        
        # Check if we have the test_accuracy column before trying to visualize
        if "test_accuracy" in df.columns:
            print("\nTest Accuracy Statistics:")
            print(df["test_accuracy"].describe())
            
            # Create a directory for plots
            plots_dir = os.path.join(args.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            if debug:
                print(f"Created plots directory: {plots_dir}")
            
            try:
                # Plot test accuracy if we have window duration data
                if "window_duration" in df.columns:
                    plt.figure(figsize=(10, 6))
                    # Use basic matplotlib instead of seaborn to avoid compatibility issues
                    plt.plot(df["window_duration"], df["test_accuracy"], 'o-')
                    plt.title("Test Accuracy vs. Window Duration")
                    plt.xlabel("Window Duration (seconds)")
                    plt.ylabel("Test Accuracy")
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save the plot
                    plot_path = os.path.join(plots_dir, "test_accuracy_vs_window_duration.png")
                    plt.savefig(plot_path)
                    print(f"Saved plot to {plot_path}")
                
                # Basic histogram of test accuracies using matplotlib instead of seaborn
                plt.figure(figsize=(10, 6))
                plt.hist(df["test_accuracy"], bins=10, alpha=0.7, edgecolor='black')
                plt.title("Distribution of Test Accuracy")
                plt.xlabel("Test Accuracy")
                plt.ylabel("Count")
                plt.grid(True)
                plt.tight_layout()
                
                # Save the histogram
                hist_path = os.path.join(plots_dir, "test_accuracy_histogram.png")
                plt.savefig(hist_path)
                print(f"Saved histogram to {hist_path}")
            except Exception as viz_error:
                print(f"Warning: Error creating test_accuracy visualizations: {str(viz_error)}")
                if debug:
                    print(traceback.format_exc())
        else:
            print("Warning: 'test_accuracy' column not found in DataFrame!")
            if debug:
                print(f"Available columns: {df.columns.tolist()}")
        
        # Add visualizations for binary accuracy
        if "test_binary_accuracy" in df.columns:
            print("\nTest Binary Accuracy Statistics:")
            print(df["test_binary_accuracy"].describe())
            
            # Create a directory for plots if not already created
            plots_dir = os.path.join(args.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            try:
                # Plot binary accuracy if we have window duration data
                if "window_duration" in df.columns:
                    plt.figure(figsize=(10, 6))
                    # Use basic matplotlib instead of seaborn
                    plt.plot(df["window_duration"], df["test_binary_accuracy"], 'o-')
                    plt.title("Binary Test Accuracy vs. Window Duration")
                    plt.xlabel("Window Duration (seconds)")
                    plt.ylabel("Binary Test Accuracy")
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save the plot
                    plot_path = os.path.join(plots_dir, "binary_test_accuracy_vs_window_duration.png")
                    plt.savefig(plot_path)
                    print(f"Saved plot to {plot_path}")
                
                # Basic histogram of binary test accuracies
                plt.figure(figsize=(10, 6))
                plt.hist(df["test_binary_accuracy"], bins=10, alpha=0.7, edgecolor='black')
                plt.title("Distribution of Binary Test Accuracy")
                plt.xlabel("Binary Test Accuracy")
                plt.ylabel("Count")
                plt.grid(True)
                plt.tight_layout()
                
                # Save the histogram
                hist_path = os.path.join(plots_dir, "binary_test_accuracy_histogram.png")
                plt.savefig(hist_path)
                print(f"Saved histogram to {hist_path}")
            except Exception as viz_error:
                print(f"Warning: Error creating test_binary_accuracy visualizations: {str(viz_error)}")
                if debug:
                    print(traceback.format_exc())
        else:
            print("Warning: 'test_binary_accuracy' column not found in DataFrame!")
            if debug:
                print(f"Available columns: {df.columns.tolist()}")
        
        print("\nDone!")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 