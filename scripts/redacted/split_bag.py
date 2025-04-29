import rosbag
import argparse
from datetime import datetime
import os

def get_bag_duration(bag_path):
    """
    Get the duration of the bag file in seconds.
    
    Args:
        bag_path (str): Path to the ROS bag file
        
    Returns:
        float: Duration in seconds
    """
    with rosbag.Bag(bag_path, 'r') as bag:
        start_time = None
        end_time = None
        
        # Get first message time
        for _, _, t in bag.read_messages():
            start_time = t.to_sec()
            break
        
        # Get last message time
        for _, _, t in bag.read_messages():
            end_time = t.to_sec()
        
        return end_time - start_time if start_time and end_time else 0

def split_bag(bag_path, split_points, output_dir=None):
    """
    Split a bag file at specified time points.
    
    Args:
        bag_path (str): Path to the ROS bag file
        split_points (list): List of times in seconds where to split the bag
        output_dir (str): Directory to place output files (optional)
    """
    duration = get_bag_duration(bag_path)
    
    # Validate split points
    for point in split_points:
        if point <= 0 or point >= duration:
            raise ValueError(f"Split point {point}s is out of bounds. Bag duration is {duration}s")
    
    # Sort split points to ensure sequential splitting
    split_points = sorted(split_points)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(bag_path))[0]
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(output_dir, base_name)
    else:
        # Use same directory as input if no output dir specified
        base_path = os.path.splitext(bag_path)[0]
        os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)
    
    with rosbag.Bag(bag_path, 'r') as bag:
        # Get start time of the bag
        for _, _, t in bag.read_messages():
            start_time = t.to_sec()
            break
        
        # Create split ranges including start and end
        ranges = [(start_time + (0 if i == 0 else split_points[i-1]), 
                  start_time + (split_points[i] if i < len(split_points) else duration)) 
                 for i in range(len(split_points) + 1)]
        
        # Create a bag file for each range
        for i, (range_start, range_end) in enumerate(ranges):
            output_path = f"{base_path}_split_{i}.bag"
            print(f"Creating {output_path} ({range_start-start_time:.2f}s to {range_end-start_time:.2f}s)")
            
            with rosbag.Bag(output_path, 'w') as outbag:
                for topic, msg, t in bag.read_messages():
                    # Include messages within the current time range
                    if range_start <= t.to_sec() <= range_end:
                        outbag.write(topic, msg, t)

def main():
    parser = argparse.ArgumentParser(description='Split ROS bag file at specified time points')
    parser.add_argument('bag_path', help='Path to the ROS bag file')
    parser.add_argument('split_points', type=float, nargs='+', 
                      help='Time points in seconds where to split the bag')
    parser.add_argument('--output-dir', '-o', 
                      help='Directory to place output files (optional)')
    args = parser.parse_args()
    
    try:
        split_bag(args.bag_path, args.split_points, args.output_dir)
        print("Bag splitting completed successfully")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
