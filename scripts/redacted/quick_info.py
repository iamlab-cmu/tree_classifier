import rosbag
import argparse
import numpy as np
from collections import defaultdict

def get_durations(bag_path):
    """Get audio and video durations from a ROS bag file."""
    with rosbag.Bag(bag_path, 'r') as bag:
        # Get camera info
        print("\nCamera Info:")
        first_info = True
        for topic, msg, t in bag.read_messages(topics=['/camera1/color/camera_info']):
            if first_info:
                print("\nFirst camera info message:")
                for attr in dir(msg):
                    if not attr.startswith('_'):
                        value = getattr(msg, attr)
                        print(f"{attr}: {value}")
                first_info = False
                
                # Print specific important fields
                print("\nImportant camera parameters:")
                print(f"Resolution: {msg.width}x{msg.height}")
                print(f"Distortion model: {msg.distortion_model}")
                print(f"Frame ID: {msg.header.frame_id}")
                print(f"K (camera matrix):\n{np.array(msg.K).reshape(3,3)}")
                print(f"D (distortion coefficients): {msg.D}")
                break
        
        # Get video info
        video_start = None
        video_end = None
        frame_times = []
        header_times = []
        
        for topic, msg, t in bag.read_messages(topics=['/camera1/color/image_raw']):
            frame_time = t.to_sec()
            header_time = msg.header.stamp.to_sec()
            frame_times.append(frame_time)
            header_times.append(header_time)
            if video_start is None:
                video_start = frame_time
            video_end = frame_time
        
        # Analyze frame timing
        if frame_times:
            frame_intervals = np.diff(frame_times)
            header_intervals = np.diff(header_times)
            
            print("\nVideo Info:")
            print(f"Number of frames: {len(frame_times)}")
            print("\nBag timestamps:")
            print(f"First frame: {frame_times[0]:.3f}")
            print(f"Last frame: {frame_times[-1]:.3f}")
            print(f"Duration: {frame_times[-1] - frame_times[0]:.3f}")
            print(f"Average frame interval: {np.mean(frame_intervals):.3f} seconds")
            print(f"Expected FPS: {1/np.mean(frame_intervals):.1f}")
            
            print("\nHeader timestamps:")
            print(f"First frame: {header_times[0]:.3f}")
            print(f"Last frame: {header_times[-1]:.3f}")
            print(f"Duration: {header_times[-1] - header_times[0]:.3f}")
            print(f"Average frame interval: {np.mean(header_intervals):.3f} seconds")
            print(f"Expected FPS: {1/np.mean(header_intervals):.1f}")
            
            # Check for large gaps
            large_gaps = frame_intervals[frame_intervals > 0.1]  # gaps > 100ms
            if len(large_gaps) > 0:
                print(f"\nFound {len(large_gaps)} large gaps in video:")
                for gap in large_gaps:
                    gap_idx = np.where(frame_intervals == gap)[0][0]
                    gap_time = frame_times[gap_idx]
                    print(f"Gap of {gap:.3f}s at time {gap_time:.2f}s")
        
        # Get audio info
        audio_info = None
        for topic, msg, t in bag.read_messages(topics=['/audio_info']):
            audio_info = msg
            print("\nAudio Info Message:")
            for attr in dir(msg):
                if not attr.startswith('_'):
                    print(f"{attr}: {getattr(msg, attr)}")
            break
        
        # Calculate durations
        video_duration = video_end - video_start if video_start and video_end else 0
        
        # Print all topics and their message counts
        print("\nTopic Info:")
        for topic, info in bag.get_type_and_topic_info().topics.items():
            print(f"{topic}: {info.message_count} messages")
        
        return video_duration

def main():
    parser = argparse.ArgumentParser(description='Get audio and video durations from ROS bag')
    parser.add_argument('bag_path', help='Path to the ROS bag file')
    args = parser.parse_args()
    
    video_duration = get_durations(args.bag_path)
    print(f"\nVideo duration: {video_duration:.2f} seconds")

if __name__ == '__main__':
    main()
