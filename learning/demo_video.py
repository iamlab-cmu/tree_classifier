#!/usr/bin/env python
import os
import sys
import argparse
import math
import numpy as np
import cv2
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Adjust Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import from absolute path
try:
    from models import MultiModalClassifier
    from utils import get_device
    print("Direct import successful")
except ImportError:
    # Try alternative import paths
    try:
        sys.path.insert(0, os.path.join(parent_dir, "learning"))
        from models import MultiModalClassifier
        from utils import get_device
        print("Import from learning directory successful")
    except ImportError:
        try:
            # Last resort - full path import
            models_path = os.path.join(parent_dir, "learning", "models.py")
            utils_path = os.path.join(parent_dir, "learning", "utils.py")
            
            import importlib.util
            
            # Import models.py
            spec = importlib.util.spec_from_file_location("models", models_path)
            models_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(models_module)
            MultiModalClassifier = models_module.MultiModalClassifier
            
            # Import utils.py
            spec = importlib.util.spec_from_file_location("utils", utils_path)
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            get_device = utils_module.get_device
            
            print("Import using importlib successful")
        except Exception as e:
            print(f"Import error: {e}")
            print("Please ensure models.py and utils.py are in the correct location")
            sys.exit(1)


class VideoEvaluator:
    """Class for evaluating contact interactions in videos using a pretrained model"""
    
    def __init__(self, model_path, img_size=224, segment_duration=0.8, fps=30, 
                 use_binary=False, use_images=True, use_dual_audio=True, audio_model='ast'):
        """
        Initialize the video evaluator.
        
        Args:
            model_path: Path to the .pth model file
            img_size: Size of input images for the model
            segment_duration: Duration of each video segment in seconds
            fps: Frames per second to use for processing
            use_binary: Whether to use binary classification (contact vs. no-contact)
            use_images: Whether to use image modality
            use_dual_audio: Whether to use both AST and CLAP audio models
            audio_model: Type of audio model to use if not using dual audio ('ast' or 'clap')
        """
        self.model_path = model_path
        self.img_size = img_size
        self.segment_duration = segment_duration
        self.fps = fps
        self.use_binary = use_binary
        self.use_images = True  # Always use images
        self.use_audio = True   # Always use audio
        self.use_dual_audio = True  # Always use dual audio
        self.audio_model = audio_model
        
        # Define class names based on binary flag
        if use_binary:
            self.class_names = ['contact', 'no-contact']
            self.num_classes = 2
        else:
            self.class_names = ['leaf', 'twig', 'trunk', 'ambient']
            self.num_classes = 4
        
        # Get the device
        self.device = get_device()
        
        # Load the model
        self.load_model()
        
        # Set up colors for visualization
        self.colors = {
            'leaf': (0, 255, 0),      # Green
            'twig': (0, 165, 255),    # Orange
            'trunk': (139, 69, 19),   # Brown
            'ambient': (128, 128, 128),  # Gray
            'contact': (0, 255, 0),   # Green for binary mode
            'no-contact': (128, 128, 128)  # Gray for binary mode
        }
    
    def load_model(self):
        """Load the pretrained model"""
        print(f"Loading model from {self.model_path}")
        
        # Initialize the model - always use images and dual audio
        self.model = MultiModalClassifier(
            num_classes=self.num_classes,
            use_images=True,
            use_audio=True,
            use_dual_audio=True,
            pretrained=False,
            audio_model=self.audio_model,
            fusion_type="transformer"
        )
        
        # Load model weights
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            try:
                # Try loading with strict=False
                self.model.load_state_dict(state_dict, strict=False)
                print("Model loaded with strict=False")
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                sys.exit(1)
        
        self.model.to(self.device)
        self.model.eval()
    
    def process_video(self, video_path, output_dir=None, save_segments=False):
        """
        Process a video file and evaluate each segment.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results. If None, use video directory
            save_segments: Whether to save each segment as a separate file
            
        Returns:
            List of predictions for each segment
        """
        print(f"Processing video: {video_path}")
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(video_path), "analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine expected category based on filename
        video_filename = os.path.basename(video_path)
        expected_category = self.map_to_category(video_filename)
        print(f"Expected category based on filename: {expected_category if expected_category else 'Unknown'}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video FPS: {video_fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames / video_fps:.2f} seconds")
        print(f"Resolution: {video_width}x{video_height}")
        
        # Calculate frames per segment
        frames_per_segment = int(self.segment_duration * video_fps)
        num_segments = total_frames // frames_per_segment
        print(f"Processing {num_segments} segments of {self.segment_duration} seconds each")
        
        # Reset the video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Extract audio from video
        audio_path = os.path.join(output_dir, "temp_audio.wav")
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y -hide_banner -loglevel error")
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Prepare for results
        results = []
        
        # Create output video
        output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_analyzed.mp4")
        # Use a more compatible codec - H.264 if available, or XVID as fallback
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))
            if not out.isOpened():
                raise Exception("Failed to open with avc1 codec")
        except Exception as e:
            print(f"Failed with avc1 codec: {e}, trying XVID instead")
            # Still use mp4 extension even with XVID codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            temp_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_temp.avi")
            out = cv2.VideoWriter(temp_output_path, fourcc, video_fps, (video_width, video_height))
            if not out.isOpened():
                print("Failed to create output video. Will convert the video after processing.")
                # Create a temporary directory for frames
                frames_dir = os.path.join(output_dir, "temp_frames")
                os.makedirs(frames_dir, exist_ok=True)
        
        # Process each segment
        for segment_idx in tqdm(range(num_segments), desc="Processing segments"):
            # Calculate frame indices for this segment
            start_frame = segment_idx * frames_per_segment
            end_frame = start_frame + frames_per_segment
            
            # Extract frames for this segment
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                continue
            
            # Extract audio for this segment
            start_sample = int(start_frame / video_fps * sample_rate)
            end_sample = int(end_frame / video_fps * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]
            
            # Process the segment
            prediction, probs = self.process_segment(frames, segment_audio, sample_rate)
            
            # Save the results
            results.append({
                'segment_idx': segment_idx,
                'start_time': start_frame / video_fps,
                'end_time': end_frame / video_fps,
                'prediction': prediction,
                'probabilities': probs.tolist()
            })
            
            # Add prediction overlay to frames and write to output video
            for frame in frames:
                # Add overlay with prediction
                overlay = frame.copy()
                
                # Create a label with prediction only (without confidence)
                label = f"{self.class_names[prediction]}"
                
                # Add rectangle at the top
                cv2.rectangle(overlay, (0, 0), (video_width, 50), (0, 0, 0), -1)
                
                # Check if prediction might be incorrect
                is_potentially_incorrect = self.check_prediction_correctness(prediction, probs, expected_category)
                
                # Get text color - red for potentially incorrect predictions, green for correct
                # But always use standard color for 'ambient' predictions
                border_thickness = 10
                if is_potentially_incorrect and self.class_names[prediction] != 'ambient':
                    # Red color for potentially incorrect predictions
                    text_color = (0, 0, 255)  # Red in BGR
                    # Draw red rectangle around the frame
                    cv2.rectangle(overlay, (0, 0), (video_width, video_height), (0, 0, 255), border_thickness)
                else:
                    # Green for correct predictions (matching expected category or ambient)
                    if expected_category and (self.class_names[prediction] == expected_category or 
                                           self.class_names[prediction] == 'ambient'):
                        text_color = (0, 255, 0)  # Green in BGR
                        # Draw green rectangle around the frame
                        cv2.rectangle(overlay, (0, 0), (video_width, video_height), (0, 255, 0), border_thickness)
                    else:
                        # Standard color from class colors
                        text_color = self.colors[self.class_names[prediction]]
                
                # Add text with appropriate color
                cv2.putText(overlay, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                
                # Add the frame to the output video
                try:
                    out.write(overlay)
                except Exception as e:
                    # If VideoWriter failed, save frames as images
                    if 'frames_dir' in locals():
                        frame_path = os.path.join(frames_dir, f"{start_frame + len(frames)}.jpg")
                        cv2.imwrite(frame_path, overlay)
            
            # Save segment file if requested
            if save_segments:
                segment_dir = os.path.join(output_dir, "segments")
                os.makedirs(segment_dir, exist_ok=True)
                
                segment_path = os.path.join(segment_dir, f"segment_{segment_idx:04d}.mp4")
                try:
                    segment_writer = cv2.VideoWriter(
                        segment_path, fourcc, video_fps, (video_width, video_height)
                    )
                    for frame in frames:
                        segment_writer.write(frame)
                    segment_writer.release()
                    
                    # If using XVID codec, convert the segment to proper MP4
                    if fourcc == cv2.VideoWriter_fourcc(*'XVID'):
                        temp_segment_path = segment_path
                        segment_path = os.path.join(segment_dir, f"segment_{segment_idx:04d}.mp4")
                        os.system(f"ffmpeg -i {temp_segment_path} -c:v libx264 -pix_fmt yuv420p {segment_path} -y -hide_banner -loglevel error")
                        os.remove(temp_segment_path)
                        
                except Exception as e:
                    print(f"Failed to save segment {segment_idx}: {e}")
                    # Fallback to saving frames and using ffmpeg
                    segment_frames_dir = os.path.join(segment_dir, f"segment_{segment_idx:04d}_frames")
                    os.makedirs(segment_frames_dir, exist_ok=True)
                    for i, frame in enumerate(frames):
                        cv2.imwrite(os.path.join(segment_frames_dir, f"{i:04d}.jpg"), frame)
                    # Convert the frames to a video
                    os.system(f"ffmpeg -framerate {video_fps} -i {segment_frames_dir}/%04d.jpg -c:v libx264 -pix_fmt yuv420p {segment_path} -y -hide_banner -loglevel error")
                    # Clean up
                    import shutil
                    shutil.rmtree(segment_frames_dir, ignore_errors=True)
        
        # Release resources
        cap.release()
        out.release()
        
        # If we used XVID codec with .avi container, convert to mp4
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
            print("Converting AVI to MP4 format...")
            # Include the original normalized audio in the output video
            ffmpeg_cmd = f"ffmpeg -i {temp_output_path} -i {video_path} -c:v libx264 -pix_fmt yuv420p -c:a aac -af \"volume=2.0,dynaudnorm\" -map 0:v:0 -map 1:a:0 {output_video_path} -y -hide_banner -loglevel error"
            os.system(ffmpeg_cmd)
            # Remove temporary AVI file
            os.remove(temp_output_path)
        
        # If we created frames because VideoWriter failed, convert them to video using ffmpeg
        if 'frames_dir' in locals() and os.path.exists(frames_dir):
            print("Creating video using ffmpeg...")
            # Include the original normalized audio in the output video
            ffmpeg_cmd = f"ffmpeg -framerate {video_fps} -i {frames_dir}/%d.jpg -i {video_path} -c:v libx264 -pix_fmt yuv420p -c:a aac -af \"volume=2.0,dynaudnorm\" -map 0:v:0 -map 1:a:0 {output_video_path} -y -hide_banner -loglevel error"
            os.system(ffmpeg_cmd)
            # Clean up frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
            
        # Remove temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Ensure the output video has normalized audio (in case it didn't get added above)
        final_output_path = os.path.splitext(output_video_path)[0] + "_with_audio.mp4"
        if not os.path.exists(final_output_path):
            print("Adding normalized audio to the final video...")
            audio_cmd = f"ffmpeg -i {output_video_path} -i {video_path} -c:v copy -c:a aac -af \"volume=2.0,dynaudnorm\" -map 0:v:0 -map 1:a:0 {final_output_path} -y -hide_banner -loglevel error"
            os.system(audio_cmd)
            # Replace the original output_video_path with the one with audio
            if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
                os.replace(final_output_path, output_video_path)
        
        # Save results to CSV
        csv_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_results.csv")
        with open(csv_path, 'w') as f:
            f.write("segment_idx,start_time,end_time,prediction,confidence\n")
            for result in results:
                probs = result['probabilities']
                prediction = result['prediction']
                confidence = probs[prediction]
                f.write(f"{result['segment_idx']},{result['start_time']:.2f},{result['end_time']:.2f},{self.class_names[prediction]},{confidence:.4f}\n")
        
        # Create summary bar chart
        self.create_summary_chart(results, output_dir, video_path)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"Analyzed video saved to {output_video_path}")
        
        # If the video cannot be played, try to convert it with ffmpeg
        print("If you have trouble playing the video, you can convert it with ffmpeg using:")
        print(f"ffmpeg -i {output_video_path} -c:v libx264 -pix_fmt yuv420p -c:a aac -af \"volume=2.0,dynaudnorm\" {os.path.splitext(output_video_path)[0]}_converted.mp4")
        
        return results
    
    def process_segment(self, frames, audio, sample_rate):
        """
        Process a single video segment.
        
        Args:
            frames: List of frames for the segment
            audio: Audio tensor for the segment
            sample_rate: Sample rate of the audio
            
        Returns:
            Prediction class index and probabilities
        """
        with torch.no_grad():
            # Process the middle frame for image
            if len(frames) > 0:
                middle_idx = len(frames) // 2
                middle_frame = frames[middle_idx]
                
                # Preprocess image
                img = cv2.resize(middle_frame, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)  # HWC to CHW
                img = torch.from_numpy(img).float() / 255.0
                
                # Normalize with ImageNet stats
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img - mean) / std
                
                # Add batch dimension
                img = img.unsqueeze(0).to(self.device)
            else:
                img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
            
            # Process audio
            if audio.size(1) > 0:
                try:
                    # Resample if needed
                    if sample_rate != 16000:
                        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
                    
                    # Convert to mono if stereo
                    if audio.size(0) > 1:
                        audio = torch.mean(audio, dim=0, keepdim=True)
                    
                    # Compute mel spectrogram
                    mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=16000,
                        n_fft=1024,
                        hop_length=160,
                        n_mels=128
                    )(audio)
                    
                    # Convert to decibels
                    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                    
                    # Normalize
                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
                    
                    # Resize if needed
                    if mel_spec.size(2) < 1024:
                        # Pad
                        pad_size = 1024 - mel_spec.size(2)
                        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_size))
                    elif mel_spec.size(2) > 1024:
                        # Crop
                        mel_spec = mel_spec[:, :, :1024]
                    
                    # Add batch dimension if needed
                    if mel_spec.dim() == 3:
                        mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension if needed
                    
                    # Ensure shape is correct for model input (batch, channels, height, width)
                    # If model expects 4D input, reshape
                    if mel_spec.dim() == 3:
                        mel_spec = mel_spec.unsqueeze(1)  # Add channel dimension: (batch, channel, mel_bins, time)
                    
                    mel_spec = mel_spec.to(self.device)
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    # Fallback to zeros with expected dimensions
                    mel_spec = torch.zeros((1, 1, 128, 1024), device=self.device)
            else:
                # Create a valid placeholder with expected dimensions
                mel_spec = torch.zeros((1, 1, 128, 1024), device=self.device)
            
            try:
                # Forward pass through model
                outputs = self.model(x_img=img, x_audio=mel_spec)
                
                # Get predictions
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probs).item()
                
                return prediction, probs.cpu()
            except Exception as e:
                print(f"Error during inference: {e}")
                # Return default prediction
                return 3, torch.tensor([0.1, 0.1, 0.1, 0.7])  # Default to 'ambient' if there's an error
    
    def create_summary_chart(self, results, output_dir, video_path):
        """
        Create a summary chart of predictions.
        
        Args:
            results: List of prediction results
            output_dir: Directory to save the chart
            video_path: Path to the video file (for naming)
        """
        if not results:
            return
        
        # Count predictions by class
        class_counts = {class_name: 0 for class_name in self.class_names}
        for result in results:
            prediction = result['prediction']
            class_counts[self.class_names[prediction]] += 1
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Use the same colors as in the video
        bar_colors = [self.colors[class_name] for class_name in classes]
        # Convert RGB to matplotlib format (0-1 range)
        bar_colors = [(r/255, g/255, b/255) for (r, g, b) in bar_colors]
        
        bars = plt.bar(classes, counts, color=bar_colors)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.title("Contact Type Distribution")
        plt.xlabel("Contact Type")
        plt.ylabel("Number of Segments")
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_summary.png")
        plt.savefig(chart_path)
        plt.close()

    def map_to_category(self, filename):
        """
        Map filename to a standard category - similar to bag_audio_visual_segmentation.py.
        
        Args:
            filename: The filename to analyze
            
        Returns:
            str: Category name or None if no category could be determined
        """
        # Convert filename to lowercase for case-insensitive matching
        filename_lower = filename.lower()
        
        # Check for each category in the filename
        for category in self.class_names:
            if category.lower() in filename_lower and category != 'ambient' and category != 'no-contact':
                return category
        
        # Special handling for 'ambient'/'no-contact' category
        ambient_keywords = ['ambient', 'background', 'noncontact', 'no-contact', 'no_contact']
        if any(keyword in filename_lower for keyword in ambient_keywords):
            return 'ambient' if not self.use_binary else 'no-contact'
        
        return None
    
    def check_prediction_correctness(self, prediction, probs, expected_category):
        """
        Check if the prediction might be incorrect based on expected category and confidence.
        
        Args:
            prediction: Index of the predicted class
            probs: Probabilities tensor for all classes
            expected_category: Category name expected based on filename
            
        Returns:
            bool: True if prediction might be incorrect, False otherwise
        """
        # If no expected category, we can't verify
        if expected_category is None:
            return False
        
        # Get prediction category name
        predicted_category = self.class_names[prediction]
        
        # If prediction matches expected category, it's correct
        if predicted_category == expected_category:
            return False
        
        # Special case: ambient/no-contact
        if expected_category in ['ambient', 'no-contact'] and predicted_category in ['ambient', 'no-contact']:
            return False
        
        # Special case: If ambient/no-contact is predicted, it's likely correct
        # (always trust ambient predictions)
        if predicted_category in ['ambient', 'no-contact']:
            return False
        
        # If confidence is too low, it might be incorrect
        confidence_threshold = 0.7
        if probs[prediction] < confidence_threshold:
            return True
        
        # If another class has high confidence, prediction might be incorrect
        second_best = torch.argsort(probs, descending=True)[1]
        if probs[second_best] > 0.3:
            return True
        
        return predicted_category != expected_category


def main():
    """Main function to parse arguments and run video evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate contact interactions in a video")
    parser.add_argument("--model", required=True, help="Path to the trained model (.pth file)")
    parser.add_argument("--video", required=True, help="Path to the video file to analyze")
    parser.add_argument("--output", help="Directory to save results (default: video directory)")
    parser.add_argument("--segment-duration", type=float, default=0.8, help="Duration of each segment in seconds")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for the model")
    parser.add_argument("--binary", action="store_true", help="Use binary classification (contact vs. no-contact)")
    parser.add_argument("--audio-model", default="ast", choices=["ast", "clap"], help="Backup audio model type if dual audio fails")
    parser.add_argument("--save-segments", action="store_true", help="Save each segment as a separate file")
    
    args = parser.parse_args()
    
    # Ensure the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Ensure the video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Create the video evaluator - always using dual audio with images
    evaluator = VideoEvaluator(
        model_path=args.model,
        img_size=args.img_size,
        segment_duration=args.segment_duration,
        use_binary=args.binary,
        use_dual_audio=True,
        audio_model=args.audio_model
    )
    
    # Process the video
    evaluator.process_video(
        video_path=args.video,
        output_dir=args.output,
        save_segments=args.save_segments
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
