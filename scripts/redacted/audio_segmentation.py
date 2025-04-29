import librosa
import numpy as np
import os
import shutil
from segment_audio import segment_audio
import soundfile as sf
import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

def process_window(window_path, category, force_data=None):
    """Process a single window and return its metadata"""
    metadata = {
        'filename': window_path,
        'category': category,
    }
    
    # Add force components if available - only means, no vectors
    if force_data is not None:
        metadata.update({
            'force_x_mean': force_data['fx_mean'],
            'force_y_mean': force_data['fy_mean'],
            'force_z_mean': force_data['fz_mean'],
            'force_magnitude_mean': force_data['magnitude_mean']
        })
    
    return metadata

def process_force_data(force_df, start_time, window_length):
    """Process force data for a given time window, returning only mean values"""
    # Filter force data for the current window
    window_data = force_df[
        (force_df['time'] >= start_time) & 
        (force_df['time'] < start_time + window_length)
    ]
    
    if len(window_data) == 0:
        return None
    
    # Calculate mean values only
    force_features = {
        'force_x_mean': window_data['force_x'].mean(),
        'force_y_mean': window_data['force_y'].mean(),
        'force_z_mean': window_data['force_z'].mean(),
        'force_magnitude_mean': window_data['magnitude'].mean()
    }
    
    return force_features

def save_segments(audio_file, segments, output_dir, force_df=None):
    """Save audio segments and their features to files"""
    # ... (existing setup code)
    
    for i, segment in enumerate(segments):
        # ... (existing audio processing code)
        
        # Process force data if available
        force_features = {}
        if force_df is not None:
            segment_force = process_force_data(force_df, segment['start'], segment['duration'])
            if segment_force:
                force_features.update(segment_force)
        
        # Create feature row
        feature_row = {
            'filename': segment_path,
            'category': category,
            **force_features  # Only includes mean values now
        }
        
        feature_rows.append(feature_row)
    
    return feature_rows

@hydra.main(version_base=None, config_path="./config", config_name="config")
def process_audio(cfg: DictConfig):
    # Wipe and recreate segments directory
    segments_dir = cfg.output.segments_dir
    if os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)
    os.makedirs(segments_dir)
    
    # Create directories for each category
    for category in cfg.output.categories:
        os.makedirs(os.path.join(segments_dir, category), exist_ok=True)
    
    # Initialize dataset list
    dataset = []
    
    # Determine if force data should be used
    use_force = cfg.data.get('use_force', True)  # Default to True if not specified
    
    if not use_force:
        print("Force data processing is disabled")
    
    for category in tqdm(cfg.output.categories):
        category_path = os.path.join(cfg.output.base_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory {category_path} does not exist")
            continue
            
        wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        
        for file in wav_files:
            audio_path = os.path.join(category_path, file)
            try:
                # Load the audio file first to get sample rate
                y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves original sample rate
                print(f"Processing {audio_path}, Sample Rate: {sr}Hz")
                
                # Get segments and force data
                if use_force:
                    segments, force_data = segment_audio(
                        audio_path,
                        window_length_seconds=cfg.data.window_length_seconds,
                        window_stride_seconds=cfg.data.window_stride_seconds,
                        force_dir=cfg.output.force_dir
                    )
                else:
                    # Don't pass force_dir if use_force is False
                    segments, force_data = segment_audio(
                        audio_path,
                        window_length_seconds=cfg.data.window_length_seconds,
                        window_stride_seconds=cfg.data.window_stride_seconds,
                        force_dir=None
                    )

                # Process segments and their windows
                for i, (start, end, is_contact, windows, _) in enumerate(segments):
                    if not is_contact:  # Skip non-contact segments
                        continue
                    
                    base_filename = os.path.splitext(file)[0]
                    
                    # Save each window and its metadata
                    for j, window in enumerate(windows):
                        window_filename = f"{base_filename}_segment_{i}_window_{j}.wav"
                        window_path = os.path.join(segments_dir, category, window_filename)
                        
                        try:
                            # Save window audio
                            window = window.astype(np.float32)
                            sf.write(window_path, window, sr)
                            
                            # Get force data for this window
                            window_force = None
                            if use_force and force_data is not None:
                                # Get force values for this window's time period
                                window_start = start/sr + j*cfg.data.window_stride_seconds
                                window_end = window_start + len(window)/sr
                                force_time = np.linspace(0, len(y)/sr, len(force_data['magnitude']))
                                window_mask = (force_time >= window_start) & (force_time <= window_end)
                                
                                window_force = {
                                    # Store only mean values
                                    'fx_mean': np.mean(force_data['force_x'][window_mask]),
                                    'fy_mean': np.mean(force_data['force_y'][window_mask]),
                                    'fz_mean': np.mean(force_data['force_z'][window_mask]),
                                    'magnitude_mean': np.mean(force_data['magnitude'][window_mask])
                                }
                            
                            # Add window metadata to dataset
                            metadata = process_window(window_path, category, window_force)
                            dataset.append(metadata)
                            
                        except Exception as e:
                            print(f"Error processing window {j} from segment {i}: {str(e)}")
                            continue
            
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue

    # Create and save dataset
    if dataset:
        df = pd.DataFrame(dataset)
        df.to_csv(cfg.output.features_file, index=False)
        print(f"\nDataset created successfully: {cfg.output.features_file}")
        print(f"Total samples: {len(df)}")
        print("\nSamples per category:")
        print(df['category'].value_counts())
        print("\nColumns in dataset:")
        print(df.columns.tolist())
    else:
        print("Warning: No data was processed for the dataset")

if __name__ == "__main__":
    process_audio() 
