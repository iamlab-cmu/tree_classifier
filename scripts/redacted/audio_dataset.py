import pandas as pd
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

def process_audio_file(audio_file, category):
    """
    Process a single audio file - only store filename and category
    """
    return {
        'filename': str(audio_file),
        'category': category
    }

@hydra.main(version_base=None, config_path="./config", config_name="config")
def create_audio_dataset(cfg: DictConfig):
    """
    Create a CSV dataset with just audio file paths and categories
    """
    input_folder = cfg.output.segments_dir
    output_file = cfg.output.features_file
    
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist")
        
    file_list = []
    total_files = 0
    
    # Count total files for progress bar
    for category in cfg.output.categories:
        category_path = os.path.join(input_folder, category)
        if os.path.isdir(category_path):
            total_files += len(list(Path(category_path).glob('*.wav')))
    
    # Process files with progress bar
    pbar = tqdm(total=total_files, desc="Processing audio files")
    
    for category in cfg.output.categories:
        category_path = os.path.join(input_folder, category)
        if not os.path.isdir(category_path):
            print(f"Warning: Category directory {category_path} does not exist")
            continue
            
        audio_files = list(Path(category_path).glob('*.wav'))
        
        for audio_file in audio_files:
            file_info = process_audio_file(str(audio_file), category)
            file_list.append(file_info)
            pbar.update(1)
            pbar.set_postfix({'category': category})
    
    pbar.close()
    
    if not file_list:
        print("Warning: No audio files were found")
        return
    
    # Create DataFrame
    print("\nCreating DataFrame and saving to CSV...")
    df = pd.DataFrame(file_list)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save file list
    df.to_csv(output_file, index=False)
    print(f"\nDataset created successfully: {output_file}")
    print(f"Total samples: {len(df)}")
    print("\nSamples per category:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    create_audio_dataset()
