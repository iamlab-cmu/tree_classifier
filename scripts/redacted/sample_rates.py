import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
import soundfile as sf

@hydra.main(version_base=None, config_path="./config", config_name="config")
def analyze_audio_sample_rates(cfg: DictConfig):
    output_base_dir = cfg.output.base_dir
    os.makedirs(output_base_dir, exist_ok=True)
    
    sample_rates = []
    durations = []
    lengths = []
    file_names = []
    
    # Initialize counters for each category
    category_counts = {
        'short': 0,  # for files shorter than min_duration
        'standard': 0,  # for files meeting requirements
        'error': 0  # for files with processing errors
    }
    
    for category in os.listdir('./test'):
        category_path = os.path.join('./test', category)
        
        for file in os.listdir(category_path):
            audio_path = os.path.join(category_path, file)
            try:
                print(f"Processing file: {audio_path}")
                
                # Load the audio file
                y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves original sample rate
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Store information
                sample_rates.append(sr)
                durations.append(duration)
                lengths.append(len(y))
                file_names.append(file)
                
                # Categorize based on duration
                if duration < cfg.data.min_duration_factor:
                    category_counts['short'] += 1
                else:
                    category_counts['standard'] += 1
                
                print(f"Sample Rate: {sr} Hz")
                print(f"Duration: {duration:.2f} seconds")
                print(f"Number of samples: {len(y)}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                category_counts['error'] += 1
    
    # Create visualizations directory
    viz_dir = os.path.join(output_base_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Sample rates distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sample_rates, bins='auto', alpha=0.7)
    plt.title('Distribution of Sample Rates')
    plt.xlabel('Sample Rate (Hz)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(viz_dir, 'sample_rates_distribution.png'))
    plt.close()
    
    # Lengths distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7)
    plt.title('Distribution of Audio Lengths (samples)')
    plt.xlabel('Number of Samples')
    plt.ylabel('Count')
    plt.savefig(os.path.join(viz_dir, 'lengths_distribution.png'))
    plt.close()
    
    # Create pie chart of categories
    plt.figure(figsize=(10, 8))
    labels = list(category_counts.keys())
    sizes = list(category_counts.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Distribution of Audio Files by Category')
    plt.axis('equal')
    plt.savefig(os.path.join(viz_dir, 'category_distribution.png'))
    plt.close()
    
    # Save summary statistics
    with open(os.path.join(output_base_dir, 'sample_rate_statistics.txt'), 'w') as f:
        f.write("Summary Statistics:\n")
        f.write(f"Average sample rate: {np.mean(sample_rates):.2f} Hz\n")
        f.write(f"Most common sample rate: {max(set(sample_rates), key=sample_rates.count)} Hz\n")
        f.write(f"Average duration: {np.mean(durations):.2f} seconds\n")
        f.write(f"Average number of samples: {np.mean(lengths):.2f}\n\n")
        
        f.write("File counts by category:\n")
        for category, count in category_counts.items():
            f.write(f"{category}: {count}\n")

if __name__ == "__main__":
    analyze_audio_sample_rates()
