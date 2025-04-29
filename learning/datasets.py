import os
import glob
import random
import torch
import numpy as np
import cv2
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from hydra.utils import to_absolute_path


class AudioVisualDataset(Dataset):
    def __init__(self, dataframe, base_path="audio_visual_dataset", img_size=224, spec_size=(224, 224), 
                 use_binary_classification=False, augment=False, aug_config=None, do_norm=True):
        self.dataframe = dataframe
        self.base_path = base_path
        self.img_size = img_size
        self.spec_size = spec_size
        self.use_binary_classification = use_binary_classification
        self.augment = augment  
        self.aug_config = aug_config or {}  
        self.do_norm = do_norm  
        
        self.augmentation_probability = self.aug_config.get('augmentation_probability', 0.5)
        
        if augment and self.aug_config.get('image_rotation', True):
            max_degrees = self.aug_config.get('image_rotation_degrees', 30)
            self.image_transform = transforms.Compose([
                transforms.RandomRotation(degrees=max_degrees)
            ])
        else:
            self.image_transform = None
        
        self.use_frequency_scaling = augment and self.aug_config.get('audio_frequency_scaling', True)
        self.frequency_scaling_range = self.aug_config.get('frequency_scaling_range', [0.8, 1.2])
        
        self.use_frequency_shift = augment and self.aug_config.get('audio_frequency_shift', False)
        self.frequency_shift_range = self.aug_config.get('frequency_shift_range', [-20, 20])
        
        self.use_power_augmentation = augment and self.aug_config.get('audio_power', False)
        self.power_range = self.aug_config.get('power_range', [0.7, 1.3])
        
        self.use_time_stretch = augment and self.aug_config.get('audio_time_stretch', False)
        self.time_stretch_range = self.aug_config.get('time_stretch_range', [0.8, 1.2])
        
        self.use_gaussian_noise = augment and self.aug_config.get('audio_gaussian_noise', False)
        self.noise_level_range = self.aug_config.get('noise_level_range', [0.001, 0.01])
        
        self.use_clipping = augment and self.aug_config.get('audio_clipping', False)
        self.clipping_range = self.aug_config.get('clipping_range', [0.7, 0.95])
        
        self.use_robot_humming = augment and self.aug_config.get('audio_robot_humming', False)
        self.robot_humming_mix_ratio = self.aug_config.get('robot_humming_mix_ratio', [0.1, 0.4])
        
        self.use_polarity_inversion = augment and self.aug_config.get('audio_polarity_inversion', False)
        
        self.use_harmonic_distortion = augment and self.aug_config.get('audio_harmonic_distortion', False)
        self.harmonic_distortion_range = self.aug_config.get('harmonic_distortion_range', [0.1, 0.5])
        
        self.use_phase_distortion = augment and self.aug_config.get('audio_phase_distortion', False)
        self.phase_distortion_range = self.aug_config.get('phase_distortion_range', [0.1, 0.3])
        
        self.use_compression = augment and self.aug_config.get('audio_compression', False)
        self.compression_threshold_range = self.aug_config.get('compression_threshold_range', [-20, -10])
        self.compression_ratio_range = self.aug_config.get('compression_ratio_range', [2.0, 5.0])
        
        if self.use_robot_humming:
            self.robot_humming_samples = []
            robot_audio_path = to_absolute_path(self.aug_config.get('robot_humming_audio_path', ''))
            
            if os.path.exists(robot_audio_path):
                audio_files = []
                for ext in ['*.wav', '*.mp3', '*.ogg']:
                    audio_files.extend(glob.glob(os.path.join(robot_audio_path, ext)))
                
                
                for audio_file in audio_files:
                    try:
                        waveform, sample_rate = torchaudio.load(audio_file)
                        
                        if sample_rate != 16000:
                            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                        
                        rms = torch.sqrt(torch.mean(waveform**2))
                        if rms > 0:
                            waveform = waveform / rms
                        
                        self.robot_humming_samples.append(waveform)
                    except Exception as e:
                        print(f"Error loading robot humming sample {audio_file}: {e}")
                
            else:
                print(f"WARNING: Robot humming directory not found: {robot_audio_path}")
                print("Robot humming augmentation will be disabled")
                self.use_robot_humming = False
        
        if use_binary_classification:
            self.category_to_idx = {
                'leaf': 0,    # contact
                'twig': 0,    # contact
                'trunk': 0,   # contact
                'ambient': 1  # no-contact
            }
            self.idx_to_category = {
                0: 'contact',
                1: 'no-contact'
            }
        else:
            self.category_to_idx = {
                'leaf': 0,
                'twig': 1,
                'trunk': 2,
                'ambient': 3
            }
            self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}
    
    def __len__(self):
        return len(self.dataframe)
    
    def load_image(self, image_path):
        """Load and preprocess an image file"""
        full_path = os.path.join(self.base_path, image_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Image file not found: {full_path}")
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        
        try:
            image = cv2.imread(full_path)
            if image is None:
                print(f"Warning: Failed to read image: {full_path}")
                return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
                
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = image.transpose(2, 0, 1)  # HWC to CHW
            image = torch.from_numpy(image).float() / 255.0
            
            if self.augment and self.image_transform is not None:
                image_pil = transforms.ToPILImage()(image)
                image_pil = self.image_transform(image_pil)
                image = transforms.ToTensor()(image_pil)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
            return image
            
        except Exception as e:
            print(f"Error processing image {full_path}: {str(e)}")
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
    
    def load_audio(self, audio_path):
        """Load and preprocess an audio file for AST with random augmentations"""
        full_path = os.path.join(self.base_path, audio_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Audio file not found: {full_path}")
            return torch.zeros((1, 128, 1024), dtype=torch.float32)
        
        try:
            waveform, sample_rate = torchaudio.load(full_path)
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                sample_rate = 16000
            
            if waveform.numel() == 0 or waveform.shape[1] < sample_rate * 0.5:  # Less than 0.5 seconds
                print(f"Warning: Very short or empty audio in {full_path}")
                return torch.zeros((1, 128, 1024), dtype=torch.float32)
            
            if self.do_norm:
                rms = torch.sqrt(torch.mean(waveform**2))
                if rms > 0:
                    waveform = waveform / rms
            
            if self.augment:
                aug_probability = self.augmentation_probability
                
                if self.use_time_stretch and random.random() < aug_probability:
                    min_rate, max_rate = self.time_stretch_range
                    stretch_rate = random.uniform(min_rate, max_rate)
                    
                    if stretch_rate != 1.0:
                        effect = [
                            ["tempo", str(stretch_rate)]
                        ]
                        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                            waveform, sample_rate, effect
                        )
                
                if self.use_gaussian_noise and random.random() < aug_probability:
                    min_noise, max_noise = self.noise_level_range
                    noise_level = random.uniform(min_noise, max_noise)
                    
                    noise = torch.randn_like(waveform) * noise_level
                    waveform = waveform + noise
                
                if self.use_power_augmentation and random.random() < aug_probability:
                    min_power, max_power = self.power_range
                    power_factor = random.uniform(min_power, max_power)
                    
                    waveform = waveform * power_factor
                
                if self.use_clipping and random.random() < aug_probability:
                    min_clip, max_clip = self.clipping_range
                    clip_threshold = random.uniform(min_clip, max_clip)
                    
                    waveform = torch.clamp(waveform, min=-clip_threshold, max=clip_threshold)
                    
                    rms = torch.sqrt(torch.mean(waveform**2))
                    if rms > 0:
                        waveform = waveform / rms
                
                if self.use_robot_humming and self.robot_humming_samples and random.random() < aug_probability:
                    humming_idx = random.randint(0, len(self.robot_humming_samples) - 1)
                    humming_audio = self.robot_humming_samples[humming_idx]
                    
                    min_ratio, max_ratio = self.robot_humming_mix_ratio
                    mix_ratio = random.uniform(min_ratio, max_ratio)
                    
                    if humming_audio.shape[1] < waveform.shape[1]:
                        repeats = (waveform.shape[1] // humming_audio.shape[1]) + 1
                        humming_audio = humming_audio.repeat(1, repeats)
                    
                    humming_audio = humming_audio[:, :waveform.shape[1]]
                    
                    waveform = (1 - mix_ratio) * waveform + mix_ratio * humming_audio
                    
                    rms = torch.sqrt(torch.mean(waveform**2))
                    if rms > 0:
                        waveform = waveform / rms
                
                if self.use_polarity_inversion and random.random() < aug_probability:
                    waveform = -waveform
                
                if self.use_harmonic_distortion and random.random() < aug_probability:
                    min_distortion, max_distortion = self.harmonic_distortion_range
                    distortion_amount = random.uniform(min_distortion, max_distortion)
                    
                    waveform = torch.tanh(waveform * (1.0 + distortion_amount)) / torch.tanh(torch.tensor(1.0 + distortion_amount))
                    
                    rms = torch.sqrt(torch.mean(waveform**2))
                    if rms > 0:
                        waveform = waveform / rms
                
                if self.use_phase_distortion and random.random() < aug_probability:
                    min_phase, max_phase = self.phase_distortion_range
                    phase_amount = random.uniform(min_phase, max_phase)
                    
                    fft = torch.fft.rfft(waveform, dim=1)
                    
                    magnitude = torch.abs(fft)
                    phase = torch.angle(fft)
                    
                    phase_noise = torch.rand_like(phase) * phase_amount * 2 * 3.14159 - (phase_amount * 3.14159)
                    modified_phase = phase + phase_noise
                    
                    real = magnitude * torch.cos(modified_phase)
                    imag = magnitude * torch.sin(modified_phase)
                    modified_fft = torch.complex(real, imag)
                    
                    waveform = torch.fft.irfft(modified_fft, dim=1, n=waveform.shape[1])
                    
                    rms = torch.sqrt(torch.mean(waveform**2))
                    if rms > 0:
                        waveform = waveform / rms
                
                if self.use_compression and random.random() < aug_probability:
                    min_threshold, max_threshold = self.compression_threshold_range
                    min_ratio, max_ratio = self.compression_ratio_range
                    
                    threshold_db = random.uniform(min_threshold, max_threshold)
                    ratio = random.uniform(min_ratio, max_ratio)
                    
                    threshold = 10 ** (threshold_db / 20.0)
                    
                    amplitude = torch.abs(waveform)
                    
                    mask = amplitude > threshold
                    compressed = torch.where(
                        mask,
                        threshold + (amplitude - threshold) / ratio,
                        amplitude
                    )
                    
                    waveform = torch.sign(waveform) * compressed
                    
                    rms = torch.sqrt(torch.mean(waveform**2))
                    if rms > 0:
                        waveform = waveform / rms
            
            if self.do_norm:
                rms = torch.sqrt(torch.mean(waveform**2))
                if rms > 0:
                    waveform = waveform / rms
            
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128
            )(waveform)
            
            
            mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            
            if self.augment:
                aug_probability = self.augmentation_probability
                
                if self.use_frequency_scaling and random.random() < aug_probability:
                    min_scale, max_scale = self.frequency_scaling_range
                    scale_factor = random.uniform(min_scale, max_scale)
                    
                    _, freq_bins, time_frames = mel_spec_db.shape
                    
                    new_freq_bins = int(freq_bins * scale_factor)
                    new_freq_bins = max(16, min(freq_bins*2, new_freq_bins))  # Keep reasonable bounds
                    
                    mel_spec_db = mel_spec_db.permute(0, 2, 1)  # [1, time, freq]
                    mel_spec_db = torch.nn.functional.interpolate(
                        mel_spec_db, 
                        size=new_freq_bins, 
                        mode='linear', 
                        align_corners=False
                    )
                    
                    mel_spec_db = mel_spec_db.permute(0, 2, 1)  # [1, freq, time]
                    
                    if new_freq_bins != freq_bins:
                        mel_spec_db = mel_spec_db.permute(0, 2, 1)  # [1, time, freq]
                        mel_spec_db = torch.nn.functional.interpolate(
                            mel_spec_db, 
                            size=freq_bins, 
                            mode='linear', 
                            align_corners=False
                        )
                        mel_spec_db = mel_spec_db.permute(0, 2, 1)  # [1, freq, time]
                
                if self.use_frequency_shift and random.random() < aug_probability:
                    min_shift, max_shift = self.frequency_shift_range
                    shift_bins = random.randint(min_shift, max_shift)
                    
                    if shift_bins != 0:
                        _, freq_bins, time_frames = mel_spec_db.shape
                        
                        shifted_spec = torch.zeros_like(mel_spec_db)
                        
                        if shift_bins > 0:
                            shifted_spec[:, shift_bins:, :] = mel_spec_db[:, :(freq_bins-shift_bins), :]
                        else:
                            abs_shift = abs(shift_bins)
                            shifted_spec[:, :(freq_bins-abs_shift), :] = mel_spec_db[:, abs_shift:, :]
                        
                        mel_spec_db = shifted_spec
            
            if self.do_norm:
                mel_spec_db_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-10)
            else:
                mel_spec_db_norm = mel_spec_db
            
            target_length = 1024
            current_length = mel_spec_db_norm.shape[2]
            
            if current_length < target_length:
                padding = target_length - current_length
                log_mel_spectrogram = torch.nn.functional.pad(mel_spec_db_norm, (0, padding))
            else:
                log_mel_spectrogram = mel_spec_db_norm[:, :, :target_length]
            
            log_mel_spectrogram = log_mel_spectrogram.to(torch.float32)
            
            return log_mel_spectrogram
            
        except Exception as e:
            print(f"Error processing audio for {full_path}: {str(e)}")
            return torch.zeros((1, 128, 1024), dtype=torch.float32)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = row['audio_file']
        image_path = row['image_file']
        category = row['category']
        
        image = self.load_image(image_path)
        audio = self.load_audio(audio_path)
        
        if category in self.category_to_idx:
            label = torch.tensor(self.category_to_idx[category])
        else:
            print(f"Warning: Unknown category '{category}', defaulting to 0")
            label = torch.tensor(0)
        
        if self.use_binary_classification:
            display_category = 'contact' if self.category_to_idx[category] == 0 else 'no-contact'
        else:
            display_category = category
        
        return {
            'image': image,
            'audio': audio,
            'label': label,
            'category': display_category
        }


def custom_collate(batch):
    """Custom collate function to handle variable-sized tensors"""
    
    images = torch.stack([item['image'] for item in batch])
    
    processed_audio = []
    for i, item in enumerate(batch):
        audio = item['audio']
        if audio.dim() != 3 or audio.shape[0] != 1 or audio.shape[1] != 128 or audio.shape[2] != 1024:
            try:
                audio = audio.reshape(1, 128, 1024)
            except RuntimeError:
                audio = torch.zeros(1, 128, 1024, dtype=torch.float32)
        processed_audio.append(audio)
    
    audio = torch.stack(processed_audio)
    
    labels = torch.stack([item['label'] for item in batch])
    
    categories = [item['category'] for item in batch]
    
    return {
        'image': images,
        'audio': audio,
        'label': labels,
        'category': categories
    } 