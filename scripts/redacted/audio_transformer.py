import torch
import torch.nn as nn
import torchaudio
import numpy as np

class AudioTransformerConfig:
    def __init__(self):
        self.n_mels = 80  # Number of mel filterbanks
        self.hidden_dim = 256
        self.n_heads = 8
        self.n_layers = 4
        self.dropout = 0.1
        self.max_len = 1000  # Maximum sequence length

class AudioTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Mel spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_mels=config.n_mels
        )
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Input projection
        self.input_projection = nn.Linear(config.n_mels, config.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, 3)  # 3 classes: contact, no-contact, ambiguous
        
    def forward(self, x):
        # Convert to mel spectrogram
        x = self.mel_spec(x)  # [batch, n_mels, time]
        x = x.transpose(1, 2)  # [batch, time, n_mels]
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer(x)
        
        # Take mean over time dimension
        x = x.mean(dim=1)
        
        # Project to output classes
        x = self.output_projection(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 