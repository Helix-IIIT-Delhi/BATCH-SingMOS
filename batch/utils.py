"""Utility Functions for BATCH"""

import torch
import numpy as np


def extract_singing_features(audio_path: str, model) -> torch.Tensor:
    """Extract features from singing audio using pre-trained model"""
    import torchaudio

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Extract features
    with torch.no_grad():
        features = model(waveform)

    return features


def normalize_singmos(score: float, min_val: float = 0, max_val: float = 5) -> float:
    """Normalize SingMOS score to [0, 5] range"""
    return max(min_val, min(max_val, float(score)))


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
