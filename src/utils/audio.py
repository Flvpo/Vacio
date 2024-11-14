import numpy as np
import soundfile as sf
import librosa
import tempfile
import os
from typing import Tuple, Optional
import subprocess

def verify_audio_quality(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Verify the quality of audio data.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sampling rate
        
    Returns:
        Dictionary with quality metrics
    """
    diagnostics = {
        'duration': len(audio_data) / sample_rate,
        'max_amplitude': np.max(np.abs(audio_data)),
        'mean_amplitude': np.mean(np.abs(audio_data)),
        'silence_ratio': np.sum(np.abs(audio_data) < 0.01) / len(audio_data),
        'is_good_quality': True,
        'issues': []
    }
    
    if diagnostics['duration'] < 0.3:
        diagnostics['is_good_quality'] = False
        diagnostics['issues'].append('Audio too short')
    
    if diagnostics['max_amplitude'] < 0.1:
        diagnostics['is_good_quality'] = False
        diagnostics['issues'].append('Audio too quiet')
    
    if diagnostics['silence_ratio'] > 0.5:
        diagnostics['is_good_quality'] = False
        diagnostics['issues'].append('Too much silence')
    
    return diagnostics

def convert_audio_format(input_file: str, 
                        output_format: str = 'wav',
                        sample_rate: int = 44100) -> Optional[str]:
    """
    Convert audio file to specified format.
    
    Args:
        input_file: Path to input audio file
        output_format: Desired output format
        sample_rate: Desired sample rate
        
    Returns:
        Path to converted file or None if conversion fails
    """
    try:
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"converted_{os.path.basename(input_file)}.{output_format}")
        
        command = [
            'ffmpeg',
            '-y',
            '-i', input_file,
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-vn',
            output_file
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        return output_file
        
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def load_and_normalize_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load and normalize audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (normalized_audio, sample_rate)
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        
        # Normalize to [-1, 1] range
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

def save_debug_audio(audio_data: np.ndarray, 
                    sample_rate: int,
                    filename: str):
    """Save audio data for debugging purposes"""
    try:
        sf.write(filename, audio_data, sample_rate)
        print(f"Saved debug audio to {filename}")
    except Exception as e:
        print(f"Error saving debug audio: {e}")

def calculate_audio_features(audio_data: np.ndarray,
                           sample_rate: int) -> dict:
    """
    Calculate basic audio features.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sampling rate
        
    Returns:
        Dictionary of audio features
    """
    features = {}
    
    # Basic features
    features['duration'] = len(audio_data) / sample_rate
    features['rms_energy'] = librosa.feature.rms(y=audio_data)[0]
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_data)[0]
    
    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
    
    features['spectral_centroid'] = np.mean(spec_centroid)
    features['spectral_bandwidth'] = np.mean(spec_bandwidth)
    
    return features