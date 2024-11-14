"""
Módulo de procesamiento de voz para el sistema de cifrado.
Proporciona funcionalidades para el análisis y verificación de características vocales.
"""

from .processor import VoiceProcessor
from .analyzer import AudioFFTAnalyzer, SpectralFeatures
from .verification import VoiceVerification

__version__ = '1.0.0'

# Configuración del procesamiento de voz
VOICE_CONFIG = {
    'SAMPLE_RATE': 44100,
    'FRAME_SIZE': 2048,
    'HOP_LENGTH': 512,
    'N_MELS': 128,
    'MIN_FREQ': 50,
    'MAX_FREQ': 8000,
    'WINDOW_TYPE': 'hann',
    'N_MFCC': 13,
    'N_FFT': 2048
}

# Configuración de las características de voz
FEATURE_CONFIG = {
    'MIN_DURATION': 0.5,  # segundos
    'MAX_DURATION': 5.0,  # segundos
    'SILENCE_THRESHOLD': 0.01,
    'FEATURE_WEIGHTS': {
        'pitch': 0.3,
        'mfcc': 0.4,
        'spectral': 0.3
    },
    'FORMANT_CONFIG': {
        'NUM_FORMANTS': 4,
        'MAX_FREQ': 5000,
        'WINDOW_LENGTH': 0.025,  # segundos
        'PRE_EMPHASIS': 0.97
    }
}

# Configuración de la verificación de voz
VERIFICATION_CONFIG = {
    'MIN_SIMILARITY_THRESHOLD': 0.3,
    'ADAPTIVE_THRESHOLD': True,
    'QUALITY_WEIGHTS': {
        'snr': 0.3,
        'clarity': 0.3,
        'stability': 0.4
    },
    'COMPARISON_WEIGHTS': {
        'mfcc': 0.4,
        'pitch': 0.3,
        'spectral': 0.3
    }
}

def get_feature_config():
    """Obtener la configuración actual de características de voz."""
    return FEATURE_CONFIG.copy()

def get_verification_config():
    """Obtener la configuración actual de verificación de voz."""
    return VERIFICATION_CONFIG.copy()

def set_feature_config(config):
    """
    Actualizar la configuración de características de voz.
    
    Args:
        config: Diccionario con la nueva configuración
    """
    global FEATURE_CONFIG
    FEATURE_CONFIG.update(config)

def set_verification_config(config):
    """
    Actualizar la configuración de verificación de voz.
    
    Args:
        config: Diccionario con la nueva configuración
    """
    global VERIFICATION_CONFIG
    VERIFICATION_CONFIG.update(config)

__all__ = [
    'VoiceProcessor',
    'AudioFFTAnalyzer',
    'SpectralFeatures',
    'VoiceVerification',
    'VOICE_CONFIG',
    'FEATURE_CONFIG',
    'VERIFICATION_CONFIG',
    'get_feature_config',
    'get_verification_config',
    'set_feature_config',
    'set_verification_config'
]