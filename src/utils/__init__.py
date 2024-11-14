"""
Utilidades generales para el sistema de cifrado por voz.
Este módulo proporciona funciones auxiliares y herramientas comunes.
"""

from .audio import (
    verify_audio_quality,
    convert_audio_format,
    load_and_normalize_audio,
    save_debug_audio,
    calculate_audio_features
)

__all__ = [
    'verify_audio_quality',
    'convert_audio_format',
    'load_and_normalize_audio',
    'save_debug_audio',
    'calculate_audio_features'
]