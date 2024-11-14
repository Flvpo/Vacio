"""
Sistema de Cifrado por Voz
--------------------------
Un sistema seguro para cifrar y descifrar archivos usando características de voz como clave.

Este paquete proporciona una implementación completa del sistema, incluyendo:
- Procesamiento y análisis de voz
- Generación de claves basadas en características vocales
- Cifrado/descifrado de archivos
- Interfaz web para interacción del usuario
"""

import os
from typing import Dict

# Versión del sistema
__version__ = '1.0.0'

# Configuración por defecto
DEFAULT_CONFIG: Dict = {
    'SAMPLE_RATE': 44100,
    'FRAME_SIZE': 2048,
    'HOP_LENGTH': 512,
    'MIN_FREQUENCY': 50,
    'MAX_FREQUENCY': 8000,
    'KEY_ITERATIONS': 200000,
    'KEY_LENGTH': 32,
    'SALT_LENGTH': 16,
    'MIN_SIMILARITY_THRESHOLD': 0.3,
    'VOICE_FEATURE_WEIGHTS': {
        'pitch': 0.3,
        'mfcc': 0.4,
        'spectral': 0.3,
        'text': 0.0
    }
}

# Rutas importantes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATABASE_PATH = os.path.join(BASE_DIR, 'voice_encrypted_files.db')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'vosk-model-small-es-0.42')

# Asegurar que existen los directorios necesarios
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Importaciones para facilitar el acceso a componentes principales
from .voice.processor import VoiceProcessor
from .voice.analyzer import AudioFFTAnalyzer
from .voice.verification import VoiceVerification
from .crypto.keys import VoiceKeyGenerator
from .crypto.cipher import (
    create_encryption_key,
    encrypt_file,
    decrypt_file
)
from .database.models import Database
from .utils.audio import (
    verify_audio_quality,
    convert_audio_format,
    load_and_normalize_audio
)

# Funciones de utilidad para el sistema
def initialize_system():
    """Inicializar todos los componentes del sistema"""
    # Verificar directorios y permisos
    required_dirs = [UPLOAD_FOLDER]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif not os.access(directory, os.W_OK):
            raise PermissionError(f"No hay permisos de escritura en {directory}")

    # Verificar modelo Vosk
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo Vosk no encontrado en {VOSK_MODEL_PATH}. "
            "Por favor, descarga el modelo desde https://alphacephei.com/vosk/models"
        )

    # Inicializar base de datos
    db = Database(DATABASE_PATH)
    db.init_db()

def get_version() -> str:
    """Obtener la versión actual del sistema"""
    return __version__

def get_system_info() -> Dict:
    """Obtener información del sistema"""
    return {
        'version': __version__,
        'base_dir': BASE_DIR,
        'upload_folder': UPLOAD_FOLDER,
        'database_path': DATABASE_PATH,
        'vosk_model_path': VOSK_MODEL_PATH,
        'config': DEFAULT_CONFIG
    }

# Definir qué se exporta
__all__ = [
    'VoiceProcessor',
    'AudioFFTAnalyzer',
    'VoiceVerification',
    'VoiceKeyGenerator',
    'create_encryption_key',
    'encrypt_file',
    'decrypt_file',
    'Database',
    'verify_audio_quality',
    'convert_audio_format',
    'load_and_normalize_audio',
    'initialize_system',
    'get_version',
    'get_system_info',
    'DEFAULT_CONFIG'
]