"""
Módulo de criptografía para el sistema de cifrado por voz.
Proporciona funcionalidades para el cifrado y descifrado seguro de archivos.
"""

from .keys import VoiceKeyGenerator
from .cipher import (
    create_encryption_key,
    encrypt_file,
    decrypt_file,
    verify_key_format,
    generate_random_key
)

__version__ = '1.0.0'

# Constantes de configuración criptográfica
CRYPTO_CONFIG = {
    'KEY_SIZE': 32,  # 256 bits
    'SALT_SIZE': 16,
    'IV_SIZE': 16,
    'PBKDF2_ITERATIONS': 200000,
    'HASH_ALGORITHM': 'SHA256',
    'CIPHER_MODE': 'GCM'
}

__all__ = [
    'VoiceKeyGenerator',
    'create_encryption_key',
    'encrypt_file',
    'decrypt_file',
    'verify_key_format',
    'generate_random_key',
    'CRYPTO_CONFIG'
]