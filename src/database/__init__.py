"""
Módulo de base de datos para el sistema de cifrado por voz.
Proporciona funcionalidades para el almacenamiento y recuperación de archivos cifrados.
"""

from .models import Database

__version__ = '1.0.0'

# Constantes de configuración de la base de datos
DEFAULT_DB_CONFIG = {
    'PAGE_SIZE': 4096,
    'CACHE_SIZE': 2000,
    'JOURNAL_MODE': 'WAL',
    'FOREIGN_KEYS': True,
    'AUTO_VACUUM': 'INCREMENTAL'
}

__all__ = [
    'Database',
    'DEFAULT_DB_CONFIG'
]