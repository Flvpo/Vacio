from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256, HMAC
from Crypto.Random import get_random_bytes
from typing import List, Tuple, Dict, Optional
import base64
import json
import time
import os

class VoiceKeyGenerator:
    def __init__(self, 
                 iterations: int = 200000, 
                 key_length: int = 32,
                 salt_length: int = 16):
        """
        Inicializar generador de claves basadas en voz.
        
        Args:
            iterations: Número de iteraciones para PBKDF2
            key_length: Longitud de la clave en bytes
            salt_length: Longitud del salt en bytes
        """
        self.iterations = iterations
        self.key_length = key_length
        self.salt_length = salt_length
        self.version = "1.0"
        
    def generate_key(self, 
                    voice_features: Dict,
                    salt: Optional[bytes] = None) -> Tuple[bytes, bytes, Dict]:
        """
        Generar clave criptográfica a partir de características de voz.
        
        Args:
            voice_features: Diccionario con características de voz
            salt: Salt opcional (se genera uno nuevo si no se proporciona)
            
        Returns:
            Tupla (key, salt, metadata)
        """
        try:
            # Normalizar características de voz
            normalized_features = self._normalize_features(voice_features)
            
            # Generar cadena de características
            feature_string = self._features_to_string(normalized_features)
            
            # Generar o usar salt proporcionado
            if salt is None:
                salt = self._generate_salt()
            
            # Generar clave usando PBKDF2
            key = PBKDF2(
                password=feature_string.encode(),
                salt=salt,
                dkLen=self.key_length,
                count=self.iterations,
                hmac_hash_module=SHA256
            )
            
            # Generar metadata
            metadata = self._generate_metadata(normalized_features)
            
            return key, salt, metadata
            
        except Exception as e:
            print(f"Error generating key: {e}")
            raise

    def verify_key(self, 
                   stored_metadata: Dict,
                   current_features: Dict,
                   tolerance: float = 0.2) -> bool:
        """
        Verificar si las características actuales coinciden con la metadata almacenada.
        
        Args:
            stored_metadata: Metadata de la clave almacenada
            current_features: Características de voz actuales
            tolerance: Tolerancia permitida en la comparación
            
        Returns:
            Boolean indicando si la verificación fue exitosa
        """
        try:
            # Normalizar características actuales
            normalized_current = self._normalize_features(current_features)
            
            # Extraer características de referencia de la metadata
            reference_features = stored_metadata.get('reference_features', {})
            
            # Comparar características principales
            for feature_name, ref_value in reference_features.items():
                if feature_name not in normalized_current:
                    return False
                
                current_value = normalized_current[feature_name]
                if not self._compare_feature(current_value, ref_value, tolerance):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying key: {e}")
            return False

    def _normalize_features(self, features: Dict) -> Dict:
        """Normalizar características de voz para consistencia"""
        normalized = {}
        
        # Normalizar características espectrales
        if 'spectral' in features:
            spec = features['spectral']
            normalized['spectral_centroid'] = float(spec.get('centroid', 0))
            normalized['spectral_bandwidth'] = float(spec.get('bandwidth', 0))
            normalized['spectral_rolloff'] = float(spec.get('rolloff', 0))
            
            # Normalizar bandas de frecuencia
            bands = spec.get('frequency_bands', {})
            total_energy = sum(bands.values()) if bands else 1
            normalized['frequency_bands'] = {
                k: v/total_energy for k, v in bands.items()
            }
        
        # Normalizar características de pitch
        if 'pitch' in features:
            pitch = features['pitch']
            normalized['fundamental_frequency'] = float(pitch.get('fundamental_frequency', 0))
            normalized['pitch_variance'] = float(pitch.get('pitch_variance', 0))
        
        # Normalizar MFCC
        if 'mfcc' in features:
            mfcc = features['mfcc']
            if isinstance(mfcc, dict) and 'mfcc_stats' in mfcc:
                normalized['mfcc_mean'] = float(mfcc['mfcc_stats'].get('mean', 0))
                normalized['mfcc_std'] = float(mfcc['mfcc_stats'].get('std', 0))
        
        return normalized

    def _features_to_string(self, features: Dict) -> str:
        """Convertir características normalizadas a string para generación de clave"""
        # Ordenar características para consistencia
        sorted_features = sorted(
            features.items(),
            key=lambda x: x[0]
        )
        
        # Convertir a string con precisión fija
        feature_strings = []
        for key, value in sorted_features:
            if isinstance(value, dict):
                # Manejar sub-diccionarios (e.g., bandas de frecuencia)
                sub_features = sorted(
                    value.items(),
                    key=lambda x: x[0]
                )
                sub_strings = [f"{k}:{v:.6f}" for k, v in sub_features]
                feature_strings.append(f"{key}:{','.join(sub_strings)}")
            else:
                # Manejar valores simples
                feature_strings.append(f"{key}:{value:.6f}")
        
        return "|".join(feature_strings)

    def _generate_salt(self) -> bytes:
        """Generar salt aleatorio"""
        return get_random_bytes(self.salt_length)

    def _generate_metadata(self, features: Dict) -> Dict:
        """Generar metadata para la clave"""
        return {
            'version': self.version,
            'timestamp': int(time.time()),
            'iterations': self.iterations,
            'key_length': self.key_length,
            'reference_features': features,
            'feature_ranges': self._calculate_feature_ranges(features)
        }

    def _calculate_feature_ranges(self, features: Dict) -> Dict:
        """Calcular rangos aceptables para cada característica"""
        ranges = {}
        
        for feature_name, value in features.items():
            if isinstance(value, dict):
                # Manejar sub-características
                ranges[feature_name] = {
                    k: {'min': v * 0.8, 'max': v * 1.2}
                    for k, v in value.items()
                }
            else:
                # Manejar valores simples
                ranges[feature_name] = {
                    'min': value * 0.8,
                    'max': value * 1.2
                }
                
        return ranges

    def _compare_feature(self, 
                        current_value: float, 
                        reference_value: float, 
                        tolerance: float) -> bool:
        """Comparar una característica individual con tolerancia"""
        if isinstance(current_value, dict) and isinstance(reference_value, dict):
            # Comparar diccionarios de características
            return all(
                self._compare_feature(current_value.get(k, 0),
                                    v,
                                    tolerance)
                for k, v in reference_value.items()
            )
        
        # Comparar valores numéricos
        if isinstance(current_value, (int, float)) and isinstance(reference_value, (int, float)):
            diff = abs(current_value - reference_value)
            max_val = max(abs(current_value), abs(reference_value))
            if max_val == 0:
                return diff < tolerance
            return (diff / max_val) <= tolerance
            
        return False

    def export_key_data(self, 
                       key: bytes, 
                       salt: bytes, 
                       metadata: Dict,
                       output_path: str):
        """
        Exportar datos de la clave a un archivo.
        
        Args:
            key: Clave generada
            salt: Salt usado
            metadata: Metadata asociada
            output_path: Ruta donde guardar el archivo
        """
        try:
            key_data = {
                'key': base64.b64encode(key).decode('utf-8'),
                'salt': base64.b64encode(salt).decode('utf-8'),
                'metadata': metadata
            }
            
            with open(output_path, 'w') as f:
                json.dump(key_data, f, indent=2)
                
        except Exception as e:
            print(f"Error exporting key data: {e}")
            raise

    def import_key_data(self, input_path: str) -> Tuple[bytes, bytes, Dict]:
        """
        Importar datos de clave desde archivo.
        
        Args:
            input_path: Ruta del archivo
            
        Returns:
            Tupla (key, salt, metadata)
        """
        try:
            with open(input_path, 'r') as f:
                key_data = json.load(f)
                
            key = base64.b64decode(key_data['key'])
            salt = base64.b64decode(key_data['salt'])
            metadata = key_data['metadata']
            
            return key, salt, metadata
            
        except Exception as e:
            print(f"Error importing key data: {e}")
            raise

    def derive_subkeys(self, 
                      master_key: bytes, 
                      num_keys: int = 2, 
                      context: str = "") -> List[bytes]:
        """
        Derivar subclaves de una clave maestra.
        
        Args:
            master_key: Clave maestra
            num_keys: Número de subclaves a generar
            context: Contexto adicional para la derivación
            
        Returns:
            Lista de subclaves
        """
        subkeys = []
        
        for i in range(num_keys):
            # Crear contexto único para cada subclave
            key_context = f"{context}|key{i}"
            
            # Usar HMAC para derivar subclave
            h = HMAC.new(master_key, digestmod=SHA256)
            h.update(key_context.encode())
            
            subkey = h.digest()
            subkeys.append(subkey)
            
        return subkeys