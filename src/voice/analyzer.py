import os
from flask import json
import numpy as np
from scipy.fft import fft, fftfreq
import librosa
import scipy.signal as signal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import hashlib
import json

@dataclass
class SpectralFeatures:
    frequencies: List[float]
    magnitudes: List[float]
    phase: List[float]
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_contrast: List[List[float]]
    mel_spectrogram: List[List[float]]
    formants: List[float]

class AudioFFTAnalyzer:
    def __init__(self, 
                 sample_rate: int = 44100,
                 frame_size: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Inicializar el analizador FFT de audio.
        
        Args:
            sample_rate: Frecuencia de muestreo del audio
            frame_size: Tamaño de la ventana para FFT
            hop_length: Tamaño del salto entre ventanas
            n_mels: Número de bandas mel para el espectrograma
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Ventana de Hann para reducir el efecto de fuga espectral
        self.window = signal.windows.hann(frame_size)
        
    def analyze_signal(self, audio_data: np.ndarray) -> SpectralFeatures:
        """
        Realiza un análisis espectral completo de la señal de audio.
        
        Args:
            audio_data: Array numpy con los datos de audio normalizados
            
        Returns:
            SpectralFeatures con todas las características espectrales
        """
        try:
            # Asegurarse de que el audio está normalizado
            audio_data = self._normalize_audio(audio_data)
            
            # Calcular FFT
            fft_data = self._compute_fft(audio_data)
            frequencies = fftfreq(self.frame_size, 1/self.sample_rate)
            magnitudes = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # Convertir arrays a listas para serialización JSON
            frequencies_list = list(frequencies[:len(frequencies)//2])
            magnitudes_list = list(magnitudes[:len(magnitudes)//2])
            phase_list = list(phase[:len(phase)//2])
            
            # Calcular características espectrales
            spectral_centroid = float(self._compute_spectral_centroid(magnitudes, frequencies))
            spectral_bandwidth = float(self._compute_spectral_bandwidth(magnitudes, frequencies, spectral_centroid))
            spectral_rolloff = float(self._compute_spectral_rolloff(magnitudes, frequencies))
            
            # Convertir contraste espectral a lista
            spectral_contrast = [list(row) for row in self._compute_spectral_contrast(audio_data)]
            # Calcular mel espectrograma
            mel_spec = [list(row) for row in self._compute_mel_spectrogram(audio_data)]
            
            # Detectar formantes
            formants = self._detect_formants(audio_data)
            
            return SpectralFeatures(
                frequencies=frequencies_list,
                magnitudes=magnitudes_list,
                phase=phase_list,
                spectral_centroid=spectral_centroid,
                spectral_bandwidth=spectral_bandwidth,
                spectral_rolloff=spectral_rolloff,
                spectral_contrast=spectral_contrast,
                mel_spectrogram=mel_spec,
                formants=formants
        )
            
        except Exception as e:
            print(f"Error in analyze_signal: {str(e)}")
            raise
    
    def compare_signals(self, features1: SpectralFeatures, features2: SpectralFeatures) -> Dict[str, float]:
        """
        Compara dos conjuntos de características espectrales.
        
        Args:
            features1: Primer conjunto de características
            features2: Segundo conjunto de características
            
        Returns:
            Diccionario con diferentes métricas de similitud
        """
        try:
            similarities = {}
            
            # Comparar magnitudes espectrales
            similarities['magnitude_correlation'] = np.corrcoef(
                features1.magnitudes, 
                features2.magnitudes
            )[0,1]
            
            # Comparar centroides espectrales
            centroid_diff = abs(features1.spectral_centroid - features2.spectral_centroid)
            similarities['centroid_similarity'] = 1.0 / (1.0 + centroid_diff)
            
            # Comparar formantes
            formant_similarities = []
            for f1, f2 in zip(features1.formants, features2.formants):
                diff = abs(f1 - f2)
                formant_similarities.append(1.0 / (1.0 + diff))
            similarities['formant_similarity'] = np.mean(formant_similarities)
            
            # Comparar contraste espectral
            similarities['contrast_correlation'] = np.corrcoef(
                features1.spectral_contrast.flatten(), 
                features2.spectral_contrast.flatten()
            )[0,1]
            
            # Calcular similitud global
            similarities['global_similarity'] = np.mean([
                similarities['magnitude_correlation'],
                similarities['centroid_similarity'],
                similarities['formant_similarity'],
                similarities['contrast_correlation']
            ])
            
            return similarities
            
        except Exception as e:
            print(f"Error comparing signals: {str(e)}")
            return {
                'magnitude_correlation': 0.0,
                'centroid_similarity': 0.0,
                'formant_similarity': 0.0,
                'contrast_correlation': 0.0,
                'global_similarity': 0.0
            }

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normaliza los datos de audio al rango [-1, 1]"""
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
            
        return audio_data / np.max(np.abs(audio_data))
    
    def _compute_fft(self, audio_data: np.ndarray) -> np.ndarray:
        """Calcular la FFT usando ventana de Hann"""
        windowed_data = audio_data[:self.frame_size] * self.window
        return fft(windowed_data)
    
    def _compute_spectral_centroid(self, magnitudes: np.ndarray, frequencies: np.ndarray) -> float:
        """Calcular el centroide espectral"""
        return np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    
    def _compute_spectral_bandwidth(self, magnitudes: np.ndarray, frequencies: np.ndarray, centroid: float) -> float:
        """Calcular el ancho de banda espectral"""
        return np.sqrt(np.sum(((frequencies - centroid) ** 2) * magnitudes) / np.sum(magnitudes))
    
    def _compute_spectral_rolloff(self, magnitudes: np.ndarray, frequencies: np.ndarray, percentile: float = 0.85) -> float:
        """Calcular la frecuencia de rolloff espectral"""
        threshold = np.sum(magnitudes) * percentile
        cumsum = np.cumsum(magnitudes)
        rolloff_point = np.where(cumsum >= threshold)[0][0]
        return frequencies[rolloff_point]
    
    def _compute_spectral_contrast(self, audio_data: np.ndarray) -> np.ndarray:
        """Calcular el contraste espectral"""
        return librosa.feature.spectral_contrast(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_length
        )
    
    def _compute_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """Calcular el espectrograma mel"""
        return librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
    
    def _detect_formants(self, audio_data: np.ndarray) -> List[float]:
        """
        Detecta las frecuencias formantes usando análisis LPC.
        Los formantes son importantes para identificar vocales y características de voz.
        """
        try:
            # Parámetros para el análisis de formantes
            frame_length = int(0.025 * self.sample_rate)  # 25ms ventana
            hop_length = int(0.010 * self.sample_rate)    # 10ms salto
            n_formants = 4                                # Número de formantes a detectar
            pre_emphasis_coeff = 0.97                     # Coeficiente de pre-énfasis
            
            # Aplicar pre-énfasis para acentuar altas frecuencias
            pre_emphasis = np.append(audio_data[0], audio_data[1:] - pre_emphasis_coeff * audio_data[:-1])
            
            # Enventanar la señal
            frames = librosa.util.frame(pre_emphasis, 
                                      frame_length=frame_length, 
                                      hop_length=hop_length)
            
            # Aplicar ventana de Hamming
            window = np.hamming(frame_length)
            frames = frames.T * window
            
            # Almacenar formantes detectados
            all_formants = []
            
            # Procesar cada frame
            for frame in frames:
                try:
                    # Orden del análisis LPC
                    n_coeff = 2 + n_formants * 2
                    
                    # Calcular coeficientes LPC
                    lpc_coeffs = librosa.lpc(frame, order=n_coeff)
                    
                    # Obtener raíces del polinomio LPC
                    roots = np.roots(lpc_coeffs)
                    
                    # Filtrar raíces
                    roots = roots[np.imag(roots) >= 0]  # Solo tomar la mitad superior
                    angles = np.angle(roots)            # Obtener ángulos
                    
                    # Convertir a frecuencias
                    frequencies = angles * (self.sample_rate / (2 * np.pi))
                    
                    # Filtrar frecuencias válidas
                    frequencies = frequencies[
                        (frequencies > 50) &    # Eliminar frecuencias muy bajas
                        (frequencies < 5000) &  # Eliminar frecuencias muy altas
                        (np.abs(np.imag(roots)/np.abs(roots)) < 0.3)  # Filtrar por ancho de banda
                    ]
                    
                    if len(frequencies) > 0:
                        all_formants.append(frequencies)
                
                except Exception as frame_error:
                    print(f"Error procesando frame: {frame_error}")
                    continue
            
            if not all_formants:
                return [0.0] * n_formants
            
            max_len = max(len(x) for x in all_formants)

            # Rellenar con ceros los arrays más cortos
            padded_formants = [np.pad(x, (0, max_len - len(x)), 'constant') for x in all_formants]

            # Convertir a array numpy 2D
            all_formants_np = np.array(padded_formants)

            # Promediar formantes de todos los frames
            mean_formants = np.mean(all_formants_np, axis=0)
            
            # Ordenar formantes por frecuencia
            sorted_formants = np.sort(mean_formants)
            
            # Asegurar que tenemos el número correcto de formantes
            if len(sorted_formants) < n_formants:
                return list(sorted_formants) + [0.0] * (n_formants - len(sorted_formants))
            else:
                return list(sorted_formants[:n_formants])
                
        except Exception as e:
            print(f"Error en detect_formants: {str(e)}")
            return [0.0] * n_formants

    def get_voice_signature(self, features: SpectralFeatures) -> str:
        """
        Genera una firma única basada en las características espectrales.
        Útil para comparación y verificación de voz.
        """
        try:
            # Crear vector de características
            feature_vector = np.concatenate([
                features.magnitudes[:100],  # Primeras 100 magnitudes
                [features.spectral_centroid],
                [features.spectral_bandwidth],
                [features.spectral_rolloff],
                features.formants,
                features.spectral_contrast.flatten()[:50]  # Primeros 50 valores de contraste
            ])
            
            # Normalizar vector
            if np.std(feature_vector) != 0:
                feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
            
            # Generar hash del vector
            return hashlib.sha256(feature_vector.tobytes()).hexdigest()
            
        except Exception as e:
            print(f"Error generating voice signature: {str(e)}")
            return hashlib.sha256(os.urandom(32)).hexdigest()

    def save_debug_info(self, features: SpectralFeatures, filename: str):
        """
        Guarda información de debug de las características espectrales.
        
        Args:
            features: Características espectrales
            filename: Nombre del archivo para guardar la información
        """
        try:
            debug_info = {
                'spectral_centroid': float(features.spectral_centroid),
                'spectral_bandwidth': float(features.spectral_bandwidth),
                'spectral_rolloff': float(features.spectral_rolloff),
                'formants': [float(f) for f in features.formants],
                'magnitudes_stats': {
                    'mean': float(np.mean(features.magnitudes)),
                    'std': float(np.std(features.magnitudes)),
                    'max': float(np.max(features.magnitudes)),
                    'min': float(np.min(features.magnitudes))
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(debug_info, f, indent=2)
                
        except Exception as e:
            print(f"Error saving debug info: {str(e)}")