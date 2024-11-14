import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import json
import wave
from scipy import signal
from scipy.io import wavfile
from vosk import Model, KaldiRecognizer
import hashlib
import os
import json

class VoiceProcessor:
    def __init__(self, sample_rate: int = 44100):
        """
        Inicializar el procesador de voz.
        
        Args:
            sample_rate: Frecuencia de muestreo deseada
        """
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.min_freq = 50
        self.max_freq = 8000
        
    def process_audio(self, audio_data: np.ndarray, original_sr: int = None) -> Dict:
        """
        Procesar archivo de audio y extraer características.
        
        Args:
            audio_data: Datos de audio en numpy array
            original_sr: Sample rate original del audio
            
        Returns:
            Diccionario con características extraídas
        """
        try:
            # Resamplear si es necesario
            if original_sr and original_sr != self.sample_rate:
                audio_data = librosa.resample(
                    y=audio_data,
                    orig_sr=original_sr,
                    target_sr=self.sample_rate
                )
            
            # Normalizar audio
            audio_data = self._normalize_audio(audio_data)
            
            # Extraer características
            features = {
                'waveform': self._get_waveform_features(audio_data),
                'spectral': self._get_spectral_features(audio_data),
                'pitch': self._get_pitch_features(audio_data),
                'mfcc': self._get_mfcc_features(audio_data),
                'energy': self._get_energy_features(audio_data),
                'timing': self._get_timing_features(audio_data)
            }
            
            # Calcular huella de voz
            voice_key = self._generate_voice_key(features)
            
            return {
                'features': features,
                'voice_key': voice_key,
                'quality_metrics': self._calculate_quality_metrics(audio_data)
            }
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalizar audio a rango [-1, 1]"""
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        return audio_data / np.max(np.abs(audio_data))

    def _get_spectral_features(self, audio_data: np.ndarray) -> Dict:
        """Extraer características espectrales"""
        try:
            # Calcular espectrograma
            D = librosa.stft(
                y=audio_data,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            magnitude = np.abs(D)
            
            # Características espectrales
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate
            )
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=self.sample_rate
            )
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=self.sample_rate
            )
            
            # Calcular bandas de frecuencia
            frequency_bands = self._calculate_frequency_bands(magnitude, self.sample_rate)
            
            return {
                'centroid': float(np.mean(spectral_centroid)),
                'bandwidth': float(np.mean(spectral_bandwidth)),
                'rolloff': float(np.mean(spectral_rolloff)),
                'frequency_bands': frequency_bands
            }
            
        except Exception as e:
            print(f"Error en extracción de características espectrales: {str(e)}")
            return {
                'centroid': 0.0,
                'bandwidth': 0.0,
                'rolloff': 0.0,
                'frequency_bands': {}
            }

    def _get_pitch_features(self, audio_data: np.ndarray) -> Dict:
        """Extraer características de tono"""
        try:
            # Extraer pitch usando piptrack
            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                fmin=self.min_freq,
                fmax=self.max_freq
            )
            
            # Obtener pitch fundamental
            pit = []
            for t in range(magnitudes.shape[1]):
                index = magnitudes[:, t].argmax()
                pit.append(pitches[index, t])
            
            pitch_mean = float(np.mean(pit))
            pitch_std = float(np.std(pit))
            
            return {
                'fundamental_frequency': pitch_mean,
                'pitch_variance': pitch_std,
                'pitch_range': float(np.ptp(pit))
            }
            
        except Exception as e:
            print(f"Error en extracción de características de tono: {str(e)}")
            return {
                'fundamental_frequency': 0.0,
                'pitch_variance': 0.0,
                'pitch_range': 0.0
            }

    def _get_mfcc_features(self, audio_data: np.ndarray) -> Dict:
        """Extraer coeficientes MFCC"""
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )
            
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            return {
                'mfcc': list(mfccs),
            'delta_mfcc': list(delta_mfccs),
            'delta2_mfcc': list(delta2_mfccs),
            'mfcc_stats': {
                'mean': float(np.mean(mfccs)),
                'std': float(np.std(mfccs)),
                'skew': float(self._calculate_skewness(mfccs))
                }
            }
            
        except Exception as e:
            print(f"Error en extracción de MFCC: {str(e)}")
            return {
                'mfcc': [],
                'delta_mfcc': [],
                'delta2_mfcc': [],
                'mfcc_stats': {
                    'mean': 0.0,
                    'std': 0.0,
                    'skew': 0.0
                }
            }

    def _get_waveform_features(self, audio_data: np.ndarray) -> Dict:
        """Extraer características de la forma de onda"""
        try:
            return {
                'rms': float(np.sqrt(np.mean(audio_data**2))),
                'peak_amplitude': float(np.max(np.abs(audio_data))),
                'zero_crossings': int(np.sum(np.diff(np.signbit(audio_data)))),
                'crest_factor': float(np.max(np.abs(audio_data)) / np.sqrt(np.mean(audio_data**2)))
            }
        except Exception as e:
            print(f"Error en extracción de características de forma de onda: {str(e)}")
            return {
                'rms': 0.0,
                'peak_amplitude': 0.0,
                'zero_crossings': 0,
                'crest_factor': 0.0
            }

    def _get_energy_features(self, audio_data: np.ndarray) -> Dict:
        """Extraer características de energía"""
        try:
            energy = librosa.feature.rms(
                y=audio_data,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            return {
                'mean_energy': float(np.mean(energy)),
                'energy_std': float(np.std(energy)),
                'energy_range': float(np.ptp(energy))
            }
        except Exception as e:
            print(f"Error en extracción de características de energía: {str(e)}")
            return {
                'mean_energy': 0.0,
                'energy_std': 0.0,
                'energy_range': 0.0
            }

    def _get_timing_features(self, audio_data: np.ndarray) -> Dict:
        """Extraer características temporales"""
        try:
            onset_env = librosa.onset.onset_strength(
                y=audio_data,
                sr=self.sample_rate
            )
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sample_rate
            )
            
            return {
                'duration': float(len(audio_data) / self.sample_rate),
                'tempo': float(tempo),
                'onset_strength_mean': float(np.mean(onset_env))
            }
        except Exception as e:
            print(f"Error en extracción de características temporales: {str(e)}")
            return {
                'duration': 0.0,
                'tempo': 0.0,
                'onset_strength_mean': 0.0
            }

    def _calculate_frequency_bands(self, magnitude: np.ndarray, sr: int) -> Dict[str, float]:
        """Calcular energía en diferentes bandas de frecuencia"""
        try:
            bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'high': (4000, 6000),
                'presence': (6000, 8000),
                'brilliance': (8000, 20000)
            }
            
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            band_energies = {}
            
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    band_energies[band_name] = float(np.mean(np.mean(magnitude[mask], axis=0)))
                else:
                    band_energies[band_name] = 0.0
                    
            return band_energies
            
        except Exception as e:
            print(f"Error en cálculo de bandas de frecuencia: {str(e)}")
            return {band: 0.0 for band in [
                'sub_bass', 'bass', 'low_mid', 'mid', 
                'high_mid', 'high', 'presence', 'brilliance'
            ]}

    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calcular asimetría de una distribución"""
        try:
            mean = np.mean(x)
            std = np.std(x)
            if std == 0:
                return 0.0
            return float(np.mean(((x - mean) / std) ** 3))
        except Exception as e:
            print(f"Error en cálculo de asimetría: {str(e)}")
            return 0.0

    def _calculate_quality_metrics(self, audio_data: np.ndarray) -> Dict:
        """Calcular métricas de calidad del audio"""
        try:
            return {
                'snr': self._calculate_snr(audio_data),
                'clarity': self._calculate_clarity(audio_data),
                'silence_ratio': self._calculate_silence_ratio(audio_data)
            }
        except Exception as e:
            print(f"Error en cálculo de métricas de calidad: {str(e)}")
            return {
                'snr': 0.0,
                'clarity': 0.0,
                'silence_ratio': 0.0
            }

    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calcular relación señal-ruido"""
        try:
            noise_floor = np.sort(np.abs(audio_data))[int(len(audio_data)*0.1)]
            signal = np.mean(np.abs(audio_data))
            if noise_floor == 0:
                return float('inf')
            return float(20 * np.log10(signal / noise_floor))
        except Exception as e:
            print(f"Error en cálculo de SNR: {str(e)}")
            return 0.0

    def _calculate_clarity(self, audio_data: np.ndarray) -> float:
        """Calcular claridad de la señal"""
        try:
            envelope = np.abs(signal.hilbert(audio_data))
            return float(np.mean(envelope) / np.std(envelope))
        except Exception as e:
            print(f"Error en cálculo de claridad: {str(e)}")
            return 0.0

    def _calculate_silence_ratio(self, audio_data: np.ndarray) -> float:
        """Calcular ratio de silencio"""
        try:
            threshold = 0.01
            silence_samples = np.sum(np.abs(audio_data) < threshold)
            return float(silence_samples / len(audio_data))
        except Exception as e:
            print(f"Error en cálculo de ratio de silencio: {str(e)}")
            return 0.0

    def _generate_voice_key(self, features: Dict) -> str:
        """Generar clave única basada en características de voz"""
        try:
            # Seleccionar características discriminativas
            key_components = []
            
            # Añadir características espectrales
            spec = features['spectral']
            key_components.extend([
                f"c{spec['centroid']:.2f}",
                f"b{spec['bandwidth']:.2f}",
                f"r{spec['rolloff']:.2f}"
            ])
            
            # Añadir características de pitch
            pitch = features['pitch']
            key_components.extend([
                f"f{pitch['fundamental_frequency']:.2f}",
                f"v{pitch['pitch_variance']:.2f}"
            ])
            
            # Añadir estadísticas MFCC
            if 'mfcc' in features and isinstance(features['mfcc'], dict):
                mfcc_stats = features['mfcc'].get('mfcc_stats', {})
                key_components.extend([
                    f"m{mfcc_stats.get('mean', 0):.2f}",
                    f"s{mfcc_stats.get('std', 0):.2f}"
                ])
            
            # Generar hash final
            key_string = ":".join(key_components)
            return hashlib.sha256(key_string.encode()).hexdigest()
        
        except Exception as e:
            print(f"Error en generación de clave de voz: {str(e)}")
            # Generar una clave aleatoria en caso de error
            return hashlib.sha256(os.urandom(32)).hexdigest()

    def save_debug_audio(self, audio_data: np.ndarray, filename: str):
        """Guardar audio para debugging"""
        try:
            sf.write(
                filename,
                audio_data,
                self.sample_rate,
                subtype='PCM_16'
            )
            print(f"Debug audio saved to {filename}")
        except Exception as e:
            print(f"Error saving debug audio: {e}")

    def verify_audio_quality(self, audio_data: np.ndarray) -> Tuple[bool, Dict]:
        """
        Verificar la calidad del audio.
        
        Args:
            audio_data: Audio a verificar
            
        Returns:
            Tupla de (is_good_quality, diagnostics)
        """
        try:
            diagnostics = {
                'duration': len(audio_data) / self.sample_rate,
                'max_amplitude': np.max(np.abs(audio_data)),
                'mean_amplitude': np.mean(np.abs(audio_data)),
                'silence_ratio': np.sum(np.abs(audio_data) < 0.01) / len(audio_data),
                'is_good_quality': True,
                'issues': []
            }
            
            # Verificar duración
            if diagnostics['duration'] < 0.5:
                diagnostics['is_good_quality'] = False
                diagnostics['issues'].append('Audio demasiado corto')
            
            # Verificar amplitud
            if diagnostics['max_amplitude'] < 0.01:
                diagnostics['is_good_quality'] = False
                diagnostics['issues'].append('Volumen muy bajo')
            
            # Verificar ratio de silencio
            if diagnostics['silence_ratio'] > 0.7:
                diagnostics['is_good_quality'] = False
                diagnostics['issues'].append('Demasiado silencio')
                
            # Verificar saturación
            if diagnostics['max_amplitude'] > 0.95:
                diagnostics['is_good_quality'] = False
                diagnostics['issues'].append('Audio saturado')

            return diagnostics['is_good_quality'], diagnostics
            
        except Exception as e:
            print(f"Error en verificación de calidad: {str(e)}")
            return False, {
                'is_good_quality': False,
                'issues': ['Error en análisis de calidad'],
                'error': str(e)
            }

    def get_audio_info(self, audio_data: np.ndarray) -> Dict:
        """
        Obtener información general del audio.
        
        Args:
            audio_data: Datos de audio
            
        Returns:
            Dict con información del audio
        """
        try:
            return {
                'duration': len(audio_data) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                'num_samples': len(audio_data),
                'dtype': str(audio_data.dtype),
                'mean': float(np.mean(audio_data)),
                'std': float(np.std(audio_data)),
                'max': float(np.max(np.abs(audio_data))),
                'min': float(np.min(np.abs(audio_data)))
            }
        except Exception as e:
            print(f"Error obteniendo información del audio: {str(e)}")
            return {}