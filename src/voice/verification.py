from typing import Tuple, Dict, Optional
import numpy as np
from difflib import SequenceMatcher
import re

class VoiceVerification:
    def __init__(self):
        self.min_similarity_threshold = 0.3
        self.feature_weights = {
            'pitch': 0.3,
            'mfcc': 0.4,
            'spectral': 0.3,
            'text': 0.0
        }
    
    def verify_voice(self, features1: Dict, features2: Dict, text1: str, text2: str) -> Tuple[bool, float]:
        try:
            mfcc_similarity = self.compare_mfcc(
                features1.get('mfcc', []),
                features2.get('mfcc', [])
            )

            pitch_similarity = self.compare_pitch(
                features1.get('pitch', 0),
                features2.get('pitch', 0)
            )

            spectral_similarity = self.compare_spectral(
                features1.get('frequency_bands', {}),
                features2.get('frequency_bands', {})
            )

            total_similarity = (
                self.feature_weights['mfcc'] * mfcc_similarity +
                self.feature_weights['pitch'] * pitch_similarity +
                self.feature_weights['spectral'] * spectral_similarity
            )

            print(f"\n=== Análisis de Similitud ===")
            print(f"MFCC Similarity: {mfcc_similarity:.3f}")
            print(f"Pitch Similarity: {pitch_similarity:.3f}")
            print(f"Spectral Similarity: {spectral_similarity:.3f}")
            print(f"Total Similarity: {total_similarity:.3f}")

            return total_similarity >= self.min_similarity_threshold, total_similarity

        except Exception as e:
            print(f"Error in voice verification: {str(e)}")
            return False, 0.0

    def compare_spectral(self, bands1: Dict, bands2: Dict) -> float:
        try:
            if not bands1 or not bands2:
                return 0.0

            def normalize_bands(bands):
                total = sum(bands.values())
                if total == 0:
                    return bands
                return {k: v/total for k, v in bands.items()}

            norm_bands1 = normalize_bands(bands1)
            norm_bands2 = normalize_bands(bands2)

            similarities = []
            for band in set(norm_bands1.keys()) | set(norm_bands2.keys()):
                val1 = norm_bands1.get(band, 0)
                val2 = norm_bands2.get(band, 0)
                if max(val1, val2) > 0:
                    similarity = 1 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            print(f"Error in spectral comparison: {str(e)}")
            return 0.0

    def compare_mfcc(self, mfcc1: list, mfcc2: list) -> float:
        try:
            if not mfcc1 or not mfcc2:
                return 0.0

            mfcc1 = np.array(mfcc1)
            mfcc2 = np.array(mfcc2)

            mfcc1 = (mfcc1 - np.mean(mfcc1)) / (np.std(mfcc1) + 1e-10)
            mfcc2 = (mfcc2 - np.mean(mfcc2)) / (np.std(mfcc2) + 1e-10)

            correlation = np.corrcoef(mfcc1, mfcc2)[0,1]
            return max(0, (correlation + 1) / 2)

        except Exception as e:
            print(f"Error in MFCC comparison: {str(e)}")
            return 0.0

    def compare_pitch(self, pitch1: float, pitch2: float) -> float:
        try:
            if pitch1 == 0 or pitch2 == 0:
                return 0.0

            ratio = min(pitch1, pitch2) / max(pitch1, pitch2)
            similarity = np.exp(-np.abs(1 - ratio) * 2)
            
            return similarity

        except Exception as e:
            print(f"Error in pitch comparison: {str(e)}")
            return 0.0

    def calculate_adaptive_threshold(self, features1: Dict, features2: Dict) -> float:
        base_threshold = self.min_similarity_threshold
        quality_factors = []
        
        if features1.get('pitch', 0) > 0 and features2.get('pitch', 0) > 0:
            quality_factors.append(1.0)
        
        if features1.get('mfcc', []) and features2.get('mfcc', []):
            quality_factors.append(1.0)
        
        if features1.get('frequency_bands', {}) and features2.get('frequency_bands', {}):
            quality_factors.append(1.0)
            
        quality_score = np.mean(quality_factors) if quality_factors else 0.5
        adjusted_threshold = base_threshold * quality_score
        
        return max(0.2, min(0.7, adjusted_threshold))

    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text