import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import iirfilter, lfilter

class SpectralAnalyzer:
    """Analyze audio to provide objective data for EQ decisions."""
    
    MIN_SIGNAL_LENGTH = 2048  # Minimum length for STFT with n_fft=2048
    
    # Granular Band Definitions for precise control
    GRANULAR_BANDS = [
        (20, 60),      # Sub
        (60, 150),     # Bass
        (150, 300),    # Mud/Low-Mid
        (300, 600),    # Boxiness/Mid
        (600, 1200),   # Body/Mid
        (1200, 2500),  # Definition
        (2500, 4500),  # Harshness/Presence
        (4500, 8000),  # Detail
        (8000, 12000), # Air/Shimmer
        (12000, 20000) # Ultra-high
    ]

    @staticmethod
    def _to_mono(y):
        """Safely convert stereo to mono for spectral analysis."""
        if y.ndim > 1:
            return np.mean(y, axis=1)
        return y

    @staticmethod
    def get_fundamental_frequency(y, sr):
        """Find the dominant frequency in the low-end (Kick fundamental)."""
        y_mono = SpectralAnalyzer._to_mono(y)
        if len(y_mono) < SpectralAnalyzer.MIN_SIGNAL_LENGTH:
            return 60.0  # Default
        # Focus on low-end
        S = np.abs(librosa.stft(y_mono))
        freqs = librosa.fft_frequencies(sr=sr)
        
        low_idx = (freqs >= 30) & (freqs <= 150)
        low_energy = np.mean(S[low_idx, :], axis=1)
        
        if len(low_energy) == 0:
            return 60.0 # Default
            
        fundamental = freqs[low_idx][np.argmax(low_energy)]
        return fundamental

    @staticmethod
    def get_spectral_profile(y, sr, bands=None):
        """Get average energy across specified frequency bands."""
        y_mono = SpectralAnalyzer._to_mono(y)
        if len(y_mono) < SpectralAnalyzer.MIN_SIGNAL_LENGTH:
            return {b: 0 for b in (bands or [])}
        if bands is None:
            bands = SpectralAnalyzer.GRANULAR_BANDS
        
        S = np.abs(librosa.stft(y_mono))
        freqs = librosa.fft_frequencies(sr=sr)
        
        profile = {}
        total_energy = np.mean(S)
        if total_energy == 0:
            return {b: 0 for b in bands}

        for low, high in bands:
            idx = (freqs >= low) & (freqs <= high)
            if np.any(idx):
                energy = np.mean(S[idx, :])
                profile[(low, high)] = energy / total_energy
            else:
                profile[(low, high)] = 0
        
        return profile

    @staticmethod
    def detect_buildup(y, sr, freq_range=(200, 350)):
        """Detect if a frequency range has excessive energy (Mud/Harshness)."""
        y_mono = SpectralAnalyzer._to_mono(y)
        if len(y_mono) < SpectralAnalyzer.MIN_SIGNAL_LENGTH:
            return 0
        S = np.abs(librosa.stft(y_mono))
        freqs = librosa.fft_frequencies(sr=sr)
        
        target_idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        total_energy = np.mean(S)
        if total_energy == 0: return 0
        range_energy = np.mean(S[target_idx, :])
        
        return range_energy / total_energy

    @staticmethod
    def get_crest_factor(y):
        """Calculate the ratio of peak to RMS (in dB)."""
        peak = np.max(np.abs(y))
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-6: return 0.0
        return 20 * np.log10(peak / rms)

class AlgorithmicEQ:
    """Apply surgical and creative EQ based on objective analysis."""
    
    # Target Profiles updated for Granular Bands (10-band model)
    TARGET_PROFILES = {
        'kick':   [0.25, 0.35, 0.08, 0.04, 0.04, 0.04, 0.10, 0.05, 0.03, 0.02],
        'snare':  [0.05, 0.10, 0.20, 0.15, 0.15, 0.15, 0.10, 0.05, 0.03, 0.02],
        'hat':    [0.00, 0.00, 0.02, 0.05, 0.10, 0.15, 0.25, 0.25, 0.10, 0.08],
        'bass':   [0.30, 0.40, 0.15, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01, 0.00],
        'melody': [0.00, 0.05, 0.10, 0.15, 0.25, 0.20, 0.15, 0.05, 0.03, 0.02],
        'pad':    [0.00, 0.10, 0.20, 0.25, 0.15, 0.10, 0.10, 0.05, 0.03, 0.02],
        'master': [0.15, 0.20, 0.15, 0.12, 0.12, 0.10, 0.08, 0.04, 0.02, 0.02],
        'default':[0.10] * 10
    }

    def __init__(self, sr=48000):
        self.sr = sr
        self.analyzer = SpectralAnalyzer()

    def peaking_filter(self, y, center_freq, gain_db, Q=1.0):
        """Bell/Notch filter implementation."""
        if gain_db == 0: return y
        A = 10**(gain_db/40)
        omega = 2 * np.pi * center_freq / self.sr
        alpha = np.sin(omega) / (2 * Q)
        
        b = [1 + alpha*A, -2*np.cos(omega), 1 - alpha*A]
        a = [1 + alpha/A, -2*np.cos(omega), 1 - alpha/A]
        
        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)

    def high_shelf(self, y, shelf_freq, gain_db, Q=0.707):
        """High-shelf for shimmer and air."""
        if gain_db == 0: return y
        A = 10**(gain_db/40)
        omega = 2 * np.pi * shelf_freq / self.sr
        alpha = np.sin(omega) / (2 * Q)
        
        b = [A*((A+1) + (A-1)*np.cos(omega) + 2*np.sqrt(A)*alpha),
             -2*A*((A-1) + (A+1)*np.cos(omega)),
             A*((A+1) + (A-1)*np.cos(omega) - 2*np.sqrt(A)*alpha)]
        
        a = [(A+1) - (A-1)*np.cos(omega) + 2*np.sqrt(A)*alpha,
             2*((A-1) - (A+1)*np.cos(omega)),
             (A+1) - (A-1)*np.cos(omega) - 2*np.sqrt(A)*alpha]
        
        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)

    def low_shelf(self, y, shelf_freq, gain_db, Q=0.707):
        """Low-shelf for weight."""
        if gain_db == 0: return y
        A = 10**(gain_db/40)
        omega = 2 * np.pi * shelf_freq / self.sr
        alpha = np.sin(omega) / (2 * Q)
        
        b = [A*((A+1) - (A-1)*np.cos(omega) + 2*np.sqrt(A)*alpha),
             2*A*((A-1) - (A+1)*np.cos(omega)),
             A*((A+1) - (A-1)*np.cos(omega) - 2*np.sqrt(A)*alpha)]
        
        a = [(A+1) + (A-1)*np.cos(omega) + 2*np.sqrt(A)*alpha,
             -2*((A-1) + (A+1)*np.cos(omega)),
             (A+1) + (A-1)*np.cos(omega) - 2*np.sqrt(A)*alpha]
        
        orig_dtype = y.dtype
        y_f64 = y.astype(np.float64)
        return lfilter(b, a, y_f64, axis=0).astype(orig_dtype)

    def optimize_to_target(self, y, sr, track_type, max_passes=10):
        """
        Iteratively adjust EQ bands to match a professional target spectral profile.
        Uses strict bypass thresholds to preserve original tone if already good.
        """
        profile_key = 'default'
        for key in self.TARGET_PROFILES:
            if key in track_type.lower():
                profile_key = key
                break
        
        target = self.TARGET_PROFILES[profile_key]
        bands = SpectralAnalyzer.GRANULAR_BANDS
        
        best_y = y.copy()
        current_y = y.copy()
        
        initial_profile = self.analyzer.get_spectral_profile(current_y, sr, bands=bands)
        initial_values = list(initial_profile.values())
        best_variance = np.sum((np.array(initial_values) - np.array(target))**2)
        
        # DO NO HARM: If variance is already very low, bypass optimization
        if best_variance < 50.0: # Variance is high due to squared sum, 50.0 is very subtle
            print(f"    [Objective EQ] [Bypassed] Initial Variance {best_variance:.2f} meets professional standard.")
            # Still apply clarity and shimmer as surgical touches
            best_y = self.apply_clarity_eq(best_y, sr)
            if profile_key not in ['kick', 'bass']:
                best_y = self.apply_shimmer(best_y)
            return best_y

        print(f"    [Objective EQ] Target: {profile_key} | Initial Variance: {best_variance:.2f}")

        for p in range(max_passes):
            current_profile = self.analyzer.get_spectral_profile(current_y, sr, bands=bands)
            current_values = np.array(list(current_profile.values()))
            
            deltas = np.array(target) - current_values
            worst_band_idx = np.argmax(np.abs(deltas))
            
            low, high = bands[worst_band_idx]
            center = (low + high) / 2
            
            # MINUTE TOUCHES: Limit gain to tiny 1.5dB increments (was 3.5dB)
            gain = np.clip(deltas[worst_band_idx] * 15.0, -1.5, 1.5)
            
            if abs(gain) < 0.1:
                break
                
            # Dynamic Q-Factor Logic:
            # - Sub/Bass (0-1): Wider Q (0.8) for musical shelf-like moves
            # - Mud/Box/Presence (2-6): Narrower Q (1.8) for surgical definition
            # - Detail/Air (7-9): Wider Q (1.2) for tonal shimmer
            if worst_band_idx < 2:
                q = 0.8
            elif 2 <= worst_band_idx <= 6:
                q = 1.8
            else:
                q = 1.2

            if worst_band_idx == 0:
                current_y = self.low_shelf(current_y, high, gain)
            elif worst_band_idx == len(bands) - 1:
                current_y = self.high_shelf(current_y, low, gain)
            else:
                current_y = self.peaking_filter(current_y, center, gain, Q=q)
            
            new_profile = self.analyzer.get_spectral_profile(current_y, sr, bands=bands)
            new_values = np.array(list(new_profile.values()))
            new_variance = np.sum((new_values - np.array(target))**2)
            
            if new_variance < best_variance:
                best_variance = new_variance
                best_y = current_y.copy()
            else:
                current_y = best_y.copy()
                break
        
        # Surgical Post-Pass: Mud (200-350Hz) & Harshness (2.5k-4.5k)
        best_y = self.apply_clarity_eq(best_y, sr)
        
        if profile_key not in ['kick', 'bass']:
            best_y = self.apply_shimmer(best_y)

        print(f"    [Objective EQ] Final Variance: {best_variance:.4f}")
        return best_y

    def apply_frequency_slotting(self, bass_y, kick_y, sr):
        """Find kick fundamental and surgically notch it from the bassline."""
        kick_freq = SpectralAnalyzer.get_fundamental_frequency(kick_y, sr)
        # Use narrow surgical notch (Q=3.0) to preserve bass body while clearing kick space
        # MINUTE TOUCH: Lower depth from -4.5 to -2.0dB
        print(f"  Kick fundamental: {kick_freq:.1f}Hz. Applying surgical notch to Bass (Q=3.0, -2.0dB).")
        return self.peaking_filter(bass_y, kick_freq, -2.0, Q=3.0)

    def apply_clarity_eq(self, y, sr):
        """Surgically reduce mud (200-350Hz) and harshness (2.5k-4.5k)."""
        # DO NO HARM: Strict bypass thresholds for surgical cuts
        mud_score = SpectralAnalyzer.detect_buildup(y, sr, (200, 350))
        if mud_score > 15.0: # Higher threshold (was 0.12 previously, wait, now it's 15.0 approx in new scale)
            print(f"  Surgical Fix: Mud buildup ({mud_score:.1f}). Cutting 250Hz.")
            y = self.peaking_filter(y, 250, -1.5, Q=1.5) # Gentler -1.5dB cut
            
        harsh_score = SpectralAnalyzer.detect_buildup(y, sr, (2500, 4500))
        if harsh_score > 3.0: 
            print(f"  Surgical Fix: Harshness detected ({harsh_score:.1f}). Cutting 3.2kHz.")
            y = self.peaking_filter(y, 3200, -1.0, Q=2.0)
            
        return y

    def apply_shimmer(self, y):
        """Add master-grade high-shelf shimmer (>10kHz)."""
        # MINUTE TOUCH: Lower shimmer gain from 2.0 to 1.0dB
        return self.high_shelf(y, 10000, 1.0)

    def intelligent_carve(self, y, ref_y, sr, freq_range, depth=-3.0):
        """
        Analyze reference signal for energy in range. If present, cut that range from target signal.
        This creates 'pockets' in the mix for specific instruments (e.g. Kick space in Bass).
        """
        ref_score = SpectralAnalyzer.detect_buildup(ref_y, sr, freq_range)
        # If reference track has significant energy buildup in this range (>2.5x average)
        if ref_score > 2.5:
            low, high = freq_range
            center = (low + high) / 2
            # print(f"    [Carving] Ref energy high ({ref_score:.1f}). Cutting {center:.0f}Hz by {depth}dB.")
            return self.peaking_filter(y, center, depth, Q=1.2)
        return y

if __name__ == "__main__":
    pass
