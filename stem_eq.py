"""
STEM EQ — Tech house specific EQ curves for each stem category.
Based on producer research: iZotope, EDMProd, gearslutz, YouTube tutorials.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, iirfilter, sosfilt
import os
import sys

# ============================================================================
# TECH HOUSE EQ CURVES
# Each curve is a list of (freq_hz, gain_db, Q, filter_type)
# filter_type: 'peak', 'highpass', 'lowpass', 'lowshelf', 'highshelf'
# ============================================================================
EQ_CURVES = {
    'kick': [
        ('highpass', 30, 0.707, 0),      # Remove sub-rumble below 30 Hz
        ('peak', 60, 3.0, 0.7),          # Boost fundamental body
        ('peak', 300, -3.0, 2.0),        # Cut boxiness
        ('peak', 3500, 2.5, 1.5),        # Boost click/snap
        ('lowpass', 12000, 0.707, 0),    # Clean up high-end
    ],
    'closed_hat': [
        ('highpass', 400, 0.707, 0),     # Remove low-end bleed
        ('peak', 6000, -3.0, 4.0),       # Tame harshness (dynamic)
        ('peak', 9000, 1.5, 0.7),        # Clarity
        ('peak', 13000, 1.0, 0.7),       # Air
        ('lowpass', 16000, 0.707, 0),    # Clean up top
    ],
    'open_hat': [
        ('highpass', 350, 0.707, 0),     # Remove low-end bleed
        ('peak', 5500, -4.0, 4.0),       # TAME HARSHNESS (the problem freq!)
        ('peak', 7000, -2.0, 3.0),       # Additional harsh cut
        ('peak', 9000, 1.0, 0.7),        # Clarity
        ('lowpass', 11000, 0.707, 0),    # Differentiate from closed hat
    ],
    'clap': [
        ('highpass', 150, 0.707, 0),     # Remove low-end
        ('peak', 250, 2.0, 1.0),         # Body/weight
        ('peak', 400, -2.0, 2.0),        # Cut boxiness
        ('peak', 3000, 3.0, 1.5),        # Presence/snap
        ('peak', 8000, 2.0, 0.7),        # Brightness
    ],
    'snare': [
        ('highpass', 100, 0.707, 0),     # Remove low-end
        ('peak', 200, 2.0, 1.0),         # Body
        ('peak', 350, -2.0, 2.0),        # Cut boxiness
        ('peak', 2500, 2.5, 1.5),        # Crack
        ('peak', 7000, 1.5, 0.7),        # Brightness
    ],
    'bass': [
        ('highpass', 25, 0.707, 0),      # Remove sub-rumble
        ('peak', 50, 2.0, 0.7),          # Sub weight
        ('peak', 100, 2.5, 0.7),         # Mid-bass punch
        ('peak', 250, -3.0, 2.0),        # Cut mud
        ('peak', 1000, 2.0, 1.5),        # Presence/grind
        ('lowpass', 6000, 0.707, 0),     # Remove high-end
    ],
    'sub_bass': [
        ('highpass', 20, 0.707, 0),      # Remove sub-rumble
        ('peak', 40, 2.0, 0.7),          # Sub fundamental
        ('peak', 80, 1.5, 0.7),          # Upper sub
        ('lowpass', 150, 0.707, 0),      # Keep it clean
    ],
    'stab': [
        ('highpass', 200, 0.707, 0),     # Remove low-end masking
        ('peak', 350, -3.0, 2.0),        # Cut mud
        ('peak', 3000, -2.0, 2.0),       # Cut presence (make room for clap)
        ('peak', 8000, -1.5, 0.7),       # Reduce harshness
    ],
    'acid': [
        ('highpass', 100, 0.707, 0),     # Remove low-end
        ('peak', 300, -2.0, 2.0),        # Cut mud
        ('peak', 800, 2.0, 1.5),         # Resonance character
        ('peak', 3000, -1.5, 2.0),       # Cut presence (make room)
        ('lowpass', 8000, 0.707, 0),     # Clean up
    ],
    'pad': [
        ('highpass', 250, 0.707, 0),     # Aggressive HPF — kick/bass own low-end
        ('peak', 350, -4.0, 2.0),        # Cut mud aggressively
        ('peak', 3000, -2.5, 2.0),       # Cut presence (make room for clap)
        ('peak', 8000, -1.5, 0.7),       # Reduce harshness
    ],
    'fx': [
        ('highpass', 200, 0.707, 0),     # Remove low-end
        ('peak', 500, -2.0, 2.0),        # Cut mud
        ('peak', 4000, 1.5, 1.5),        # Presence
    ],
    'shaker': [
        ('highpass', 500, 0.707, 0),     # Remove low-end
        ('peak', 6000, -2.0, 3.0),       # Tame harshness
        ('peak', 10000, 1.0, 0.7),       # Air
    ],
    'tambourine': [
        ('highpass', 400, 0.707, 0),     # Remove low-end
        ('peak', 5000, -2.0, 3.0),       # Tame harshness
        ('peak', 8000, 1.5, 0.7),        # Brightness
    ],
    'ride': [
        ('highpass', 300, 0.707, 0),     # Remove low-end
        ('peak', 4000, -2.0, 3.0),       # Tame harshness
        ('peak', 10000, 1.0, 0.7),       # Air
    ],
    'crash': [
        ('highpass', 250, 0.707, 0),     # Remove low-end
        ('peak', 4500, -3.0, 3.0),       # Tame harshness
        ('peak', 8000, 1.0, 0.7),        # Brightness
    ],
}

# ============================================================================
# STEM CLASSIFICATION
# ============================================================================
CATEGORIES = {
    'kick': ['kick_n36'], 'bass': ['bass'], 'sub_bass': ['sub_bass'],
    'clap': ['clap_n39'], 'snare': ['snare_n38'],
    'closed_hat': ['closedhat_n42'], 'open_hat': ['openhat_n46'],
    'ride': ['ride_n51'], 'crash': ['crash_n49'],
    'tambourine': ['tambourine_n54'], 'shaker': ['shaker', 'maracas'],
    'sidestick': ['sidestick_n37'], 'cowbell': ['cowbell_n56'],
    'stab': ['chord_stab'], 'acid': ['acid_line'],
    'pad': ['pad'], 'fx': ['fx'],
}

def classify(fname):
    for cat, patterns in CATEGORIES.items():
        for p in patterns:
            if p in fname.lower():
                return cat
    return 'other'

# ============================================================================
# EQ PROCESSING
# ============================================================================
def apply_peak_eq(signal, sr, freq, gain_db, Q):
    """Apply a peaking EQ filter."""
    w0 = 2 * np.pi * freq / sr
    A = 10 ** (gain_db / 40)
    alpha = np.sin(w0) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    # Normalize
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1.0, a1/a0, a2/a0])
    
    # Apply filter
    from scipy.signal import lfilter
    return lfilter(b, a, signal)

def apply_eq_curve(signal, sr, curve):
    """Apply a full EQ curve to a signal."""
    result = signal.copy()
    
    for filter_type, freq, param1, param2 in curve:
        if filter_type == 'highpass':
            sos = butter(2, freq, btype='high', fs=sr, output='sos')
            if result.ndim == 2:
                result = np.column_stack([sosfiltfilt(sos, result[:, ch]) for ch in range(result.shape[1])])
            else:
                result = sosfiltfilt(sos, result)
        
        elif filter_type == 'lowpass':
            sos = butter(2, freq, btype='low', fs=sr, output='sos')
            if result.ndim == 2:
                result = np.column_stack([sosfiltfilt(sos, result[:, ch]) for ch in range(result.shape[1])])
            else:
                result = sosfiltfilt(sos, result)
        
        elif filter_type == 'peak':
            gain_db = param1
            Q = param2
            if result.ndim == 2:
                result = np.column_stack([apply_peak_eq(result[:, ch], sr, freq, gain_db, Q) for ch in range(result.shape[1])])
            else:
                result = apply_peak_eq(result, sr, freq, gain_db, Q)
        
        elif filter_type == 'lowshelf':
            gain_db = param1
            Q = param2
            # Simple low shelf using peak EQ at low frequency with wide Q
            if result.ndim == 2:
                result = np.column_stack([apply_peak_eq(result[:, ch], sr, freq, gain_db, max(Q, 0.5)) for ch in range(result.shape[1])])
            else:
                result = apply_peak_eq(result, sr, freq, gain_db, max(Q, 0.5))
        
        elif filter_type == 'highshelf':
            gain_db = param1
            Q = param2
            if result.ndim == 2:
                result = np.column_stack([apply_peak_eq(result[:, ch], sr, freq, gain_db, max(Q, 0.5)) for ch in range(result.shape[1])])
            else:
                result = apply_peak_eq(result, sr, freq, gain_db, max(Q, 0.5))
    
    return result

def process_stems(stems_dir, output_dir):
    """Apply tech house EQ curves to all stems."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEM EQ — Tech House Curves")
    print("=" * 60)
    
    processed = 0
    
    for fname in sorted(os.listdir(stems_dir)):
        if not fname.endswith('.wav') or fname.startswith('pass'):
            continue
        
        path = os.path.join(stems_dir, fname)
        output_path = os.path.join(output_dir, fname)
        
        try:
            data, sr = sf.read(path)
            category = classify(fname)
            curve = EQ_CURVES.get(category)
            
            if curve:
                print(f"\n  {fname:<35} category={category}")
                print(f"    EQ curve:")
                for ftype, freq, p1, p2 in curve:
                    if ftype in ('highpass', 'lowpass'):
                        print(f"      {ftype} {freq} Hz")
                    else:
                        print(f"      {ftype} {freq} Hz {p1:+.1f} dB Q={p2}")
                
                # Apply EQ
                eqd = apply_eq_curve(data, sr, curve)
                
                # Clip to prevent overflow
                eqd = np.clip(eqd, -1.0, 1.0)
                
                sf.write(output_path, eqd, sr, subtype='PCM_24')
                processed += 1
                
                # Report level change
                orig_rms = 20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-10)
                new_rms = 20 * np.log10(np.sqrt(np.mean(eqd**2)) + 1e-10)
                print(f"    Level: {orig_rms:.1f} → {new_rms:.1f} dBFS ({new_rms - orig_rms:+.1f} dB)")
            else:
                # No EQ curve, copy unchanged
                sf.write(output_path, data, sr, subtype='PCM_24')
                print(f"\n  {fname:<35} category={category} — no curve, copied")
        
        except Exception as e:
            print(f"\n  {fname}: ERROR — {e}")
    
    print(f"\n{'='*60}")
    print(f"  Processed: {processed} stems")
    print(f"  Output:    {output_dir}")
    
    return output_dir, processed

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python stem_eq.py <stems_dir> [output_dir]")
        sys.exit(1)
    
    stems_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else stems_dir.rstrip('/') + '_eq'
    process_stems(stems_dir, output_dir)
