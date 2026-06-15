"""
Stem Context Analyzer — analyzes all stems together for spectral collision detection.

Finds frequency collisions between stems and generates EQ suggestions
based on priority order (lower-priority stem gets cut).
"""

import os
import numpy as np
import soundfile as sf


# Stem priority order (lower number = higher priority, gets precedence)
STEM_PRIORITY = {
    'main_melody': 1,
    'bass': 2,
    'pad': 3,
    'counter_melody': 4,
    'kick': 5,
    'snare': 5,
    'hat': 6,
    'clap': 6,
    'tambourine': 6,
    'maracas': 6,
    'perc': 6,
    'fx': 7,
}


def detect_role(name: str) -> str:
    """Detect stem role from filename."""
    name_lower = name.lower()
    if any(x in name_lower for x in ['kick']):
        return 'kick'
    elif any(x in name_lower for x in ['snare']):
        return 'snare'
    elif any(x in name_lower for x in ['hat', 'clap', 'tambourine', 'maracas', 'perc']):
        return 'hat'
    elif 'bass' in name_lower:
        return 'bass'
    elif any(x in name_lower for x in ['main_melody', 'melody_lead']):
        return 'main_melody'
    elif any(x in name_lower for x in ['counter_melody', 'counter_lead']):
        return 'counter_melody'
    elif any(x in name_lower for x in ['pad', 'chord']):
        return 'pad'
    elif 'fx' in name_lower:
        return 'fx'
    elif any(x in name_lower for x in ['melody', 'chorus']):
        return 'main_melody'
    return 'fx'


def get_priority(role: str) -> int:
    """Get priority number for a role (lower = higher priority)."""
    return STEM_PRIORITY.get(role, 7)


def analyze_stem_context(stem_paths: list, sr: int, n_bands: int = 10) -> dict:
    """
    Analyze all stems together to find spectral collisions.

    Args:
        stem_paths: list of paths to stem WAV files
        sr: sample rate
        n_bands: number of frequency bands for analysis

    Returns:
        dict with:
        - 'profiles': {stem_name: [band_energies]}
        - 'collisions': list of collision dicts
        - 'suggestions': {stem_name: [(band_idx, cut_db), ...]}
    """
    # Frequency band edges (log-spaced, 20Hz to 20kHz)
    band_edges = np.logspace(np.log10(20), np.log10(20000), n_bands + 1)
    band_centers = [(band_edges[i] + band_edges[i + 1]) / 2 for i in range(n_bands)]

    # Compute spectral profile per stem
    profiles = {}
    stem_roles = {}

    for path in stem_paths:
        name = os.path.basename(path)
        role = detect_role(name)
        stem_roles[name] = role

        try:
            y, file_sr = sf.read(path, always_2d=True)
            y = np.asarray(y, dtype=np.float64)
            if y.ndim == 2:
                y = np.mean(y, axis=1)
        except Exception:
            continue

        # Compute band energies via FFT
        n_fft = 8192
        spectrum = np.abs(np.fft.rfft(y, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / file_sr)

        band_energies = []
        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            if np.any(mask):
                energy = np.mean(spectrum[mask])
            else:
                energy = 0.0
            band_energies.append(energy)

        # Normalize to 0-1
        max_e = max(band_energies) if band_energies else 1.0
        if max_e > 0:
            band_energies = [e / max_e for e in band_energies]

        profiles[name] = band_energies

    # Find collisions (pairs with high energy in same band)
    collisions = []
    stem_names = list(profiles.keys())

    for i in range(len(stem_names)):
        for j in range(i + 1, len(stem_names)):
            name_a = stem_names[i]
            name_b = stem_names[j]
            profile_a = profiles[name_a]
            profile_b = profiles[name_b]

            for band_idx in range(n_bands):
                # Both stems have significant energy in this band
                if profile_a[band_idx] > 0.6 and profile_b[band_idx] > 0.6:
                    role_a = stem_roles.get(name_a, 'fx')
                    role_b = stem_roles.get(name_b, 'fx')
                    priority_a = get_priority(role_a)
                    priority_b = get_priority(role_b)

                    # Lower-priority stem gets cut
                    if priority_a > priority_b:
                        cut_target = name_a
                    elif priority_b > priority_a:
                        cut_target = name_b
                    else:
                        # Same priority: cut the one with less energy in that band
                        cut_target = name_a if profile_a[band_idx] < profile_b[band_idx] else name_b

                    # Cut amount proportional to how much both exceed threshold
                    excess = min(profile_a[band_idx], profile_b[band_idx]) - 0.6
                    cut_db = min(-0.5, -excess * 3.0)  # scale to -0.5 to -1.5 dB
                    cut_db = max(cut_db, -1.5)  # cap at -1.5 dB

                    collisions.append({
                        'stem_a': name_a,
                        'stem_b': name_b,
                        'band_idx': band_idx,
                        'freq_center': band_centers[band_idx],
                        'cut_target': cut_target,
                        'suggested_cut_db': cut_db,
                    })

    # Build per-stem suggestions
    suggestions = {}
    for coll in collisions:
        target = coll['cut_target']
        if target not in suggestions:
            suggestions[target] = []
        suggestions[target].append((coll['band_idx'], coll['suggested_cut_db']))

    return {
        'profiles': profiles,
        'collisions': collisions,
        'suggestions': suggestions,
        'band_centers': band_centers,
        'band_edges': band_edges.tolist(),
    }
