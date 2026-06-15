from typing import List, Dict

def analyze_melody_intervals(notes: List[int]) -> Dict[str, float]:
    if len(notes) < 2: return {'stepwise': 0, 'small_leap': 0, 'large_leap': 0, 'repeated': 0}
    counts = {'stepwise': 0, 'small_leap': 0, 'large_leap': 0, 'repeated': 0}
    for i in range(1, len(notes)):
        interval = abs(notes[i] - notes[i-1])
        if interval == 0: counts['repeated'] += 1
        elif interval <= 2: counts['stepwise'] += 1
        elif interval <= 5: counts['small_leap'] += 1
        else: counts['large_leap'] += 1
    total = sum(counts.values())
    if total == 0: return counts
    return {k: (v / total) * 100 for k, v in counts.items()}

def analyze_voice_leading(bass_notes: List[int], melody_notes: List[int]) -> Dict[str, float]:
    if len(bass_notes) < 2 or len(melody_notes) < 2: return {'contrary': 0, 'parallel': 0, 'oblique': 0, 'similar': 0}
    counts = {'contrary': 0, 'parallel': 0, 'oblique': 0, 'similar': 0}
    for i in range(1, min(len(bass_notes), len(melody_notes))):
        bm, mm = bass_notes[i] - bass_notes[i-1], melody_notes[i] - melody_notes[i-1]
        if bm * mm < 0: counts['contrary'] += 1
        elif bm == 0 or mm == 0: counts['oblique'] += 1
        elif bm == mm: counts['parallel'] += 1
        else: counts['similar'] += 1
    total = sum(counts.values())
    if total == 0: return counts
    return {k: (v / total) * 100 for k, v in counts.items()}
