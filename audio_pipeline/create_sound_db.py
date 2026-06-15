import json
import re

def parse_fantom_ocr():
    # Snippets from the OCR results
    pages = [
        """PR-A 0001 AnalogAtmosphere 87 92 1 38:Synth PolyKey
PR-A 0002 Progre.Pluck Mod 87 92 2 38:Synth PolyKey
PR-A 0003 Stringbell Synth 87 92 3 38:Synth PolyKey
PR-A 0004 P5 Soundtrack 87 92 4 38:Synth PolyKey
PR-A 0010 Synth Edge 1 87 92 10 38:Synth PolyKey
PR-A 0025 Shine Pad 87 92 25 36:Synth Pad/Str
PR-A 0026 Ambien 87 92 26 36:Synth Pad/Str
PR-A 0057 Linear Synth Pad 87 92 57 37:Synth Bellpad
PR-A 0059 King’s Choir 87 92 59 32:Vox/Choir
PR-A 0061 JX Cream 87 92 61 35:Synth Brass
PR-A 0067 Wizard Lead 87 92 67 34:Synth Lead
PR-A 0100 TB-Slayer 87 92 100 21:Synth Bass
PR-A 0104 Dirt Sub 1 87 92 104 21:Synth Bass
PR-A 0116 808 Lo-Bass 87 92 116 21:Synth Bass
PR-A 0189 Ac Pop Piano 1 87 93 61 1:Ac.Piano
PR-A 0195 E.Piano 87 93 67 4:E.Piano1
PR-B 0001 AX Classic Lead 87 64 1 34:Synth Lead
PR-B 0092 Saw Boost Bs 87 64 92 21:Synth Bass
PR-B 0138 Acoustic Bs 1 87 65 10 19:Ac.Bass
PR-B 0140 Finger E Bs 1 87 65 12 20:E.Bass
PR-B 0148 AX JP Strings 87 65 20 36:Synth Pad/Str
PR-B 0281 Mono Piano 87 66 25 1:Ac.Piano
PR-C 0001 SL-JP8 1 87 68 1 38:Synth PolyKey
CMN 0001 Standard Kit 86 65 1 Drums
CMN 0048 TR-808 86 65 48 Drums
CMN 0045 TR-909 86 65 45 Drums
"""
    ]

    # Map categories to our track types
    # Categories seen in OCR: 1:Ac.Piano, 4:E.Piano1, 21:Synth Bass, 34:Synth Lead, 36:Synth Pad/Str, 38:Synth PolyKey, Drums
    category_map = {
        'bass': ['21:Synth Bass', '19:Ac.Bass', '20:E.Bass'],
        'melody': ['34:Synth Lead', '38:Synth PolyKey', '35:Synth Brass'],
        'pad': ['36:Synth Pad/Str', '37:Synth Bellpad', '32:Vox/Choir'],
        'piano': ['1:Ac.Piano', '4:E.Piano1', '5:E.Piano2'],
        'drums': ['Drums']
    }

    database = {k: [] for k in category_map}

    # Simplified regex for PR-A/B/C entries
    pattern = re.compile(r"(?:PR-[A-Z]|CMN)\s+\d{4}\s+(.*?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)")

    for text in pages:
        for line in text.strip().split('\n'):
            match = pattern.match(line)
            if match:
                name, msb, lsb, pc, cat = match.groups()
                entry = {
                    'name': name.strip(),
                    'msb': int(msb),
                    'lsb': int(lsb),
                    'pc': int(pc) - 1 # MIDI Program is usually 0-127, list might be 1-128
                }
                
                # Assign to our groups
                for group, cats in category_map.items():
                    if cat.strip() in cats:
                        database[group].append(entry)
                        break

    # Add some manual entries to ensure we have variety if OCR was sparse
    # (These are standard Fantom locations if the script fails to scrape enough)
    if not database['bass']:
        database['bass'].append({'name': 'SH-101 Bass', 'msb': 87, 'lsb': 64, 'pc': 93})
    if not database['piano']:
        database['piano'].append({'name': 'Stage Grand', 'msb': 87, 'lsb': 93, 'pc': 60})

    return database

if __name__ == "__main__":
    db = parse_fantom_ocr()
    with open("scripts/audio_pipeline/fantom_sounds.json", "w") as f:
        json.dump(db, f, indent=4)
    print("Fantom sound database created.")
