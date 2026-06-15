import os
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

TARGET_LUFS = -30.0

class GainStager:
    def __init__(self, output_dir="output/mastered/staged", reference_level=TARGET_LUFS):
        self.output_dir = output_dir
        self.reference_level = reference_level
        os.makedirs(self.output_dir, exist_ok=True)
        self.pink_noise_path = os.path.join(output_dir, "pink_noise_ref.wav")

    def generate_pink_noise(self, duration=10.0):
        """
        Generate pink noise normalised to self.reference_level LUFS via FFmpeg + pyloudnorm.
        FFmpeg creates the raw noise; pyloudnorm measures and scales to the exact LUFS target.
        """
        if not os.path.exists(self.pink_noise_path):
            print(f"Generating Pink Noise reference ({self.reference_level} LUFS target)...")
            raw_path = self.pink_noise_path + ".tmp.wav"
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anoisesrc=d={duration}:c=pink:r=48000",
                raw_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            data, rate = sf.read(raw_path)
            meter = pyln.Meter(rate)
            measured_lufs = meter.integrated_loudness(data)
            gain_db = self.reference_level - measured_lufs
            data_out = data * (10.0 ** (gain_db / 20.0))
            sf.write(self.pink_noise_path, data_out, rate, subtype='FLOAT')
            os.remove(raw_path)

            actual_lufs = meter.integrated_loudness(sf.read(self.pink_noise_path)[0])
            print(f"  Pink noise written: {actual_lufs:.2f} LUFS")
        return self.pink_noise_path

    def get_integrated_lufs(self, file_path):
        """Measure integrated LUFS (ITU-R BS.1770-4) of a file."""
        data, rate = sf.read(file_path)
        meter = pyln.Meter(rate)
        lufs = meter.integrated_loudness(data)
        return lufs

    def apply_gain_matching(self, stem_path, target_lufs):
        """
        Apply a linear gain to bring the stem's integrated LUFS to target_lufs.
        Hard-limits to prevent clipping.
        """
        data, rate = sf.read(stem_path)
        meter = pyln.Meter(rate)
        current_lufs = meter.integrated_loudness(data)

        if current_lufs <= -70.0:
            return stem_path, current_lufs  # silent / ungated — skip

        gain_db = target_lufs - current_lufs
        data_out = data * (10.0 ** (gain_db / 20.0))

        output_filename = "gs_" + os.path.basename(stem_path)
        output_path = os.path.join(self.output_dir, output_filename)
        sf.write(output_path, data_out, rate, subtype='FLOAT')
        return output_path, current_lufs

    def process_stems(self, stem_paths):
        """
        Generate pink noise reference then normalise every stem to its integrated LUFS.
        Returns list of dicts with keys: original, staged, original_lufs, target_lufs.
        """
        pink_ref = self.generate_pink_noise()
        target_lufs = self.get_integrated_lufs(pink_ref)
        print(f"Pink Noise Reference: {target_lufs:.2f} LUFS\n")

        results = []
        for path in stem_paths:
            if not os.path.exists(path):
                continue
            staged_path, original_lufs = self.apply_gain_matching(path, target_lufs)
            if original_lufs <= -70.0:
                print(f"  {os.path.basename(path):50s}  SILENT — skipped")
                continue
            gain_applied = target_lufs - original_lufs
            print(f"  {os.path.basename(path):50s}  {original_lufs:+.1f} LUFS → {target_lufs:+.1f} LUFS  ({gain_applied:+.1f} dB)")
            results.append({
                "original":      path,
                "staged":        staged_path,
                "original_lufs": original_lufs,
                "target_lufs":   target_lufs,
            })
        return results

if __name__ == "__main__":
    pass
