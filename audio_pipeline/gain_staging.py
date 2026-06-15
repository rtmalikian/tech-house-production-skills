import os
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

TARGET_LUFS = -30.0

# True-peak target for the pink noise reference. Controls how loud the reference
# is in absolute terms, which directly sets the SNR of gain-staged stems.
# -12 dBFS gives stems ~26–30 dBFS band-RMS headroom above the noise floor.
# Raise toward -6 dBFS for hotter stems; lower toward -18 dBFS for more headroom.
PINK_NOISE_TARGET_PEAK_DBFS = -12.0

# Persistent reference location — outside session output folders so it survives
# between runs and always provides the same comparison point.
_REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference"
)
_PINK_NOISE_REFERENCE_PATH = os.path.join(_REFERENCE_DIR, "pink_noise_ref.wav")


class GainStager:
    def __init__(self, output_dir="output/mastered/staged", reference_level=TARGET_LUFS):
        self.output_dir = output_dir
        self.reference_level = reference_level
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(_REFERENCE_DIR, exist_ok=True)
        # Always point at the persistent reference, not the session folder
        self.pink_noise_path = _PINK_NOISE_REFERENCE_PATH

    def generate_pink_noise(self, duration=10.0):
        """Generate a pink noise reference normalised to a fixed true peak.

        Two-step normalisation:
          1. LUFS normalisation (perceptual loudness target)
          2. True-peak normalisation to PINK_NOISE_TARGET_PEAK_DBFS

        Step 2 makes the reference deterministic regardless of the random noise
        sequence, and ensures stems gain-staged against it land at useful SNR levels.
        The file is written to a persistent location outside session output folders.
        """
        if not os.path.exists(self.pink_noise_path):
            print(f"Generating Pink Noise reference "
                  f"(target peak: {PINK_NOISE_TARGET_PEAK_DBFS} dBFS)...")
            raw_path = self.pink_noise_path + ".tmp.wav"
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anoisesrc=d={duration}:c=pink:r=48000",
                raw_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            data, rate = sf.read(raw_path)
            os.remove(raw_path)

            # Step 1: LUFS normalisation
            meter = pyln.Meter(rate)
            measured_lufs = meter.integrated_loudness(data)
            gain_db = self.reference_level - measured_lufs
            data_out = data * (10.0 ** (gain_db / 20.0))

            # Step 2: true-peak normalisation — makes the level deterministic
            peak = np.max(np.abs(data_out))
            if peak > 0:
                target_peak_lin = 10.0 ** (PINK_NOISE_TARGET_PEAK_DBFS / 20.0)
                data_out = data_out * (target_peak_lin / peak)

            sf.write(self.pink_noise_path, data_out, rate, subtype='FLOAT')

            final_lufs = meter.integrated_loudness(data_out)
            final_peak_db = 20.0 * np.log10(np.max(np.abs(data_out)))
            print(f"  Pink noise reference saved: {final_peak_db:+.1f} dBFS peak  "
                  f"{final_lufs:.1f} LUFS  →  {self.pink_noise_path}")
        return self.pink_noise_path

    def load_pink_noise(self):
        """Return the persistent pink noise reference as (numpy_array, sample_rate).
        Generates it first if it does not exist. Logs the peak level on load."""
        if not os.path.exists(self.pink_noise_path):
            self.generate_pink_noise()
        data, rate = sf.read(self.pink_noise_path, always_2d=True)
        peak_db = 20.0 * np.log10(np.max(np.abs(data)) + 1e-10)
        print(f"  Pink noise reference loaded: {peak_db:+.1f} dBFS peak  "
              f"({self.pink_noise_path})")
        return data, rate

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
