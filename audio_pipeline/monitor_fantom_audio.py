import sounddevice as sd
import numpy as np
import time

def monitor_fantom_audio(duration=10, device_index=9):
    print(f"Monitoring audio from device {device_index} (Fantom) for {duration} seconds...")
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        # Calculate RMS or Peak level
        peak = np.max(np.abs(indata))
        # Print a simple visual meter
        bars = int(peak * 50)
        print(f"Peak Level: {peak:.4f} " + "|" * bars, end="\r")

    try:
        with sd.InputStream(device=device_index, channels=2, callback=callback):
            time.sleep(duration)
    except Exception as e:
        print(f"\nError: {e}")
    print("\nMonitoring finished.")

if __name__ == "__main__":
    # We found Fantom is index 7 from the previous command
    monitor_fantom_audio(duration=30, device_index=9)
