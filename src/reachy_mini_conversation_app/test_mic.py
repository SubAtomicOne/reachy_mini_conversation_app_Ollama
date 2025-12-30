import sounddevice as sd
import numpy as np

print("\n--- AVAILABLE AUDIO DEVICES ---")
print(sd.query_devices())

def test_mic(device_id):
    print(f"\nTesting Device {device_id}...")
    try:
        def callback(indata, frames, time, status):
            if status:
                print(status)
            volume = np.linalg.norm(indata) * 10
            print(f"\rVolume: {int(volume)} |" + "=" * int(volume/5), end="")

        with sd.InputStream(device=device_id, channels=1, callback=callback):
            print(f"Listening on Device {device_id}. Speak now! (Ctrl+C to stop)")
            while True:
                sd.sleep(100)
    except Exception as e:
        print(f"\nError on device {device_id}: {e}")

# Ask user for ID
try:
    target = int(input("\n\nEnter the ID number of your Microphone from the list above: "))
    test_mic(target)
except KeyboardInterrupt:
    print("\nDone.")