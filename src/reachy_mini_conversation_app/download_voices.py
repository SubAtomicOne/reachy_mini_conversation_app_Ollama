import requests
import json
import os

# URLs for the correct files
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"

os.makedirs("models", exist_ok=True)

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f"models/{filename}", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ {filename} downloaded successfully.")
    else:
        print(f"❌ Failed to download {filename} (Status: {response.status_code})")

# Download both to be safe
download_file(VOICES_URL, "voices.json")

if not os.path.exists("models/kokoro-v0_19.onnx"):
    download_file(MODEL_URL, "kokoro-v0_19.onnx")

# VERIFY the voices file is actual JSON
try:
    with open("models/voices.json", "r") as f:
        data = json.load(f)
    print("✅ VALIDATION PASSED: voices.json is valid text.")
except Exception as e:
    print(f"❌ VALIDATION FAILED: voices.json is still corrupt! Error: {e}")