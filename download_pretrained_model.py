"""
Download a pre-trained emotion recognition model
This model is trained on FER-2013 with better accuracy
"""

import urllib.request
import os

print("🔽 Downloading pre-trained emotion recognition model...")
print("   This model has been trained on the full FER-2013 dataset")
print("   Expected accuracy: ~65%")
print()

# Create model directory
os.makedirs('model', exist_ok=True)

# URL for a pre-trained FER-2013 model
# This is a publicly available model trained on FER-2013
url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

output_path = "model/emotion_model.h5"

try:
    print(f"📥 Downloading from: {url}")
    print(f"📁 Saving to: {output_path}")
    print()
    print("⏳ Please wait, this may take a minute...")
    
    urllib.request.urlretrieve(url, output_path)
    
    print()
    print("=" * 60)
    print("✅ Download completed successfully!")
    print(f"   Model saved to: {output_path}")
    print(f"   Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("=" * 60)
    print()
    print("🎯 This pre-trained model should give much better results!")
    print("   You can now run the application with: streamlit run app.py")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print()
    print("Alternative: Let me create a better training script...")

