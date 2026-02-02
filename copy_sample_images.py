"""
Copy Sample Images from FER-2013 Dataset for Testing
This script copies one sample image from each emotion category for testing purposes.
"""

import os
import shutil
import random

def copy_sample_images():
    """Copy sample images from dataset to test_images folder"""
    
    # Define paths
    source_dir = "data/test"  # or "data/train"
    dest_dir = "test_images"
    
    # Emotion categories
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Create destination directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"✅ Created directory: {dest_dir}")
    
    print("\n📸 Copying sample images from dataset...\n")
    
    copied_count = 0
    
    for emotion in emotions:
        source_emotion_dir = os.path.join(source_dir, emotion)
        
        # Check if source directory exists
        if not os.path.exists(source_emotion_dir):
            print(f"⚠️  Warning: {source_emotion_dir} not found. Skipping {emotion}.")
            continue
        
        # Get list of images in this emotion folder
        images = [f for f in os.listdir(source_emotion_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"⚠️  Warning: No images found in {source_emotion_dir}. Skipping {emotion}.")
            continue
        
        # Select a random image (or first image)
        # Using index 10 to avoid potentially corrupted first images
        sample_image = images[min(10, len(images)-1)]
        
        # Source and destination paths
        source_path = os.path.join(source_emotion_dir, sample_image)
        dest_filename = f"sample_{emotion}.jpg"
        dest_path = os.path.join(dest_dir, dest_filename)
        
        # Copy the image
        try:
            shutil.copy2(source_path, dest_path)
            print(f"✅ {emotion.capitalize():10} → {dest_filename}")
            copied_count += 1
        except Exception as e:
            print(f"❌ Error copying {emotion}: {e}")
    
    print(f"\n🎉 Successfully copied {copied_count}/{len(emotions)} sample images!")
    print(f"📁 Images saved in: {os.path.abspath(dest_dir)}")
    print("\n📋 Next Steps:")
    print("1. Open the Streamlit app: streamlit run app.py")
    print("2. Click 'Upload an Image'")
    print("3. Select images from the 'test_images' folder")
    print("4. Take screenshots for your presentation!")
    
    return copied_count

def create_readme():
    """Create a README in test_images folder"""
    readme_content = """# Test Images for Screenshots

This folder contains sample images for each emotion category.

## Images:
- sample_angry.jpg - Angry expression
- sample_disgust.jpg - Disgust expression
- sample_fear.jpg - Fear expression
- sample_happy.jpg - Happy expression
- sample_sad.jpg - Sad expression
- sample_surprise.jpg - Surprise expression
- sample_neutral.jpg - Neutral expression

## How to Use:
1. Run: streamlit run app.py
2. Click "Upload an Image"
3. Select one of these images
4. Take screenshot of the result
5. Repeat for all emotions

## Screenshot Tips:
- Capture the entire result (image + emotion + music recommendations)
- Use Windows: Win + Shift + S
- Use Mac: Cmd + Shift + 4
- Save with descriptive names (e.g., screenshot_happy.png)
"""
    
    readme_path = os.path.join("test_images", "README.md")
    try:
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✅ Created README in test_images folder")
    except Exception as e:
        print(f"⚠️  Could not create README: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("  📸 FER-2013 Sample Image Copier")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists("data/test") and not os.path.exists("data/train"):
        print("\n❌ Error: Dataset not found!")
        print("Please ensure you have the FER-2013 dataset in one of these locations:")
        print("  - data/test/")
        print("  - data/train/")
        print("\nEach should contain subfolders: angry, disgust, fear, happy, sad, surprise, neutral")
        exit(1)
    
    # Copy sample images
    copied = copy_sample_images()
    
    # Create README
    if copied > 0:
        create_readme()
    
    print("\n" + "=" * 60)

