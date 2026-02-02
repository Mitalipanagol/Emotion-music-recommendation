"""
FER-2013 Dataset Download Helper
Provides instructions and links for downloading the dataset
"""

import os
import webbrowser

def main():
    print("="*70)
    print("📥 FER-2013 DATASET DOWNLOAD GUIDE")
    print("="*70)
    
    print("\n📊 About FER-2013 Dataset:")
    print("   • 35,887 grayscale images of faces")
    print("   • 48x48 pixel resolution")
    print("   • 7 emotion categories")
    print("   • ~308 MB CSV file")
    
    print("\n" + "="*70)
    print("DOWNLOAD OPTIONS")
    print("="*70)
    
    print("\n🔹 Option 1: Kaggle (Recommended)")
    print("   URL: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   Steps:")
    print("   1. Create a Kaggle account (if you don't have one)")
    print("   2. Click the 'Download' button")
    print("   3. Extract the ZIP file")
    print("   4. Copy 'fer2013.csv' to the 'data/' folder")
    
    print("\n🔹 Option 2: Original Kaggle Competition")
    print("   URL: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
    print("   Steps:")
    print("   1. Join the competition")
    print("   2. Download the dataset")
    print("   3. Extract and copy 'fer2013.csv' to 'data/' folder")
    
    print("\n🔹 Option 3: Alternative Sources")
    print("   • Search for 'FER-2013 dataset download' on Google")
    print("   • Check GitHub repositories with pre-processed versions")
    print("   • Ask your project guide/instructor")
    
    print("\n" + "="*70)
    print("AFTER DOWNLOADING")
    print("="*70)
    
    print("\n✅ Verify your setup:")
    print("   1. Ensure 'fer2013.csv' is in the 'data/' folder")
    print("   2. Run: python check_setup.py")
    print("   3. If all checks pass, run: python train_emotion_model.py")
    
    print("\n" + "="*70)
    
    # Check if data folder exists
    if not os.path.exists('data'):
        print("\n⚠️  WARNING: 'data' folder not found!")
        print("   Creating 'data' folder...")
        os.makedirs('data')
        print("   ✅ 'data' folder created")
    else:
        print("\n✅ 'data' folder exists")
    
    # Check if dataset already exists
    if os.path.exists('data/fer2013.csv'):
        file_size = os.path.getsize('data/fer2013.csv') / (1024 * 1024)
        print(f"\n🎉 Dataset already exists! ({file_size:.2f} MB)")
        print("   You can proceed with training: python train_emotion_model.py")
    else:
        print("\n📥 Dataset not found. Please download it.")
        
        # Ask if user wants to open browser
        response = input("\n❓ Open Kaggle download page in browser? (y/n): ").lower()
        if response == 'y':
            print("\n🌐 Opening browser...")
            webbrowser.open('https://www.kaggle.com/datasets/msambare/fer2013')
            print("   ✅ Browser opened. Please download the dataset.")
        else:
            print("\n📋 Manual download:")
            print("   Visit: https://www.kaggle.com/datasets/msambare/fer2013")
    
    print("\n" + "="*70)
    print("💡 TIPS")
    print("="*70)
    print("   • File name must be exactly 'fer2013.csv' (case-sensitive)")
    print("   • Place it directly in 'data/' folder, not in a subfolder")
    print("   • Expected file size: ~308 MB")
    print("   • Don't modify the CSV file")
    print("="*70)

if __name__ == "__main__":
    main()

