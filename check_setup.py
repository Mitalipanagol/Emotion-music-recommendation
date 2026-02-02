"""
Setup Verification Script
Checks if all dependencies and files are properly configured
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Not installed")
            all_installed = False
    
    return all_installed

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = ['model', 'data']
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/ directory exists")
        else:
            print(f"   ❌ {dir_name}/ directory missing")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset is available"""
    print("\n📊 Checking dataset...")
    
    dataset_path = "data/fer2013.csv"
    if os.path.exists(dataset_path):
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # MB
        print(f"   ✅ FER-2013 dataset found ({file_size:.2f} MB)")
        return True
    else:
        print(f"   ❌ FER-2013 dataset not found at {dataset_path}")
        print("   📥 Please download from: https://www.kaggle.com/datasets/msambare/fer2013")
        return False

def check_model():
    """Check if trained model exists"""
    print("\n🧠 Checking trained model...")
    
    model_path = "model/emotion_model.h5"
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   ✅ Trained model found ({file_size:.2f} MB)")
        return True
    else:
        print(f"   ⚠️  Trained model not found at {model_path}")
        print("   ℹ️  Run 'python train_emotion_model.py' to train the model")
        return False

def check_webcam():
    """Check if webcam is accessible"""
    print("\n📹 Checking webcam...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("   ✅ Webcam is accessible")
                return True
            else:
                print("   ⚠️  Webcam opened but couldn't read frame")
                return False
        else:
            print("   ❌ Could not open webcam")
            return False
    except Exception as e:
        print(f"   ❌ Error checking webcam: {e}")
        return False

def main():
    """Run all checks"""
    print("="*60)
    print("🔍 EMOTION-BASED MUSIC RECOMMENDER - SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Directories': check_directories(),
        'Dataset': check_dataset(),
        'Model': check_model(),
        'Webcam': check_webcam()
    }
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    for check, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}")
    
    print("\n" + "="*60)
    
    # Overall status
    critical_checks = ['Python Version', 'Dependencies', 'Directories']
    critical_passed = all(results[check] for check in critical_checks)
    
    if critical_passed:
        if results['Dataset'] and results['Model']:
            print("🎉 ALL CHECKS PASSED! You're ready to run the application.")
            print("\n▶️  Run: streamlit run app.py")
        elif results['Dataset']:
            print("⚠️  Setup is OK, but model needs training.")
            print("\n▶️  Next step: python train_emotion_model.py")
        else:
            print("⚠️  Setup is OK, but dataset is missing.")
            print("\n▶️  Next step: Download FER-2013 dataset")
            print("   https://www.kaggle.com/datasets/msambare/fer2013")
    else:
        print("❌ SETUP INCOMPLETE - Please fix the issues above.")
        print("\n▶️  Install missing dependencies: pip install -r requirements.txt")
    
    print("="*60)
    
    if not results['Webcam']:
        print("\n💡 TIP: If webcam doesn't work, you can still use 'Upload Image' mode")

if __name__ == "__main__":
    main()

