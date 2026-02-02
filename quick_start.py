"""
Quick Start Script
Automates the setup process for the Emotion-Based Music Recommendation System
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and show progress"""
    print(f"▶️  {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} detected. Need Python 3.8+")
        return False

def install_dependencies():
    """Install required dependencies"""
    print_header("📦 INSTALLING DEPENDENCIES")
    
    print("This will install:")
    print("  • TensorFlow")
    print("  • OpenCV")
    print("  • Streamlit")
    print("  • NumPy, Pandas, Matplotlib")
    print("  • scikit-learn")
    
    response = input("\nProceed with installation? (y/n): ").lower()
    
    if response == 'y':
        return run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing dependencies"
        )
    else:
        print("⏭️  Skipping dependency installation")
        return False

def check_dataset():
    """Check if dataset exists"""
    print_header("📊 CHECKING DATASET")
    
    if os.path.exists('data/fer2013.csv'):
        file_size = os.path.getsize('data/fer2013.csv') / (1024 * 1024)
        print(f"✅ FER-2013 dataset found ({file_size:.2f} MB)")
        return True
    else:
        print("❌ FER-2013 dataset not found")
        print("\n📥 To download the dataset:")
        print("   1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
        print("   2. Download the dataset")
        print("   3. Place 'fer2013.csv' in the 'data/' folder")
        
        response = input("\nOpen download page in browser? (y/n): ").lower()
        if response == 'y':
            import webbrowser
            webbrowser.open('https://www.kaggle.com/datasets/msambare/fer2013')
            print("🌐 Browser opened")
        
        return False

def train_model():
    """Train the emotion detection model"""
    print_header("🧠 TRAINING MODEL")
    
    if os.path.exists('model/emotion_model.h5'):
        print("⚠️  Model already exists")
        response = input("Retrain model? This will take 30-60 minutes (y/n): ").lower()
        if response != 'y':
            print("⏭️  Skipping model training")
            return True
    
    print("⏰ Training will take 30-60 minutes")
    print("💡 Make sure you have:")
    print("   • FER-2013 dataset in data/ folder")
    print("   • At least 4GB RAM")
    print("   • Stable power supply")
    
    response = input("\nStart training now? (y/n): ").lower()
    
    if response == 'y':
        print("\n🚀 Starting training...")
        print("📊 You can monitor progress in the terminal")
        return run_command(
            f"{sys.executable} train_emotion_model.py",
            "Model training"
        )
    else:
        print("⏭️  Skipping model training")
        print("💡 You can train later with: python train_emotion_model.py")
        return False

def run_application():
    """Run the Streamlit application"""
    print_header("🚀 LAUNCHING APPLICATION")
    
    if not os.path.exists('model/emotion_model.h5'):
        print("❌ Model not found. Please train the model first.")
        return False
    
    print("Starting Streamlit application...")
    print("🌐 Browser will open automatically")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(
            f"{sys.executable} -m streamlit run app.py",
            shell=True
        )
        return True
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped")
        return True

def main():
    """Main setup workflow"""
    print_header("🎵 EMOTION-BASED MUSIC RECOMMENDER - QUICK START")
    
    # Step 1: Check Python version
    print("Step 1: Checking Python version...")
    if not check_python_version():
        print("\n❌ Please install Python 3.8 or higher")
        return
    
    # Step 2: Install dependencies
    print("\nStep 2: Installing dependencies...")
    deps_installed = install_dependencies()
    
    # Step 3: Check dataset
    print("\nStep 3: Checking dataset...")
    dataset_exists = check_dataset()
    
    if not dataset_exists:
        print("\n⚠️  Cannot proceed without dataset")
        print("Please download the dataset and run this script again")
        return
    
    # Step 4: Train model (optional)
    print("\nStep 4: Training model...")
    model_trained = train_model()
    
    # Step 5: Run application
    if os.path.exists('model/emotion_model.h5'):
        print("\nStep 5: Ready to launch application...")
        response = input("Launch application now? (y/n): ").lower()
        
        if response == 'y':
            run_application()
        else:
            print("\n✅ Setup complete!")
            print("\n📝 Next steps:")
            print("   • Run: streamlit run app.py")
            print("   • Or: python quick_start.py (and choose to launch)")
    else:
        print("\n⚠️  Setup incomplete")
        print("\n📝 Next steps:")
        print("   1. Download FER-2013 dataset (if not done)")
        print("   2. Run: python train_emotion_model.py")
        print("   3. Run: streamlit run app.py")
    
    print_header("✨ THANK YOU!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error and try again")

