# 🚀 Getting Started - Quick Reference

## ⚡ 3-Step Quick Start

### Step 1️⃣: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2️⃣: Download Dataset
1. Visit: https://www.kaggle.com/datasets/msambare/fer2013
2. Download `fer2013.csv`
3. Place in `data/` folder

### Step 3️⃣: Train & Run
```bash
# Train the model (30-60 minutes)
python train_emotion_model.py

# Run the application
streamlit run app.py
```

---

## 📋 Detailed Steps

### 1. Verify Python Installation
```bash
python --version
# Should be Python 3.8 or higher
```

### 2. Check Setup
```bash
python check_setup.py
```
This will show you what's missing.

### 3. Install All Dependencies
```bash
pip install -r requirements.txt
```

**If you get errors**, install individually:
```bash
pip install tensorflow
pip install opencv-python
pip install streamlit
pip install pandas numpy matplotlib scikit-learn
```

### 4. Download FER-2013 Dataset

**Option A: Automated Helper**
```bash
python download_dataset.py
```

**Option B: Manual Download**
1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file
4. Copy `fer2013.csv` to the `data/` folder

**Verify**: Check that `data/fer2013.csv` exists (~308 MB)

### 5. Train the Emotion Detection Model
```bash
python train_emotion_model.py
```

**What happens:**
- Loads FER-2013 dataset
- Trains CNN model for 50 epochs
- Saves model to `model/emotion_model.h5`
- Generates training history plot

**Time**: 30-60 minutes (depending on hardware)

**Output**: 
- `model/emotion_model.h5` (trained model)
- `model/training_history.png` (accuracy/loss graphs)

### 6. Run the Application
```bash
streamlit run app.py
```

**What happens:**
- Starts local web server
- Opens browser automatically
- Shows the application interface

**URL**: http://localhost:8501

---

## 🎮 Using the Application

### Mode 1: Webcam (Real-time)
1. Select "📸 Webcam (Real-time)" from sidebar
2. Click "Start Camera" checkbox
3. Allow camera access when prompted
4. Show different emotions to the camera
5. View real-time emotion detection and music recommendations

### Mode 2: Upload Image
1. Select "🖼️ Upload Image" from sidebar
2. Click "Choose an image..."
3. Upload a photo with a clear face
4. View detected emotion and recommendations

### Mode 3: About
- View project information
- Learn how the system works
- See supported emotions

---

## 🧪 Testing Individual Components

### Test Emotion Recognition Only
```bash
python emotion_recognition.py
```
- Opens webcam
- Shows real-time emotion detection
- Press 'q' to quit

### Test Music Recommender Only
```bash
python music_recommender.py
```
- Displays sample recommendations for each emotion
- No webcam needed

### Verify Complete Setup
```bash
python check_setup.py
```
- Checks Python version
- Verifies all dependencies
- Checks dataset and model
- Tests webcam access

---

## 🔧 Troubleshooting

### "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "Could not open webcam"
- Close other apps using camera (Zoom, Teams, etc.)
- Try "Upload Image" mode instead
- Check camera permissions

### "Dataset not found"
- Ensure `fer2013.csv` is in `data/` folder
- Check file name (case-sensitive)
- Verify file size (~308 MB)

### "Model not found"
- Run `python train_emotion_model.py` first
- Wait for training to complete
- Check `model/emotion_model.h5` exists

### Training is slow
- Normal on CPU (30-60 minutes)
- Use GPU if available (much faster)
- Reduce epochs in code if needed

### Out of memory during training
- Close other applications
- Reduce batch size in `train_emotion_model.py`
- Use computer with more RAM

---

## 📂 File Overview

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application |
| `train_emotion_model.py` | Train the CNN model |
| `emotion_recognition.py` | Emotion detection module |
| `music_recommender.py` | Music recommendation logic |
| `spotify_integration.py` | Optional Spotify API |
| `check_setup.py` | Verify installation |
| `quick_start.py` | Automated setup |
| `requirements.txt` | Python dependencies |

---

## 📚 Documentation Files

| File | Content |
|------|---------|
| `README.md` | Complete project documentation |
| `SETUP_GUIDE.md` | Detailed installation guide |
| `VIVA_GUIDE.md` | Viva/presentation preparation |
| `PROJECT_SUMMARY.md` | Project overview |
| `GETTING_STARTED.md` | This file |

---

## 💡 Tips

### Before Demo/Presentation:
✅ Train model beforehand (don't train during demo)
✅ Test webcam works
✅ Have backup images ready
✅ Close unnecessary apps
✅ Charge laptop fully

### For Best Results:
✅ Good lighting
✅ Face camera directly
✅ Clear facial expressions
✅ No obstructions (masks, hands)

---

## 🆘 Need Help?

1. **Check documentation**: Read README.md and SETUP_GUIDE.md
2. **Run diagnostics**: `python check_setup.py`
3. **Test components**: Run individual scripts
4. **Check errors**: Read error messages carefully
5. **Google it**: Search for specific error messages

---

## 📞 Quick Commands Reference

```bash
# Verify setup
python check_setup.py

# Download dataset helper
python download_dataset.py

# Train model
python train_emotion_model.py

# Run application
streamlit run app.py

# Test emotion recognition
python emotion_recognition.py

# Test music recommender
python music_recommender.py

# Automated setup
python quick_start.py
```

---

## ✅ Checklist

Before running the application, ensure:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] FER-2013 dataset in `data/fer2013.csv`
- [ ] Model trained (`model/emotion_model.h5` exists)
- [ ] Webcam accessible (or have test images ready)

---

## 🎉 You're Ready!

Once all steps are complete:
```bash
streamlit run app.py
```

Enjoy your Emotion-Based Music Recommendation System! 🎵

---

**Need more details?** Check the other documentation files:
- `README.md` - Complete documentation
- `SETUP_GUIDE.md` - Detailed setup instructions
- `VIVA_GUIDE.md` - Presentation preparation
- `PROJECT_SUMMARY.md` - Project overview

