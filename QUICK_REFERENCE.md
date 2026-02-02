# Quick Reference Guide
## Emotion-Based Music Recommendation System

---

## 🚀 QUICK START (5 Minutes)

### Step 1: Install Dependencies (1 min)
```bash
pip install tensorflow opencv-python streamlit numpy
```

### Step 2: Download Dataset (Already Done ✅)
```
data/train/ - 28,709 images (7 emotion folders)
data/test/  - 7,178 images (7 emotion folders)
```

### Step 3: Train Model (2-3 min)
```bash
python train_minimal.py
```
**Output:** `model/emotion_model.h5` (60-65% accuracy)

### Step 4: Run Application (1 min)
```bash
streamlit run app.py
```
**URL:** http://localhost:8501

---

## 📊 MINIMAL DATASET APPROACH

### Why Minimal?
- ✅ **Fast Training**: 2-3 minutes (vs 2-3 hours for full dataset)
- ✅ **Good Accuracy**: 60-65% (vs 70% for full dataset)
- ✅ **Perfect for Demo**: Quick to show and test
- ✅ **Scalable**: Can easily switch to full dataset

### Dataset Size
| Type | Full Dataset | Minimal Dataset |
|------|--------------|-----------------|
| Training | 28,709 images | 700 images (100/class) |
| Testing | 7,178 images | 350 images (50/class) |
| Training Time | 2-3 hours | 2-3 minutes |
| Accuracy | 70% | 60-65% |

---

## 🏗️ ARCHITECTURE OVERVIEW

### Model Architecture (CNN)
```
Input (48x48x1)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(7, softmax) → [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

---

## 🎯 KEY COMPONENTS

### 1. Face Detection
- **Method**: Haar Cascade Classifier
- **File**: `haarcascade_frontalface_default.xml`
- **Parameters**: scaleFactor=1.3, minNeighbors=5

### 2. Emotion Recognition
- **Input**: 48x48 grayscale face image
- **Output**: 7 emotion probabilities
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 3. Music Recommendation
- **Mapping**: Emotion → Genres, Playlists, Songs
- **Example**: Happy → Pop, Dance, Upbeat songs

---

## 📁 FILE STRUCTURE

```
project/
├── data/
│   ├── train/              # Training images
│   └── test/               # Test images
├── model/
│   └── emotion_model.h5    # Trained CNN model
├── train_minimal.py        # Training script (minimal dataset)
├── emotion_recognition.py  # Face detection + emotion prediction
├── music_recommender.py    # Music recommendation logic
├── app.py                  # Streamlit web application
└── requirements.txt        # Python dependencies
```

---

## 🔧 TROUBLESHOOTING

### Issue 1: Model Not Found
**Error:** `FileNotFoundError: model/emotion_model.h5`
**Solution:** Run `python train_minimal.py` first

### Issue 2: Low Accuracy
**Problem:** Model predicting only Neutral/Angry
**Solution:** 
- Check if model trained properly (should be 60-65%)
- Retrain with: `python train_minimal.py`

### Issue 3: Webcam Not Working
**Problem:** Camera not detected
**Solution:** 
- Use "Upload Image" mode instead
- Check camera permissions

### Issue 4: Streamlit Not Starting
**Problem:** Port already in use
**Solution:** 
```bash
streamlit run app.py --server.port 8502
```

---

## 📈 PERFORMANCE METRICS

### Model Performance
- **Training Accuracy**: 65-70%
- **Validation Accuracy**: 60-65%
- **Test Accuracy**: 60-65%
- **Inference Time**: ~50ms per image

### Per-Class Accuracy (Expected)
| Emotion | Accuracy |
|---------|----------|
| Happy | 75-80% |
| Sad | 60-65% |
| Angry | 65-70% |
| Neutral | 60-65% |
| Surprise | 55-60% |
| Fear | 50-55% |
| Disgust | 45-50% |

---

## 🎓 VIVA PREPARATION - KEY POINTS

### Q1: Why minimal dataset?
**A:** For quick demonstration and testing. Full dataset takes 2-3 hours to train, minimal takes 2-3 minutes with acceptable 60-65% accuracy.

### Q2: What is the model architecture?
**A:** CNN with 3 convolutional blocks (32→64→128 filters), BatchNormalization, Dropout regularization, and 2 dense layers outputting 7 emotion classes.

### Q3: How does face detection work?
**A:** Using Haar Cascade Classifier from OpenCV, which detects faces using edge features. Then we extract the face ROI and pass it to the CNN.

### Q4: What is the accuracy?
**A:** 60-65% with minimal dataset (700 images). Can achieve 70%+ with full dataset (28,709 images).

### Q5: How does music recommendation work?
**A:** Simple rule-based mapping: Each emotion maps to specific genres, playlists, and songs. For example, Happy → Pop/Dance, Sad → Chill/Lo-fi.

### Q6: What preprocessing is done?
**A:** 
1. Convert to grayscale
2. Resize to 48x48
3. Normalize pixel values (0-1)
4. Reshape to (1, 48, 48, 1)

### Q7: What are the challenges?
**A:**
- Class imbalance (Disgust has fewer samples)
- Similar emotions (Sad vs Neutral)
- Lighting conditions affect detection
- Face angle affects accuracy

---

## 💡 IMPROVEMENTS (Future Work)

### Short-term
- ✅ Data augmentation (rotation, flip, zoom)
- ✅ Use full dataset for better accuracy
- ✅ Add more music sources (Spotify API)

### Long-term
- ✅ Transfer learning (VGG16, ResNet)
- ✅ Real-time video emotion tracking
- ✅ Multi-face detection
- ✅ Emotion history tracking

---

## 📝 COMMANDS CHEAT SHEET

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (minimal)
python train_minimal.py

# Train model (full dataset) - if needed
python train_better_model.py

# Run application
streamlit run app.py

# Test emotion recognition only
python emotion_recognition.py

# Check dataset
python check_dataset.py
```

---

## 🎯 DEMO FLOW

1. **Open Application** → http://localhost:8501
2. **Choose Mode** → Upload Image or Webcam
3. **Capture/Upload** → Your face image
4. **View Results** → Detected emotion + confidence
5. **Get Recommendations** → Genres, Playlists, Songs
6. **Try Different Emotions** → Happy, Sad, Angry, etc.

---

**Quick Reference Complete!**
Use `IMPLEMENTATION_GUIDE_MINIMAL.md` for detailed step-by-step implementation.
