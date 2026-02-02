# 🎵 Emotion-Based Music Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based system that detects emotions from facial expressions using Computer Vision and recommends personalized music based on the detected emotion.

![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project combines **Deep Learning**, **Computer Vision**, and **Music Recommendation** to create an intelligent system that:

1. 📹 **Detects faces** in real-time using webcam or uploaded images
2. 😊 **Recognizes emotions** from facial expressions (7 emotions)
3. 🎵 **Recommends music** playlists based on detected emotions

**Supported Emotions**: Happy 😊, Sad 😢, Angry 😠, Neutral 😐, Surprise 😮, Fear 😨, Disgust 🤢

---

## ✨ Features

- ✅ **Real-time Emotion Detection** - Live webcam emotion recognition
- ✅ **Image Upload Mode** - Analyze emotions from uploaded photos
- ✅ **7 Emotion Classes** - Comprehensive emotion recognition
- ✅ **Music Recommendations** - Curated Spotify playlists for each emotion
- ✅ **Modern UI** - Beautiful Streamlit interface with gradients and animations
- ✅ **Face Detection** - Automatic face detection using Haar Cascade
- ✅ **Confidence Scores** - Shows prediction confidence for transparency
- ✅ **Lightweight Model** - Fast inference (~35-60ms per frame)

---

## 🎬 Demo

### Webcam Mode
Real-time emotion detection from your webcam feed with instant music recommendations.

### Upload Mode
Upload any image with a face to detect emotion and get personalized music suggestions.

---

## 🛠️ Technology Stack

### Deep Learning & AI
- **TensorFlow 2.20.0** - Deep learning framework
- **Keras 3.13.0** - High-level neural networks API
- **OpenCV 4.10.0** - Computer vision library

### Web Application
- **Streamlit 1.41.1** - Web application framework
- **Pillow 12.0.0** - Image processing

### Data Processing
- **NumPy 2.2.6** - Numerical computing
- **Pandas 2.3.3** - Data manipulation
- **Scikit-learn 1.6.2** - Machine learning utilities

### Visualization
- **Matplotlib 3.10.8** - Plotting library
- **Seaborn 0.13.2** - Statistical visualization

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam (optional, for real-time detection)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/emotion-music-recommendation.git
cd emotion-music-recommendation
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset (Optional - for training)
```bash
python download_dataset.py
```

### Step 5: Train Model (Optional - pre-trained model included)
```bash
# Quick training with minimal dataset (2-3 minutes)
python train_minimal.py

# Full training with complete dataset (30-60 minutes)
python train_emotion_model.py
```

---

## 🚀 Usage

### Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using Webcam Mode
1. Click **"Use Webcam"** tab
2. Click **"Start Webcam"** button
3. Allow camera access when prompted
4. Make facial expressions to see real-time detection
5. View music recommendations based on your emotion

---

## 📁 Project Structure

```
emotion-music-recommendation/
│
├── app.py                          # Main Streamlit application
├── emotion_recognition.py          # Emotion detection module
├── music_recommender.py            # Music recommendation logic
├── requirements.txt                # Python dependencies
│
├── model/                          # Trained models
│   ├── emotion_model.h5           # Pre-trained CNN model
│   └── confusion_matrix.png       # Model evaluation results
│
├── data/                           # Dataset (download separately)
│   ├── train/                     # Training images
│   └── test/                      # Test images
│
├── test_images/                    # Sample images for testing
│   ├── sample_happy.jpg
│   ├── sample_sad.jpg
│   └── ... (7 emotions)
│
├── screenshots/                    # Application screenshots
│
├── Training Scripts/
│   ├── train_minimal.py           # Quick training (3,500 images)
│   ├── train_emotion_model.py     # Full training (28,000 images)
│   └── evaluate_model.py          # Model evaluation
│
├── Documentation/
│   ├── README.md                  # Main documentation
│   ├── MODEL_PERFORMANCE_METRICS.md
│   ├── IMPLEMENTATION_GUIDE_MINIMAL.md
│   ├── VIVA_QUESTIONS_AND_ANSWERS.md
│   └── SCREENSHOT_GUIDE.md
│
└── Utilities/
    ├── download_dataset.py        # Download FER-2013 dataset
    ├── copy_sample_images.py      # Extract sample images
    └── check_setup.py             # Verify installation

```

---

## 🔧 How It Works

### 1. Face Detection
- Uses **Haar Cascade Classifier** from OpenCV
- Detects frontal faces in images/video frames
- Draws bounding boxes around detected faces

### 2. Emotion Recognition
- **CNN Architecture**: 3 convolutional blocks + 2 dense layers
- **Input**: 48×48 grayscale face images
- **Output**: Probability distribution over 7 emotions
- **Preprocessing**: Grayscale conversion, resizing, normalization

### 3. Music Recommendation
- Maps detected emotion to music genres
- Provides curated Spotify playlists
- Emotion-to-genre mapping:
  - Happy → Pop, Dance, Feel Good
  - Sad → Acoustic, Melancholic, Slow
  - Angry → Rock, Metal, Intense
  - Neutral → Chill, Ambient, Lofi
  - Surprise → Upbeat, Electronic, Energetic
  - Fear → Calm, Soothing, Peaceful
  - Disgust → Alternative, Indie, Experimental

### 4. Web Interface
- Built with **Streamlit**
- Two modes: Webcam (real-time) and Upload (static)
- Modern UI with CSS animations and gradients
- Displays emotion, confidence, and music recommendations

---

## 🧠 Model Architecture

### CNN Structure

```
Input (48×48×1 grayscale image)
    ↓
Conv2D (32 filters, 3×3) + ReLU + BatchNorm + MaxPool + Dropout(25%)
    ↓
Conv2D (64 filters, 3×3) + ReLU + BatchNorm + MaxPool + Dropout(25%)
    ↓
Conv2D (128 filters, 3×3) + ReLU + BatchNorm + MaxPool + Dropout(25%)
    ↓
Flatten
    ↓
Dense (256 units) + ReLU + BatchNorm + Dropout(50%)
    ↓
Dense (7 units) + Softmax
    ↓
Output (7 emotion probabilities)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Categorical Cross-Entropy |
| **Batch Size** | 32 |
| **Epochs** | 50 (with early stopping) |
| **Regularization** | Dropout (25%, 50%), Batch Normalization |
| **Callbacks** | Early Stopping, ReduceLROnPlateau |
| **Total Parameters** | 1,276,295 |
| **Model Size** | 4.87 MB |

---

## 📊 Dataset

### FER-2013 (Facial Expression Recognition 2013)

- **Source**: Kaggle FER-2013 Challenge
- **Total Images**: 35,887
  - Training: 28,709 images
  - Test: 7,178 images
- **Image Size**: 48×48 pixels (grayscale)
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Format**: CSV file with pixel values

### Download Dataset

```bash
# Option 1: Using provided script
python download_dataset.py

# Option 2: Manual download from Kaggle
# Visit: https://www.kaggle.com/datasets/msambare/fer2013
# Download and extract to data/ folder
```

### Dataset Distribution

| Emotion | Training Samples | Test Samples |
|---------|-----------------|--------------|
| Happy | 8,989 | 1,774 |
| Neutral | 6,198 | 1,233 |
| Sad | 6,077 | 1,247 |
| Angry | 4,953 | 958 |
| Surprise | 4,002 | 831 |
| Fear | 5,121 | 1,024 |
| Disgust | 547 | 111 |

---

## 🚀 Future Enhancements

### Short-term
- [ ] Improve model accuracy to 65-70% with full dataset training
- [ ] Add data augmentation (rotation, flipping, brightness)
- [ ] Implement real-time FPS counter
- [ ] Add emotion history tracking over time
- [ ] Export emotion logs to CSV

### Long-term
- [ ] Integrate Spotify API for direct playlist playback
- [ ] Add multi-face detection and emotion recognition
- [ ] Implement transfer learning with VGG16/ResNet
- [ ] Create mobile app version (React Native)
- [ ] Add voice-based emotion detection
- [ ] Implement emotion-based video recommendations
- [ ] Deploy to cloud (Heroku, AWS, or Google Cloud)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- Improve model accuracy
- Add new music streaming integrations
- Enhance UI/UX design
- Add unit tests
- Improve documentation
- Fix bugs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Contact

**Project Maintainer**: [Your Name]

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- **FER-2013 Dataset**: Kaggle and the original dataset creators
- **TensorFlow & Keras**: For the deep learning framework
- **OpenCV**: For computer vision capabilities
- **Streamlit**: For the amazing web framework
- **Spotify**: For music playlist inspiration

---

## 📚 References

1. Goodfellow, I. J., et al. (2013). "Challenges in Representation Learning: A report on three machine learning contests." *Neural Networks*, 64, 59-63.
2. FER-2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013
3. Facial Expression Recognition: https://arxiv.org/abs/1710.07557
4. CNN for Emotion Recognition: https://arxiv.org/abs/1608.01041

---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ and Python**


### Using Upload Mode
1. Click **"Upload an Image"** tab
2. Click **"Browse files"** and select an image
3. View detected emotion and confidence score
4. Get personalized music recommendations

### Test with Sample Images
Sample images are provided in `test_images/` folder for testing all 7 emotions.

---

## 📊 Model Performance

### Metrics (Minimal Dataset - 3,500 images)

| Metric | Score |
|--------|-------|
| **Accuracy** | 38.12% |
| **F1 Score (Macro)** | 38.22% |
| **Precision** | 41.47% |
| **Recall** | 37.69% |

### Per-Emotion Performance

| Emotion | F1 Score | Performance |
|---------|----------|-------------|
| Surprise 😮 | 57.17% | 🏆 Best |
| Happy 😊 | 53.76% | 🥈 Good |
| Disgust 🤢 | 40.21% | 🥉 Average |
| Sad 😢 | 34.03% | Average |
| Angry 😠 | 30.13% | Below Average |
| Neutral 😐 | 27.62% | Poor |
| Fear 😨 | 24.60% | ⚠️ Needs Improvement |

**Note**: Performance can be improved to 65-70% accuracy by training with the full FER-2013 dataset (28,000+ images).


