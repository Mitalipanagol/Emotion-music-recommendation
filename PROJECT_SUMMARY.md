# 🎵 Emotion-Based Music Recommendation System

## Project Summary

### 📌 Project Information
- **Title**: Emotion-Based Music Recommendation System using Deep Learning
- **Domain**: Artificial Intelligence, Computer Vision, Machine Learning
- **Level**: Final Year Project
- **Technologies**: Python, TensorFlow, OpenCV, Streamlit

---

## 🎯 Objective

To develop an intelligent system that:
1. Detects human emotions from facial expressions in real-time
2. Recommends personalized music based on detected emotions
3. Provides an interactive web-based user interface
4. Helps improve user mood through appropriate music suggestions

---

## 🔬 Methodology

### 1. Data Collection
- **Dataset**: FER-2013 (Facial Expression Recognition 2013)
- **Source**: Kaggle
- **Size**: 35,887 grayscale images
- **Resolution**: 48x48 pixels
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

### 2. Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Layers**: 
  - 3 Convolutional blocks (32, 64, 128 filters)
  - Batch Normalization layers
  - MaxPooling layers
  - Dropout layers (regularization)
  - 2 Dense layers (512, 256 neurons)
  - Output layer (7 classes, Softmax)

### 3. Training Process
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Cross-Entropy
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 64
- **Validation Split**: 20%
- **Callbacks**: Early Stopping, Learning Rate Reduction

### 4. Face Detection
- **Algorithm**: Haar Cascade Classifier
- **Library**: OpenCV
- **Purpose**: Detect and extract face regions from images/video

### 5. Music Recommendation
- **Approach**: Emotion-to-Music mapping
- **Categories**: Genres, Playlists, Songs
- **Optional**: Spotify API integration

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Programming Language | Python 3.8+ |
| Deep Learning Framework | TensorFlow/Keras |
| Computer Vision | OpenCV |
| Web Framework | Streamlit |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Model Evaluation | scikit-learn |
| Optional API | Spotipy (Spotify) |

---

## 📁 Project Structure

```
emotion-music-recommender/
│
├── model/                          # Trained models
│   ├── emotion_model.h5           # CNN model
│   └── training_history.png       # Training graphs
│
├── data/                          # Dataset
│   └── fer2013.csv               # FER-2013 dataset
│
├── test_images/                   # Test images
│   └── README.md
│
├── app.py                        # Main Streamlit application
├── train_emotion_model.py        # Model training script
├── emotion_recognition.py        # Emotion detection module
├── music_recommender.py          # Music recommendation logic
├── spotify_integration.py        # Optional Spotify API
│
├── check_setup.py                # Setup verification
├── quick_start.py                # Automated setup
├── download_dataset.py           # Dataset download helper
│
├── requirements.txt              # Dependencies
├── README.md                     # Main documentation
├── SETUP_GUIDE.md               # Installation guide
├── VIVA_GUIDE.md                # Viva preparation
└── PROJECT_SUMMARY.md           # This file
```

---

## 🚀 How to Run

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FER-2013 dataset
python download_dataset.py

# 3. Train the model
python train_emotion_model.py

# 4. Run the application
streamlit run app.py
```

### Alternative: Automated Setup
```bash
python quick_start.py
```

---

## 📊 Results

### Model Performance
- **Training Accuracy**: 70-75%
- **Validation Accuracy**: 65-70%
- **Test Accuracy**: 65-70%
- **Real-time FPS**: 15-30 fps

### Confusion Matrix
- Best performance: Happy, Surprise
- Moderate performance: Sad, Angry, Neutral
- Challenging: Fear, Disgust (fewer samples)

---

## ✨ Features

### Core Features
✅ Real-time emotion detection from webcam
✅ Image upload mode for static images
✅ 7 emotion categories
✅ Music recommendations (genres, playlists, songs)
✅ Interactive web interface
✅ Visual feedback with confidence scores
✅ Color-coded emotion display

### Optional Features
✅ Spotify API integration
✅ Setup verification tools
✅ Automated installation scripts
✅ Comprehensive documentation

---

## 🎓 Learning Outcomes

### Technical Skills
- Deep Learning model development
- Computer Vision techniques
- Real-time video processing
- Web application development
- API integration
- Model deployment

### Concepts Covered
- Convolutional Neural Networks
- Image preprocessing
- Transfer learning concepts
- Overfitting prevention
- Model evaluation metrics
- User interface design

---

## 🔍 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Long training time | Used efficient architecture, batch normalization |
| Overfitting | Dropout, early stopping, regularization |
| Class imbalance | Can use weighted loss, data augmentation |
| Real-time performance | Optimized face detection, efficient model |
| Webcam compatibility | Fallback to image upload mode |
| User experience | Streamlit for intuitive interface |

---

## 🚧 Limitations

1. **Lighting Sensitivity**: Performance degrades in poor lighting
2. **Pose Variation**: Best with frontal faces
3. **Occlusions**: Masks, glasses affect accuracy
4. **Single Emotion**: Detects one dominant emotion
5. **Cultural Bias**: Dataset primarily Western faces
6. **Resolution**: Limited to 48x48 input
7. **Grayscale**: No color information used

---

## 🔮 Future Enhancements

### Short-term
- [ ] Data augmentation for better accuracy
- [ ] Real-time emotion tracking over time
- [ ] Multiple face detection
- [ ] Improved UI/UX

### Long-term
- [ ] Multi-modal emotion detection (voice + face)
- [ ] Personalized recommendations based on history
- [ ] Mobile application (Android/iOS)
- [ ] Integration with music streaming services
- [ ] Emotion analytics dashboard
- [ ] Cloud deployment
- [ ] Advanced models (Transformer-based)

---

## 📚 References

### Datasets
1. FER-2013: https://www.kaggle.com/datasets/msambare/fer2013

### Research Papers
1. "Challenges in Representation Learning: Facial Expression Recognition Challenge" (2013)
2. "Deep Learning for Facial Expression Recognition: A Survey" (2020)
3. "Emotion Recognition using Convolutional Neural Networks" (2018)

### Libraries & Frameworks
1. TensorFlow: https://www.tensorflow.org/
2. OpenCV: https://opencv.org/
3. Streamlit: https://streamlit.io/
4. Keras: https://keras.io/

---

## 👥 Use Cases

1. **Mental Health**: Mood tracking and music therapy
2. **Entertainment**: Personalized music streaming
3. **Automotive**: Driver mood detection
4. **Education**: Student engagement monitoring
5. **Healthcare**: Patient emotion monitoring
6. **Marketing**: Customer emotion analysis
7. **Gaming**: Adaptive game difficulty

---

## 📝 Conclusion

This project successfully demonstrates:
- Integration of Computer Vision and Deep Learning
- Real-time emotion detection with acceptable accuracy
- Practical application of AI in daily life
- End-to-end ML pipeline (data → model → deployment)
- User-friendly interface for non-technical users

The system provides a foundation for emotion-aware applications and can be extended with additional features and improvements.

---

## 📞 Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `SETUP_GUIDE.md` for installation help
3. See `VIVA_GUIDE.md` for presentation preparation
4. Run `python check_setup.py` to verify setup

---

**Project Status**: ✅ Complete and Ready for Demonstration

**Last Updated**: December 2025

