# 🚀 Quick Setup Guide

## Step-by-Step Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter any issues, try installing packages individually:
```bash
pip install tensorflow==2.15.0
pip install opencv-python
pip install streamlit
pip install pandas numpy matplotlib scikit-learn
```

### 2. Download FER-2013 Dataset

#### Option A: Kaggle (Recommended)
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download" (you may need to create a Kaggle account)
3. Extract the downloaded ZIP file
4. Copy `fer2013.csv` to the `data/` folder in this project

#### Option B: Alternative Source
1. Go to: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
2. Download the dataset
3. Place `fer2013.csv` in the `data/` folder

### 3. Verify Setup
Check that your folder structure looks like this:
```
major project/
├── data/
│   └── fer2013.csv          ← Dataset file here
├── model/                   ← Will contain trained model
├── app.py
├── train_emotion_model.py
├── emotion_recognition.py
├── music_recommender.py
└── requirements.txt
```

### 4. Train the Model
```bash
python train_emotion_model.py
```

**Expected Output**:
- Training will take 30-60 minutes (depending on your hardware)
- You'll see progress bars for each epoch
- Final accuracy should be around 65-75%
- Model will be saved to `model/emotion_model.h5`

**Training Tips**:
- Use a GPU if available (much faster)
- Close other applications to free up RAM
- Don't interrupt the training process

### 5. Run the Application
```bash
streamlit run app.py
```

**Expected Output**:
- Browser will open automatically
- You'll see the Streamlit interface
- Allow camera access when prompted

## 🔧 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**: 
```bash
pip install tensorflow==2.15.0
```

### Issue: "Could not open webcam"
**Solution**: 
- Check if another application is using the camera
- Grant camera permissions to Python/Terminal
- Try using "Upload Image" mode instead

### Issue: "Model not found"
**Solution**: 
- Make sure you've run `python train_emotion_model.py` first
- Check that `model/emotion_model.h5` exists

### Issue: "Dataset not found"
**Solution**: 
- Verify `fer2013.csv` is in the `data/` folder
- Check the file name is exactly `fer2013.csv` (case-sensitive)

### Issue: Training is very slow
**Solution**: 
- Reduce epochs in `train_emotion_model.py` (line 169: change `epochs=50` to `epochs=30`)
- Reduce batch size (line 169: change `batch_size=64` to `batch_size=32`)
- Use a computer with GPU if available

### Issue: "Out of memory" during training
**Solution**: 
- Close other applications
- Reduce batch size to 32 or 16
- Use a computer with more RAM

## 📊 Testing Individual Components

### Test Emotion Recognition Only
```bash
python emotion_recognition.py
```
This will open your webcam and show emotion detection in real-time.
Press 'q' to quit.

### Test Music Recommender Only
```bash
python music_recommender.py
```
This will display sample music recommendations for each emotion.

## 🎯 Quick Start (Without Training)

If you want to test the application without training (for demonstration):

1. Download a pre-trained model (if available from your instructor/team)
2. Place it in `model/emotion_model.h5`
3. Run: `streamlit run app.py`

## 💡 Tips for Final Year Project Presentation

### Before Demo:
1. ✅ Train the model beforehand (don't train during presentation)
2. ✅ Test webcam and ensure it works
3. ✅ Have backup images ready (in case webcam fails)
4. ✅ Close unnecessary applications
5. ✅ Prepare to explain the architecture

### During Demo:
1. Show the web interface
2. Demonstrate real-time emotion detection
3. Show different emotions (happy, sad, angry, etc.)
4. Explain the music recommendations
5. Show the training history plot

### Questions to Prepare For:
- Why did you choose CNN over other algorithms?
- What is the accuracy of your model?
- How does the face detection work?
- What dataset did you use and why?
- What are the limitations?
- What are future improvements?

## 🆘 Need Help?

If you encounter any issues:
1. Check the error message carefully
2. Refer to this troubleshooting guide
3. Check the README.md for more details
4. Google the specific error message
5. Check TensorFlow/OpenCV documentation

## 📞 Common Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_emotion_model.py

# Run web app
streamlit run app.py

# Test emotion recognition
python emotion_recognition.py

# Test music recommender
python music_recommender.py

# Check Python version
python --version

# Check installed packages
pip list
```

---

Good luck with your project! 🎉

