# 📸 Test Images Guide for Screenshots

This guide will help you get test images for all 7 emotions to take screenshots of your application.

## 🎯 Quick Download Links

### Option 1: Download Sample Images from FER-2013 Dataset

Since you already have the FER-2013 dataset, I'll create a script to copy sample images for you.

### Option 2: Free Stock Photo Websites

#### 1. **Happy** 😊
- Search: "happy face", "smiling person", "joyful expression"
- Suggested sites:
  - [Unsplash - Happy](https://unsplash.com/s/photos/happy-face)
  - [Pexels - Smiling](https://www.pexels.com/search/smiling%20face/)
  - [Pixabay - Joy](https://pixabay.com/images/search/happy%20face/)

#### 2. **Sad** 😢
- Search: "sad face", "crying person", "depressed expression"
- Suggested sites:
  - [Unsplash - Sad](https://unsplash.com/s/photos/sad-face)
  - [Pexels - Sad](https://www.pexels.com/search/sad%20face/)
  - [Pixabay - Sad](https://pixabay.com/images/search/sad%20face/)

#### 3. **Angry** 😠
- Search: "angry face", "furious person", "mad expression"
- Suggested sites:
  - [Unsplash - Angry](https://unsplash.com/s/photos/angry-face)
  - [Pexels - Angry](https://www.pexels.com/search/angry%20face/)
  - [Pixabay - Angry](https://pixabay.com/images/search/angry%20face/)

#### 4. **Neutral** 😐
- Search: "neutral face", "expressionless", "poker face"
- Suggested sites:
  - [Unsplash - Neutral](https://unsplash.com/s/photos/neutral-face)
  - [Pexels - Neutral](https://www.pexels.com/search/neutral%20face/)
  - [Pixabay - Neutral](https://pixabay.com/images/search/neutral%20expression/)

#### 5. **Surprise** 😮
- Search: "surprised face", "shocked person", "amazed expression"
- Suggested sites:
  - [Unsplash - Surprise](https://unsplash.com/s/photos/surprised-face)
  - [Pexels - Surprise](https://www.pexels.com/search/surprised%20face/)
  - [Pixabay - Surprise](https://pixabay.com/images/search/surprised%20face/)

#### 6. **Fear** 😨
- Search: "fearful face", "scared person", "frightened expression"
- Suggested sites:
  - [Unsplash - Fear](https://unsplash.com/s/photos/scared-face)
  - [Pexels - Fear](https://www.pexels.com/search/scared%20face/)
  - [Pixabay - Fear](https://pixabay.com/images/search/fear%20face/)

#### 7. **Disgust** 🤢
- Search: "disgusted face", "grossed out", "repulsed expression"
- Suggested sites:
  - [Unsplash - Disgust](https://unsplash.com/s/photos/disgusted-face)
  - [Pexels - Disgust](https://www.pexels.com/search/disgusted%20face/)
  - [Pixabay - Disgust](https://pixabay.com/images/search/disgust%20face/)

---

## 🚀 Automated Script: Copy Sample Images from Dataset

Run the script below to automatically copy sample images from your FER-2013 dataset:

```bash
python copy_sample_images.py
```

This will create a `test_images/` folder with one sample image for each emotion.

---

## 📋 How to Take Screenshots

### For Each Emotion:

1. **Open the Application**
   ```bash
   streamlit run app.py
   ```

2. **Upload Image Mode**
   - Click "Upload an Image"
   - Select the test image for that emotion
   - Wait for detection
   - Take screenshot (Windows: Win + Shift + S, Mac: Cmd + Shift + 4)

3. **Save Screenshot**
   - Name it: `screenshot_happy.png`, `screenshot_sad.png`, etc.
   - Save in `screenshots/` folder

### Recommended Screenshot Composition:

✅ **Include in Screenshot:**
- Uploaded image with face detection box
- Detected emotion with emoji
- Confidence percentage
- Music recommendations section
- At least 3-4 playlist recommendations

---

## 📁 Organize Your Screenshots

Create this folder structure:
```
screenshots/
├── screenshot_happy.png
├── screenshot_sad.png
├── screenshot_angry.png
├── screenshot_neutral.png
├── screenshot_surprise.png
├── screenshot_fear.png
├── screenshot_disgust.png
└── screenshot_webcam.png  (optional: webcam mode)
```

---

## 🎨 Tips for Better Screenshots

1. **Use High-Quality Images**: Clear face, good lighting
2. **Frontal Face**: Face looking at camera
3. **Single Person**: One face per image
4. **Good Contrast**: Clear facial features
5. **Appropriate Size**: At least 200x200 pixels

---

## 🔧 Troubleshooting

**Q: Image not detecting face?**
- Use frontal face images
- Ensure good lighting
- Try different image

**Q: Wrong emotion detected?**
- Model accuracy is 38-42% (minimal dataset)
- Try multiple images
- Use clear expressions

**Q: No music recommendations?**
- Check if emotion is detected
- Verify music_recommender.py is working

---

## 📊 Sample Images Characteristics

| Emotion  | Key Features to Look For |
|----------|--------------------------|
| Happy    | Smile, raised cheeks, crinkled eyes |
| Sad      | Downturned mouth, droopy eyes |
| Angry    | Furrowed brows, tight lips, intense eyes |
| Neutral  | Relaxed face, no strong expression |
| Surprise | Wide eyes, open mouth, raised eyebrows |
| Fear     | Wide eyes, tense face, raised eyebrows |
| Disgust  | Wrinkled nose, raised upper lip |


