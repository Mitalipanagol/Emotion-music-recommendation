"""
Real-time Emotion Recognition Module
Uses trained CNN model and OpenCV for webcam-based emotion detection
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class EmotionRecognizer:
    """Class for real-time emotion recognition from webcam feed"""
    
    def __init__(self, model_path="model/emotion_model.h5"):
        """Initialize the emotion recognizer"""
        
        # Emotion labels (must match training order)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load the trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using train_emotion_model.py"
            )
        
        print(f"📦 Loading emotion detection model from {model_path}...")
        self.model = load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        
        print("✅ Face detector initialized")
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
    
    def predict_emotion(self, face_roi):
        """Predict emotion from a face ROI"""
        # Resize to model input size
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Reshape and normalize
        face_normalized = face_resized.reshape(1, 48, 48, 1) / 255.0
        
        # Predict
        prediction = self.model.predict(face_normalized, verbose=0)
        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx]
        
        return self.emotion_labels[emotion_idx], confidence
    
    def detect_emotion(self, frame):
        """
        Detect emotion from a frame
        Returns: annotated frame and detected emotion
        """
        faces, gray = self.detect_faces(frame)
        
        detected_emotion = "Neutral"
        max_confidence = 0.0
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_roi)
            
            # Keep track of the most confident detection
            if confidence > max_confidence:
                detected_emotion = emotion
                max_confidence = confidence
            
            # Draw rectangle around face
            color = self._get_emotion_color(emotion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display emotion and confidence
            label = f"{emotion}: {confidence*100:.1f}%"
            cv2.putText(
                frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
        
        return frame, detected_emotion
    
    def _get_emotion_color(self, emotion):
        """Get color for each emotion"""
        color_map = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
            'Surprise': (255, 255, 0), # Cyan
            'Fear': (128, 0, 128),     # Purple
            'Disgust': (0, 128, 128),  # Olive
            'Neutral': (128, 128, 128) # Gray
        }
        return color_map.get(emotion, (255, 255, 255))

def test_webcam():
    """Test emotion recognition with webcam"""
    print("\n🎥 Starting webcam test...")
    print("Press 'q' to quit\n")
    
    recognizer = EmotionRecognizer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        # Detect emotion
        annotated_frame, emotion = recognizer.detect_emotion(frame)
        
        # Display current emotion
        cv2.putText(
            annotated_frame, f"Current Emotion: {emotion}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        # Show frame
        cv2.imshow('Emotion Recognition - Press Q to quit', annotated_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam test completed")

if __name__ == "__main__":
    test_webcam()

