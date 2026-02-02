"""
Emotion-Based Music Recommendation System - Streamlit Web Application
Main application interface for real-time emotion detection and music recommendations
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from emotion_recognition import EmotionRecognizer
from music_recommender import MusicRecommender
import os

# Page configuration
st.set_page_config(
    page_title="Emotion-Based Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UX design
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #b3b3b3;
        margin-top: 0;
        margin-bottom: 30px;
    }

    /* Emotion display box */
    .emotion-box {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .emotion-text {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 0;
    }

    /* Music recommendations section */
    .music-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .section-title {
        color: #1DB954;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }

    .genre-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .playlist-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 12px 20px;
        margin: 8px 0;
        border-radius: 10px;
        border-left: 4px solid #1DB954;
        color: white;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .playlist-item:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(10px);
    }

    .song-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 15px;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border-left: 5px solid #1DB954;
    }

    .song-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }

    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(29, 185, 84, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(29, 185, 84, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load emotion recognizer and music recommender (cached)"""
    try:
        recognizer = EmotionRecognizer()
        recommender = MusicRecommender()
        return recognizer, recommender, None
    except Exception as e:
        return None, None, str(e)

def main():
    # Header
    st.markdown('<p class="main-header">🎵 Emotion-Based Music Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect your emotion and get personalized music recommendations</p>', unsafe_allow_html=True)
    
    # Load models
    recognizer, recommender, error = load_models()
    
    if error:
        st.error(f"❌ Error loading models: {error}")
        st.info("📝 Please make sure you have trained the model first by running: `python train_emotion_model.py`")
        return
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown('<h1 style="color: white; text-align: center;">⚙️ Settings</h1>', unsafe_allow_html=True)
        st.markdown("---")

        app_mode = st.selectbox(
            "🎯 Choose Mode",
            ["📸 Webcam (Real-time)", "🖼️ Upload Image", "ℹ️ About"]
        )

        st.markdown("---")
        st.markdown("""
        <div style="color: white; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 20px;">
            <h3 style="color: #1DB954; text-align: center;">🎭 Emotions</h3>
            <p style="font-size: 0.9rem;">
            😊 Happy<br>
            😢 Sad<br>
            😠 Angry<br>
            😨 Fear<br>
            😲 Surprise<br>
            😐 Neutral<br>
            🤢 Disgust
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="color: white; padding: 15px; background: rgba(29, 185, 84, 0.2); border-radius: 10px; margin-top: 20px;">
            <h3 style="color: #1DB954; text-align: center;">📊 Model Info</h3>
            <p style="font-size: 0.9rem; text-align: center;">
            """ + ("✅ Model Loaded<br>🧠 CNN Architecture<br>📸 Real-time Detection" if os.path.exists("model/emotion_model.h5") else "❌ Model Not Found") + """
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on mode
    if app_mode == "📸 Webcam (Real-time)":
        webcam_mode(recognizer, recommender)
    elif app_mode == "🖼️ Upload Image":
        upload_mode(recognizer, recommender)
    else:
        about_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #888;">
        <p style="margin: 5px;">Made with ❤️ using Deep Learning & Computer Vision</p>
        <p style="margin: 5px; font-size: 0.9rem;">
            🧠 CNN Model | 📸 OpenCV | 🎵 Music Recommendation | ⚡ Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

def webcam_mode(recognizer, recommender):
    """Real-time webcam emotion detection mode"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
        <h1 style="color: white; margin: 0;">📸 Real-time Emotion Detection</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">
            Let AI detect your emotions in real-time and get personalized music!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h3 style="color: #1DB954;">📹 Camera Feed</h3>', unsafe_allow_html=True)
        run = st.checkbox("🎥 Start Camera", key="webcam_run")
        frame_placeholder = st.empty()

    with col2:
        st.markdown('<h3 style="color: #1DB954;">🎭 Detected Emotion</h3>', unsafe_allow_html=True)
        emotion_placeholder = st.empty()

        st.markdown('<h3 style="color: #1DB954; margin-top: 30px;">🎵 Music Recommendations</h3>', unsafe_allow_html=True)
        music_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera permissions.")
            return
        
        current_emotion = "Neutral"
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame")
                break
            
            # Detect emotion
            annotated_frame, detected_emotion = recognizer.detect_emotion(frame)
            
            if detected_emotion:
                current_emotion = detected_emotion
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Display emotion with enhanced styling
            emotion_placeholder.markdown(
                f'<div class="emotion-box"><p class="emotion-text">{current_emotion}</p></div>',
                unsafe_allow_html=True
            )
            
            # Get and display music recommendations
            recommendations = recommender.recommend_music(current_emotion)
            display_recommendations(music_placeholder, recommendations)
            
            time.sleep(0.1)  # Small delay to reduce CPU usage
        
        cap.release()
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px; border-radius: 15px; text-align: center; margin-top: 20px;">
            <h2 style="color: white; margin-bottom: 15px;">🎥 Ready to Start?</h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                Click the checkbox above to start your camera and begin real-time emotion detection!
            </p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 15px;">
                💡 Tip: Make sure your face is well-lit and clearly visible for best results
            </p>
        </div>
        """, unsafe_allow_html=True)

def upload_mode(recognizer, recommender):
    """Image upload emotion detection mode"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
        <h1 style="color: white; margin: 0;">🖼️ Upload Image for Emotion Detection</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">
            Upload a photo and let AI analyze the emotions!
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎭 Emotion Detection")
            
            # Detect emotion
            annotated_frame, detected_emotion = recognizer.detect_emotion(image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_rgb, use_container_width=True)
            
            st.markdown(
                f'<div class="emotion-box"><p class="emotion-text">{detected_emotion}</p></div>',
                unsafe_allow_html=True
            )

        # Music recommendations
        st.markdown("---")
        recommendations = recommender.recommend_music(detected_emotion)
        display_recommendations(st, recommendations)

def display_recommendations(placeholder, recommendations):
    """Display music recommendations with enhanced styling"""
    # Build the complete HTML content with custom styling
    content = '<div class="music-section">'

    # Genres with badges
    content += '<p style="color: white; font-size: 1.2rem; margin-bottom: 10px;"><strong>🎸 Genres</strong></p>'
    content += '<div style="margin-bottom: 20px;">'
    for genre in recommendations['genres']:
        content += f'<span class="genre-badge">{genre}</span>'
    content += '</div>'

    # Playlists
    content += '<p style="color: white; font-size: 1.2rem; margin-bottom: 10px;"><strong>📻 Playlists</strong></p>'
    for playlist in recommendations['playlists']:
        content += f'<div class="playlist-item">▶️ {playlist}</div>'

    # Songs
    content += '<p style="color: white; font-size: 1.2rem; margin-top: 20px; margin-bottom: 10px;"><strong>🎵 Recommended Songs</strong></p>'
    for song in recommendations['songs']:
        content += f'<div class="song-item">🎶 {song}</div>'

    content += '</div>'

    # Display using the placeholder
    placeholder.markdown(content, unsafe_allow_html=True)

def about_page():
    """About page with project information"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This is an **Emotion-Based Music Recommendation System** that uses:
    - **Deep Learning (CNN)** for facial emotion recognition
    - **Computer Vision (OpenCV)** for real-time face detection
    - **FER-2013 Dataset** for model training
    
    ### 🔧 How It Works
    1. **Face Detection**: Uses Haar Cascade to detect faces in real-time
    2. **Emotion Recognition**: CNN model predicts emotion from facial features
    3. **Music Recommendation**: Maps detected emotion to appropriate music
    
    ### 🎭 Supported Emotions
    - Happy 😊
    - Sad 😢
    - Angry 😠
    - Fear 😨
    - Surprise 😲
    - Neutral 😐
    - Disgust 🤢
    
    ### 📚 Technologies Used
    - TensorFlow/Keras for deep learning
    - OpenCV for computer vision
    - Streamlit for web interface
    - Python for backend logic
    
    ### 🚀 Getting Started
    1. Train the model: `python train_emotion_model.py`
    2. Run the app: `streamlit run app.py`
    3. Allow camera access when prompted
    
    ### 👨‍💻 Developer
    Final Year Project - Emotion-Based Music Recommendation System
    """)

if __name__ == "__main__":
    main()

