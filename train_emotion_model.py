"""
Emotion Detection Model Training Script
Trains a CNN model on the FER-2013 dataset for facial emotion recognition
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2

def load_fer2013_from_images(data_dir="data"):
    """Load FER-2013 dataset from image folders"""
    print("📂 Loading FER-2013 dataset from images...")

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Check if image folders exist
    if os.path.exists(train_dir):
        print(f"✅ Found image dataset at {train_dir}")
        return load_images_from_folders(train_dir, test_dir)

    # Fallback to CSV if available
    csv_path = os.path.join(data_dir, "fer2013.csv")
    if os.path.exists(csv_path):
        print(f"✅ Found CSV dataset at {csv_path}")
        return load_fer2013_from_csv(csv_path)

    print(f"❌ Error: Dataset not found!")
    print("📥 Please download FER-2013 dataset from:")
    print("   https://www.kaggle.com/datasets/msambare/fer2013")
    return None, None, None, None

def load_images_from_folders(train_dir, test_dir):
    """Load images from folder structure"""
    emotion_labels = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'sad': 4, 'surprise': 5, 'neutral': 6
    }

    def load_from_dir(directory):
        images = []
        labels = []

        for emotion_name, emotion_id in emotion_labels.items():
            emotion_dir = os.path.join(directory, emotion_name)
            if not os.path.exists(emotion_dir):
                continue

            print(f"   Loading {emotion_name} images...")
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        images.append(img)
                        labels.append(emotion_id)
                except Exception as e:
                    continue

        return np.array(images), np.array(labels)

    print("📂 Loading training images...")
    X_train, y_train = load_from_dir(train_dir)

    print("📂 Loading test images...")
    X_test, y_test = load_from_dir(test_dir) if os.path.exists(test_dir) else (np.array([]), np.array([]))

    # Reshape and normalize
    X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
    if len(X_test) > 0:
        X_test = X_test.reshape(-1, 48, 48, 1) / 255.0

    # Convert to categorical
    y_train = to_categorical(y_train, num_classes=7)
    if len(y_test) > 0:
        y_test = to_categorical(y_test, num_classes=7)

    print(f"✅ Loaded {len(X_train)} training images")
    if len(X_test) > 0:
        print(f"✅ Loaded {len(X_test)} test images")

    return X_train, y_train, X_test, y_test

def load_fer2013_from_csv(csv_path):
    """Load and preprocess FER-2013 dataset from CSV"""
    print("📂 Loading FER-2013 dataset from CSV...")

    data = pd.read_csv(csv_path)

    # Convert pixel strings to numpy arrays
    pixels = data['pixels'].tolist()
    faces = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48, 1) for pixel in pixels])

    # Normalize pixel values
    faces = faces / 255.0

    # Convert emotions to categorical
    emotions = to_categorical(data['emotion'], num_classes=7)

    print(f"✅ Loaded {len(faces)} images")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        faces, emotions, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test

def create_emotion_model():
    """Create CNN model for emotion detection"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    return model

def train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=64):
    """Train the emotion detection model"""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    print("\n🚀 Starting model training...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png')
    print("📊 Training history plot saved to model/training_history.png")

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_fer2013_from_images()

    if X_train is None:
        print("\n⚠️  Please download the FER-2013 dataset and place it in the 'data' folder")
        exit(1)

    # If no test set, split from training data
    if X_test is None or len(X_test) == 0:
        print("\n📊 Splitting training data for validation...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    # Create model
    print("\n🏗️  Building CNN model...")
    model = create_emotion_model()
    model.summary()

    # Train model
    history = train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=64)

    # Save model
    model_path = "model/emotion_model.h5"
    model.save(model_path)
    print(f"\n✅ Emotion model saved successfully to {model_path}")

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    print("\n📈 Final Model Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\n🎉 Training completed successfully!")
    print("\nEmotion Labels:")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, label in enumerate(emotion_labels):
        print(f"  {i}: {label}")

