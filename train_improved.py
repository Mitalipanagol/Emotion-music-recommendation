"""
Improved Training Script with Better Data Sampling
Uses 500 samples per emotion for better accuracy
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2

def load_balanced_dataset(data_dir="data", samples_per_emotion=500):
    """Load a balanced dataset with equal samples per emotion"""
    print(f"📂 Loading balanced dataset ({samples_per_emotion} samples per emotion)...")
    
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    emotion_labels = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'sad': 4, 'surprise': 5, 'neutral': 6
    }
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    # Load training data
    for emotion_name, emotion_id in emotion_labels.items():
        emotion_dir = os.path.join(train_dir, emotion_name)
        if not os.path.exists(emotion_dir):
            print(f"⚠️  Warning: {emotion_name} folder not found")
            continue
            
        print(f"   Loading {emotion_name} training images...")
        files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Shuffle and take samples
        np.random.shuffle(files)
        files = files[:samples_per_emotion]
        
        for img_file in files:
            img_path = os.path.join(emotion_dir, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    X_train_list.append(img)
                    y_train_list.append(emotion_id)
            except Exception as e:
                continue
    
    # Load test data
    for emotion_name, emotion_id in emotion_labels.items():
        emotion_dir = os.path.join(test_dir, emotion_name)
        if not os.path.exists(emotion_dir):
            continue
            
        print(f"   Loading {emotion_name} test images...")
        files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Take all test images (or limit to 100 per emotion)
        files = files[:100]
        
        for img_file in files:
            img_path = os.path.join(emotion_dir, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    X_test_list.append(img)
                    y_test_list.append(emotion_id)
            except Exception as e:
                continue
    
    # Convert to numpy arrays
    X_train = np.array(X_train_list, dtype='float32')
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list, dtype='float32')
    y_test = np.array(y_test_list)
    
    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Reshape
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 7)
    y_test = to_categorical(y_test, 7)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    return X_train, y_train, X_test, y_test

def build_emotion_model():
    """Build CNN model for emotion recognition"""
    model = Sequential([
        # Conv Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Conv Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    return model

def main():
    print("🚀 Starting Improved Model Training...")
    print("=" * 60)
    
    # Load dataset (500 samples per emotion = 3500 total training)
    X_train, y_train, X_test, y_test = load_balanced_dataset(samples_per_emotion=500)
    
    # Split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"\n📊 Data Split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # Build model
    print("\n🏗️  Building model...")
    model = build_emotion_model()

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n📋 Model Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint('model/emotion_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Train model
    print("\n🎯 Training model...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")

    # Plot training history
    print("\n📈 Plotting training history...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_improved.png')
    print("   Saved training history plot to 'training_history_improved.png'")

    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print(f"   Model saved to: model/emotion_model.h5")
    print(f"   Final Test Accuracy: {test_accuracy*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    main()

