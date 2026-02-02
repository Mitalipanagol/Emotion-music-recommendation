"""
Quick Training Script with Minimal Dataset
For fast demonstration and testing
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import cv2

def load_minimal_dataset(data_dir="data", samples_per_emotion=100):
    """Load a minimal subset of the dataset for quick training"""
    print(f"📂 Loading minimal dataset ({samples_per_emotion} samples per emotion)...")
    
    train_dir = os.path.join(data_dir, "train")
    
    emotion_labels = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'sad': 4, 'surprise': 5, 'neutral': 6
    }
    
    images = []
    labels = []
    
    for emotion_name, emotion_id in emotion_labels.items():
        emotion_dir = os.path.join(train_dir, emotion_name)
        if not os.path.exists(emotion_dir):
            print(f"⚠️  Warning: {emotion_name} folder not found")
            continue
            
        print(f"   Loading {emotion_name} images...")
        count = 0
        for img_file in os.listdir(emotion_dir):
            if count >= samples_per_emotion:
                break
                
            img_path = os.path.join(emotion_dir, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(emotion_id)
                    count += 1
            except Exception as e:
                continue
    
    X = np.array(images).reshape(-1, 48, 48, 1) / 255.0
    y = to_categorical(np.array(labels), num_classes=7)
    
    print(f"✅ Loaded {len(X)} images total")
    return X, y

def create_simple_model():
    """Create a simpler CNN model for faster training"""
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully Connected
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    return model

def main():
    print("=" * 60)
    print("🚀 MINIMAL EMOTION MODEL TRAINING")
    print("=" * 60)
    
    # Load minimal dataset (use 500 samples per emotion for better accuracy)
    X, y = load_minimal_dataset(samples_per_emotion=500)
    
    if len(X) == 0:
        print("❌ No data loaded. Please check your dataset.")
        return
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Create model
    print("\n🏗️  Building simplified CNN model...")
    model = create_simple_model()
    model.summary()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\n🚀 Starting training (this will be quick)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = "model/emotion_model.h5"
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📈 Final Results:")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print("\n🎉 Training completed!")
    print("\nℹ️  Note: This is a minimal model for demonstration.")
    print("   For better accuracy, train with the full dataset using train_emotion_model.py")

if __name__ == "__main__":
    main()

