"""
Model Evaluation Script - Calculate F1 Score, Precision, Recall, and other metrics
"""

import numpy as np
import os
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data(samples_per_class=500):
    """Load test data from FER-2013 dataset"""
    
    data_dir = 'data/test'
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    X_test = []
    y_test = []
    
    print("📂 Loading test data...")
    
    for label, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        
        if not os.path.exists(emotion_dir):
            print(f"⚠️  Warning: {emotion_dir} not found")
            continue
        
        images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Limit samples per class
        images = images[:samples_per_class]
        
        for img_name in images:
            img_path = os.path.join(emotion_dir, img_name)
            try:
                img = keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                
                X_test.append(img_array)
                y_test.append(label)
            except Exception as e:
                continue
        
        print(f"  ✅ {emotion.capitalize():10} - {len([y for y in y_test if y == label])} images")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\n📊 Total test samples: {len(X_test)}")
    
    return X_test, y_test, emotions

def evaluate_model():
    """Evaluate the trained model and calculate all metrics"""
    
    print("=" * 70)
    print("  🎯 MODEL EVALUATION - F1 Score, Precision, Recall")
    print("=" * 70)
    
    # Load model
    model_path = 'model/emotion_model.h5'
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the model first using: python train_minimal.py")
        return
    
    print(f"\n📥 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Load test data
    X_test, y_test, emotions = load_test_data(samples_per_class=500)
    
    # Make predictions
    print("\n🔮 Making predictions on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("  📊 PERFORMANCE METRICS")
    print("=" * 70)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # F1 Scores
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    
    # Precision and Recall
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Overall Metrics:")
    print(f"  • Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📈 F1 Scores:")
    print(f"  • F1 Score (Macro):   {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  • F1 Score (Weighted):{f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"  • F1 Score (Micro):   {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    print(f"\n🎯 Precision:")
    print(f"  • Precision (Macro):  {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"  • Precision (Weighted):{precision_weighted:.4f} ({precision_weighted*100:.2f}%)")
    print(f"\n🎯 Recall:")
    print(f"  • Recall (Macro):     {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"  • Recall (Weighted):  {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")
    
    # Per-class metrics
    print("\n" + "=" * 70)
    print("  📊 PER-CLASS METRICS")
    print("=" * 70)
    
    print("\n" + classification_report(y_test, y_pred, target_names=emotions, digits=4))
    
    # Confusion Matrix
    print("\n" + "=" * 70)
    print("  🔢 CONFUSION MATRIX")
    print("=" * 70)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix - Emotion Recognition Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plot_path = 'model/confusion_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved to: {plot_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  📋 SUMMARY")
    print("=" * 70)
    print(f"""
✅ Model Evaluation Complete!

Key Metrics:
  • Test Accuracy:  {accuracy*100:.2f}%
  • F1 Score:       {f1_macro*100:.2f}% (macro average)
  • Precision:      {precision_macro*100:.2f}% (macro average)
  • Recall:         {recall_macro*100:.2f}% (macro average)

📊 Best Performing Emotions:
""")
    
    # Find best and worst performing classes
    f1_per_class = f1_score(y_test, y_pred, average=None)
    best_idx = np.argmax(f1_per_class)
    worst_idx = np.argmin(f1_per_class)
    
    print(f"  🏆 Best:  {emotions[best_idx].capitalize()} - F1: {f1_per_class[best_idx]*100:.2f}%")
    print(f"  ⚠️  Worst: {emotions[worst_idx].capitalize()} - F1: {f1_per_class[worst_idx]*100:.2f}%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    evaluate_model()

