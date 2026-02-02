import os

print("Checking dataset structure...\n")

train_dir = "data/train"
test_dir = "data/test"

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print("=" * 50)
print("TRAINING DATA:")
print("=" * 50)
for emotion in emotions:
    emotion_path = os.path.join(train_dir, emotion)
    if os.path.exists(emotion_path):
        count = len([f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"{emotion:10s}: {count:5d} images")
    else:
        print(f"{emotion:10s}: FOLDER NOT FOUND")

print("\n" + "=" * 50)
print("TEST DATA:")
print("=" * 50)
for emotion in emotions:
    emotion_path = os.path.join(test_dir, emotion)
    if os.path.exists(emotion_path):
        count = len([f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"{emotion:10s}: {count:5d} images")
    else:
        print(f"{emotion:10s}: FOLDER NOT FOUND")

print("\n" + "=" * 50)

