import os
import librosa
import numpy as np
import tensorflow as tf
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ตั้งค่า Encoding ให้รองรับภาษาไทย
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
# 🚨 แก้ Path Dataset ให้ตรงกับเครื่องคุณ
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"

# ไฟล์โมเดลและ Label Encoder
MODEL_PATH = "ultimate_model_rtx.keras"
LABEL_PATH = "label_encoder.pkl"

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

EMOTION_KEYWORDS = {
    'angry': 'angry', 'anger': 'angry', 'ag': 'angry',
    'happy': 'happy', 'joy': 'happy', 'hap': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'neutral': 'neutral', 'neu': 'neutral',
    'surprise': 'surprise', 'sur': 'surprise'
}
RAVDESS_MAP = {'03': 'happy', '04': 'sad', '05': 'angry', '01': 'neutral'}

# ==========================================
# 🛠️ 2. PREPROCESSING FUNCTIONS (Ultimate Mode)
# ==========================================
def extract_features_ultimate(data, sr):
    # ต้องเหมือนตอน Train เป๊ะๆ (MFCC 128 + Mel)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    result = np.concatenate((mfcc, mel_db), axis=0)
    return result.T

def process_files(file_list):
    X, y = [], []
    print(f"🔄 กำลังประมวลผลไฟล์เสียงจำนวน {len(file_list)} ไฟล์...")
    
    for i, file_path in enumerate(file_list):
        # หา Label
        path_lower = file_path.lower().replace('\\', '/')
        filename = os.path.basename(path_lower)
        parts = filename.split('-')
        label = None
        
        if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
            label = RAVDESS_MAP[parts[2]]
        else:
            for k, v in EMOTION_KEYWORDS.items():
                if k in path_lower:
                    label = v
                    break
        
        if label:
            try:
                # Load Audio
                data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                if len(data) < SAMPLES_PER_TRACK:
                    data = np.pad(data, (0, int(SAMPLES_PER_TRACK - len(data))), 'constant')
                else:
                    data = data[:int(SAMPLES_PER_TRACK)]
                
                # Extract Feature
                feature = extract_features_ultimate(data, sr)
                
                # Normalize (Instance Norm) - เทคนิคสำคัญเมื่อไม่มี Scaler เดิม
                mean = np.mean(feature)
                std = np.std(feature)
                feature = (feature - mean) / (std + 1e-6)
                
                X.append(feature)
                y.append(label)
            except Exception as e:
                print(f"❌ Error: {file_path}")

        # Show Progress
        if (i+1) % 100 == 0:
            print(f"   Processed {i+1}/{len(file_list)}...", end='\r')
            
    return np.array(X), np.array(y)

# ==========================================
# 🚀 3. MAIN EVALUATION PROCESS
# ==========================================
print("⏳ กำลังเตรียมการทดสอบ (RTX 3060)...")

# 1. โหลดโมเดล
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print(f"❌ ไม่เจอไฟล์ {MODEL_PATH} หรือ {LABEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    lb = pickle.load(f)

# 2. เตรียมรายชื่อไฟล์ทั้งหมด
all_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(('.wav', '.mp3')):
            all_files.append(os.path.join(root, file))

if not all_files:
    print("❌ ไม่พบไฟล์เสียงใน Dataset")
    exit()

# 3. แยก Test Set ออกมา (ใช้ Random State 42 เพื่อให้ได้ชุดเดียวกับตอน Train เป๊ะๆ)
# เราจะเทสเฉพาะชุดนี้เท่านั้น เพื่อวัดผลจริง (Real World Accuracy)
_, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

print(f"📂 พบไฟล์ทั้งหมด: {len(all_files)}")
print(f"🧪 ใช้ทดสอบ (Test Set 20%): {len(test_files)} ไฟล์")
print("="*50)

# 4. ประมวลผล Test Set
X_test, y_test_labels = process_files(test_files)

# 5. แปลง Label เป็นตัวเลข
y_test_encoded = lb.transform(y_test_labels)

print("\n\n🧠 กำลังให้ AI ทำข้อสอบ (Predicting)...")
# 6. ทำนายผล
y_pred_prob = model.predict(X_test, verbose=1)
y_pred_indices = np.argmax(y_pred_prob, axis=1)

# ==========================================
# 📊 4. REPORT & GRAPH
# ==========================================
print("\n" + "="*50)
print("📊 REPORT ผลการทดสอบ (ใช้ใส่ในเล่ม)")
print("="*50)

# Classification Report
print(classification_report(y_test_encoded, y_pred_indices, target_names=lb.classes_))

# Confusion Matrix Plot
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_encoded, y_pred_indices)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title(f'Confusion Matrix (Total Accuracy: {np.mean(y_test_encoded == y_pred_indices):.2%})')
plt.ylabel('True Label (ของจริง)')
plt.xlabel('Predicted Label (โมเดลทาย)')
plt.show()

print("✅ เสร็จสิ้น! แคปหน้าจอนี้ไปใช้ได้เลยครับ")