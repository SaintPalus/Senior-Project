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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ตั้งค่า Encoding
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
# 🚨 แก้ Path ให้ตรงเหมือนเดิม
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"

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
# 🛠️ 2. FUNCTIONS (ต้องเหมือนตอน Train เป๊ะๆ)
# ==========================================
def extract_features_ultimate(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    result = np.concatenate((mfcc, mel_db), axis=0)
    return result.T

def process_files_batch(file_list, desc="Processing"):
    X, y = [], []
    print(f"🔄 {desc} ({len(file_list)} files)...")
    
    for i, file_path in enumerate(file_list):
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
                data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                if len(data) < SAMPLES_PER_TRACK:
                    data = np.pad(data, (0, int(SAMPLES_PER_TRACK - len(data))), 'constant')
                else:
                    data = data[:int(SAMPLES_PER_TRACK)]
                
                X.append(extract_features_ultimate(data, sr))
                y.append(label)
            except: pass

        if (i+1) % 100 == 0:
            print(f"   {desc}: {i+1}/{len(file_list)}...", end='\r')
            
    return np.array(X), np.array(y)

# ==========================================
# 🚀 3. MAIN PROCESS (REBUILD SCALER)
# ==========================================
print("⏳ เริ่มกระบวนการกู้คืนความแม่นยำ (Reconstructing Environment)...")

# 1. โหลดรายชื่อไฟล์ทั้งหมด และ "เรียงลำดับ" (สำคัญมากเพื่อให้ Split เหมือนเดิม)
all_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(('.wav', '.mp3')):
            all_files.append(os.path.join(root, file))

# ⚠️ การ Sort สำคัญที่สุด! ถ้าไม่เรียง Split จะเพี้ยน
all_files.sort() 

if not all_files:
    print("❌ ไม่พบไฟล์เสียง")
    exit()

# 2. แบ่งข้อมูลเหมือนตอน Train เป๊ะๆ (Train 80% / Test 20%)
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

print(f"📂 พบไฟล์ทั้งหมด: {len(all_files)}")
print(f"🏋️ Train Set: {len(train_files)} ไฟล์ (ใช้สร้าง Scaler)")
print(f"🧪 Test Set:  {len(test_files)} ไฟล์ (ใช้ทดสอบจริง)")
print("="*60)

# 3. โหลดข้อมูล Train เพื่อสร้าง Scaler (ยอมเสียเวลาตรงนี้เพื่อความแม่นยำ)
# เราจะโหลดแบบไม่ Augment เพื่อความรวดเร็วในการหาค่า Mean/Std
print("⚙️ ขั้นตอนที่ 1/3: อ่านข้อมูล Train เพื่อจูนค่ามาตรฐาน (StandardScaler)...")
X_train_dummy, _ = process_files_batch(train_files, desc="Reading Train Data")

print(f"\n📊 กำลัง Fit Scaler กับข้อมูล {len(X_train_dummy)} ตัวอย่าง...")
scaler = StandardScaler()
N, T, F = X_train_dummy.shape
# Fit Scaler
scaler.fit(X_train_dummy.reshape(N, -1))
print("✅ Scaler พร้อมใช้งานแล้ว! (นี่คือกุญแจสำคัญ)")
del X_train_dummy # เคลียร์แรม

# 4. โหลดข้อมูล Test มาทดสอบ
print("\n⚙️ ขั้นตอนที่ 2/3: อ่านข้อมูล Test Set...")
X_test, y_test_labels = process_files_batch(test_files, desc="Reading Test Data")

# 5. ใช้ Scaler ตัวเมื่อกี้ แปลงข้อมูล Test
print("\n⚙️ ขั้นตอนที่ 3/3: Normalizing & Predicting...")
N_test, T_test, F_test = X_test.shape
X_test_scaled = scaler.transform(X_test.reshape(N_test, -1)).reshape(N_test, T_test, F_test)

# 6. โหลดโมเดลและทำนาย
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    lb = pickle.load(f)

y_test_encoded = lb.transform(y_test_labels)
y_pred_prob = model.predict(X_test_scaled, verbose=1)
y_pred_indices = np.argmax(y_pred_prob, axis=1)

# ==========================================
# 📊 4. REPORT
# ==========================================
acc = np.mean(y_test_encoded == y_pred_indices)
print("\n" + "="*50)
print(f"🏆 FINAL ACCURACY: {acc:.2%} (ของจริงมาแล้ว!)")
print("="*50)

print(classification_report(y_test_encoded, y_pred_indices, target_names=lb.classes_))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_encoded, y_pred_indices)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()