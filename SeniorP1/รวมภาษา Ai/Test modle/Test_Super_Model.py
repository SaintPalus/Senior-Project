import os
import librosa
import numpy as np
import tensorflow as tf
import pickle
import sys
import time

# ตั้งค่า Encoding
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ 1. SETUP (ใช้ชื่อไฟล์ใหม่)
# ==========================================
# หาที่อยู่ไฟล์ปัจจุบัน
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 🚨 ชี้ไปที่ไฟล์ใหม่ (super_...)
MODEL_PATH = os.path.join(CURRENT_DIR, "super_model_multilingual.keras")
LABEL_PATH = os.path.join(CURRENT_DIR, "super_label_encoder.pkl")

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ==========================================
# 🛠️ 2. PRELOAD SYSTEM
# ==========================================
print("⏳ กำลังโหลด Super Model (Universal Version)...")

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print(f"❌ ยังไม่เจอไฟล์โมเดลใหม่ครับ ({MODEL_PATH})")
    print("👉 ใจเย็นๆ รอเทรนให้เสร็จก่อนนะครับ!")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    lb = pickle.load(f)

print(f"✅ พร้อมใช้งาน! (รองรับ {len(lb.classes_)} อารมณ์: {lb.classes_})")
print("="*50)

# ==========================================
# 🧠 3. FEATURE EXTRACTION (ต้องเหมือนตอนเทรนเป๊ะ)
# ==========================================
def extract_features(data, sr):
    # ใช้ MFCC 40 ตัว
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    return mfcc.T

def preprocess_audio(file_path):
    try:
        # Load
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # ✂️ ตัดเสียงเงียบ (สำคัญมาก!)
        y, _ = librosa.effects.trim(y, top_db=25)
        
        # Pad/Truncate
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, int(SAMPLES_PER_TRACK - len(y))), 'constant')
        else:
            y = y[:int(SAMPLES_PER_TRACK)]
            
        # Extract Features
        features = extract_features(y, sr)
        
        # Normalize (Instance Norm - แก้ขัดตอนเทสทีละไฟล์)
        mean = np.mean(features)
        std = np.std(features)
        features = (features - mean) / (std + 1e-6)
        
        return np.expand_dims(features, axis=0)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ==========================================
# 🔄 4. LOOP TEST
# ==========================================
while True:
    print("\n🎧 ลากไฟล์เสียงมาวาง (หรือพิมพ์ q ออก):")
    user_input = input("📍 Path: ").strip()
    file_path = user_input.replace('"', '') # ลบฟันหนู
    
    if file_path.lower() == 'q': break
    if not os.path.exists(file_path):
        print("❌ หาไฟล์ไม่เจอ")
        continue
        
    # ทายผล
    input_data = preprocess_audio(file_path)
    
    if input_data is not None:
        # Predict
        prediction = model.predict(input_data, verbose=0)
        idx = np.argmax(prediction)
        label = lb.classes_[idx]
        conf = prediction[0][idx] * 100
        
        print("-" * 30)
        print(f"🧠 ผลลัพธ์:  {label.upper()}")
        print(f"🔥 ความมั่นใจ: {conf:.2f}%")
        
        # กราฟแท่ง
        print("\n📊 คะแนน:")
        for i, class_name in enumerate(lb.classes_):
            score = prediction[0][i] * 100
            bar = '█' * int(score / 5)
            print(f" {class_name:<10}: {score:5.2f}% | {bar}")
        print("-" * 30)