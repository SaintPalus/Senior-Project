import os
import librosa
import numpy as np
import tensorflow as tf
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "super_model_multilingual.keras")
LABEL_PATH = os.path.join(CURRENT_DIR, "super_label_encoder.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "super_scaler.pkl") # 🔥 ตัวช่วยสำคัญ

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

print("⏳ กำลังโหลดระบบ...")
if not os.path.exists(SCALER_PATH):
    print("❌ ไม่เจอไฟล์ super_scaler.pkl (กรุณารันไฟล์ Repair_Scaler.py ก่อนครับ)")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f: lb = pickle.load(f)
with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f) # 🔥 โหลด Scaler

print(f"✅ ระบบพร้อมทำงาน! (Model + Scaler)")

# ==========================================
# 🛠️ PROCESS
# ==========================================
def extract_features(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    return mfcc.T

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        y, _ = librosa.effects.trim(y, top_db=25) # ตัดเสียงเงียบ
        
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, int(SAMPLES_PER_TRACK - len(y))), 'constant')
        else:
            y = y[:int(SAMPLES_PER_TRACK)]
            
        features = extract_features(y, sr)
        
        # 🔥 ใช้ Scaler ตัวเดียวกับตอนเทรน (กุญแจสำคัญ!)
        # ต้อง Reshape ให้เป็น 2D ก่อนเข้า Scaler แล้วค่อยกลับเป็น 3D
        features = scaler.transform(features.reshape(1, -1)).reshape(1, features.shape[0], features.shape[1])
        
        return features
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ==========================================
# 🔄 LOOP TEST
# ==========================================
while True:
    print("\n🎧 ลากไฟล์เสียงมาวาง (หรือพิมพ์ q ออก):")
    user_input = input("📍 Path: ").strip().replace('"', '')
    if user_input.lower() == 'q': break
    
    if not os.path.exists(user_input):
        print("❌ หาไฟล์ไม่เจอ")
        continue

    input_data = preprocess_audio(user_input)
    
    if input_data is not None:
        pred = model.predict(input_data, verbose=0)
        idx = np.argmax(pred)
        label = lb.classes_[idx]
        conf = pred[0][idx] # ค่าความมั่นใจ (0.0 - 1.0)
        
        # 🔥 เพิ่ม Logic: กฎเหล็กดึงสติ AI
        # ถ้าทายว่าเป็น Happy แต่ความมั่นใจไม่ถึง 80% ให้เปลี่ยนเป็น Neutral
        if label == 'happy' and conf < 0.80:
            print(f"⚠️ AI ลังเล ({conf*100:.2f}%) -> ปรับเป็น Neutral เพื่อความปลอดภัย")
            label = 'neutral'
            idx = list(lb.classes_).index('neutral') # หา index ของ neutral ใหม่

        # (โค้ดแสดงผลเดิม...)
        print("-" * 30)
        print(f"🧠 ผลลัพธ์:  {label.upper()}")