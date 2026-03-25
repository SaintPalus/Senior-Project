import os
import librosa
import numpy as np
import tensorflow as tf
import pickle
import sys

# ตั้งค่า Encoding
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
# 👇 ใส่ชื่อไฟล์เสียงที่คุณต้องการทดสอบตรงนี้
TEST_FILE = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset\angry\OAF_back_angry.wav" 
# (หรือจะเปลี่ยนเป็นไฟล์เสียงของคุณเองที่อัดใหม่ก็ได้ครับ)

MODEL_PATH = "real_acc_model.keras"
LABEL_PATH = "label_encoder.pkl"

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ==========================================
# 🛠️ 2. PREPROCESSING (ต้องเหมือนตอน Train เป๊ะๆ)
# ==========================================
def preprocess_audio(file_path):
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # 2. Pad/Truncate (ตัด/เติมให้ครบ 3 วิ)
        if len(y) < SAMPLES_PER_TRACK:
            padding = int(SAMPLES_PER_TRACK) - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            y = y[:int(SAMPLES_PER_TRACK)]
            
        # 3. Extract MFCC (40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc.T  # (Time, Features)
        
        # 4. Normalize (สำคัญมาก!)
        # เนื่องจากเราไม่ได้เซฟ Scaler ไว้ เราจะใช้การ Normalize ตัวเอง (Instance Norm)
        # ซึ่งพอถูไถได้สำหรับการเทสเบื้องต้น
        mean = np.mean(mfcc)
        std = np.std(mfcc)
        mfcc = (mfcc - mean) / (std + 1e-6) # ป้องกันหาร 0
        
        # 5. Reshape เข้า Model (1, Time, Features)
        return np.expand_dims(mfcc, axis=0)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ==========================================
# 🚀 3. LOAD MODEL & PREDICT
# ==========================================
# เช็คไฟล์ก่อน
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print("❌ ไม่เจอไฟล์โมเดล! กรุณารัน Train_Model ให้เสร็จก่อนครับ")
    exit()

print("⏳ กำลังโหลดโมเดล...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    lb = pickle.load(f)

print(f"\n🎤 กำลังวิเคราะห์ไฟล์: {os.path.basename(TEST_FILE)}")
input_data = preprocess_audio(TEST_FILE)

if input_data is not None:
    # ทายผล
    prediction = model.predict(input_data, verbose=0)
    
    # แปลผล
    predicted_index = np.argmax(prediction)
    predicted_label = lb.classes_[predicted_index]
    confidence = prediction[0][predicted_index] * 100
    
    # แสดงผลลัพธ์
    print("\n" + "="*30)
    print(f"🧠 ผลลัพธ์: {predicted_label.upper()} (อารมณ์: {predicted_label})")
    print(f"🔥 ความมั่นใจ: {confidence:.2f}%")
    print("="*30)
    
    # โชว์ความน่าจะเป็นทั้งหมด
    print("\n📊 รายละเอียดความน่าจะเป็น:")
    for i, label in enumerate(lb.classes_):
        score = prediction[0][i] * 100
        bar = "█" * int(score / 5)
        print(f" {label:<10}: {score:5.2f}% | {bar}")