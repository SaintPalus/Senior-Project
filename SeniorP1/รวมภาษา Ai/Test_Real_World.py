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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(os.path.dirname(CURRENT_DIR), "dataset", "angry", "OAF_back_angry.wav")
# (หรือจะเปลี่ยนเป็นไฟล์เสียงของคุณเองที่อัดใหม่ก็ได้ครับ)

MODEL_PATH  = os.path.join(CURRENT_DIR, "ultimate_model_rtx.keras")
SCALER_PATH = os.path.join(CURRENT_DIR, "rtx_scaler.pkl")
LABEL_PATH  = os.path.join(CURRENT_DIR, "label_encoder.pkl")

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ==========================================
# 🛠️ 2. PREPROCESSING (ต้องเหมือนตอน Train เป๊ะๆ)
# ==========================================
def extract_features(data, sr):
    mfcc    = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)
    mel     = librosa.feature.melspectrogram(y=data, sr=sr)
    zcr     = librosa.feature.zero_crossing_rate(data)
    rms     = librosa.feature.rms(y=data)
    sc      = librosa.feature.spectral_centroid(y=data, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)
    result  = np.concatenate((mfcc, mel, zcr, rms, sc, rolloff), axis=0)
    return result.T  # (Time, 260)

def preprocess_audio(file_path, scaler):
    try:
        # 1. Load & trim silence
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        y, _ = librosa.effects.trim(y, top_db=25)

        # 2. Pad / Truncate
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, int(SAMPLES_PER_TRACK) - len(y)), 'constant')
        else:
            y = y[:int(SAMPLES_PER_TRACK)]

        # 3. Feature extraction (ต้องตรงกับตอน Train เป๊ะๆ)
        feat = extract_features(y, sr)  # (T, F)

        # 4. Normalize ด้วย Scaler ที่เซฟไว้ตอน Train
        T, F = feat.shape
        feat = scaler.transform(feat.reshape(1, -1)).reshape(1, T, F)

        return feat

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ==========================================
# 🚀 3. LOAD MODEL & PREDICT
# ==========================================
# เช็คไฟล์ก่อน
for path, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (LABEL_PATH, "Label Encoder")]:
    if not os.path.exists(path):
        print(f"❌ ไม่เจอไฟล์ {name}! กรุณารัน Train_Model_RTX3060.py ให้เสร็จก่อนครับ")
        exit()

print("⏳ กำลังโหลดโมเดล...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABEL_PATH, 'rb') as f:
    lb = pickle.load(f)

print(f"\n🎤 กำลังวิเคราะห์ไฟล์: {os.path.basename(TEST_FILE)}")
input_data = preprocess_audio(TEST_FILE, scaler)

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