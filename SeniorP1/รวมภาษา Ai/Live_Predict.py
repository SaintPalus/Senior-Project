import os
import sys
import pickle
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(CURRENT_DIR, "ultimate_model_rtx.keras")
SCALER_PATH     = os.path.join(CURRENT_DIR, "rtx_scaler.pkl")
LABEL_PATH      = os.path.join(CURRENT_DIR, "label_encoder.pkl")

SAMPLE_RATE     = 22050
DURATION        = 3
SAMPLES         = SAMPLE_RATE * DURATION

EMOTION_TH = {
    'angry':    '😡 โกรธ',
    'happy':    '😊 มีความสุข',
    'sad':      '😢 เศร้า',
    'neutral':  '😐 เฉยๆ',
    'surprise': '😲 ประหลาดใจ',
}

# ==========================================
# 🛠️ 2. FEATURE EXTRACTION (ต้องตรงกับ Train เป๊ะๆ)
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

# ==========================================
# 🎤 3. RECORD FROM MICROPHONE
# ==========================================
def record_audio():
    print(f"\n🔴 อัดเสียง {DURATION} วินาที...")
    audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ อัดเสียงเสร็จแล้ว กำลังวิเคราะห์...")
    return audio.flatten()

# ==========================================
# 🔮 4. PREDICT
# ==========================================
def predict(audio, model, scaler, lb):
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Pad / Truncate
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)), 'constant')
    else:
        audio = audio[:SAMPLES]

    # Feature extraction + normalize
    feat = extract_features(audio, SAMPLE_RATE)
    T, F = feat.shape
    feat = scaler.transform(feat.reshape(1, -1)).reshape(1, T, F)

    # Predict
    probs = model.predict(feat, verbose=0)[0]
    idx   = np.argmax(probs)
    label = lb.classes_[idx]
    conf  = probs[idx] * 100

    return label, conf, probs

def print_result(label, conf, probs, lb):
    emotion_th = EMOTION_TH.get(label, label)
    print("\n" + "=" * 40)
    print(f"  ผลลัพธ์  : {emotion_th}")
    print(f"  ความมั่นใจ: {conf:.1f}%")
    print("=" * 40)
    print("  รายละเอียด:")
    for i, cls in enumerate(lb.classes_):
        score = probs[i] * 100
        bar   = "█" * int(score / 4)
        th    = EMOTION_TH.get(cls, cls)
        print(f"  {th:<16}: {score:5.1f}%  {bar}")
    print()

# ==========================================
# 🚀 5. MAIN LOOP
# ==========================================
print("=" * 40)
print("  🎙️  Live Emotion Detector")
print("=" * 40)

# โหลด Model / Scaler / Label
for path, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (LABEL_PATH, "Label Encoder")]:
    if not os.path.exists(path):
        print(f"❌ ไม่เจอไฟล์ {name}! กรุณารัน Train_Model_RTX3060.py ก่อนครับ")
        sys.exit(1)

print("⏳ กำลังโหลดโมเดล...")
model  = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(LABEL_PATH, 'rb') as f:
    lb = pickle.load(f)

print(f"✅ โมเดลพร้อมใช้งาน | อารมณ์ที่รู้จัก: {list(lb.classes_)}\n")

while True:
    try:
        cmd = input("กด Enter เพื่ออัดเสียง  หรือพิมพ์ 'q' เพื่อออก: ").strip().lower()
        if cmd == 'q':
            print("👋 ออกจากโปรแกรมแล้ว")
            break

        audio = record_audio()
        label, conf, probs = predict(audio, model, scaler, lb)
        print_result(label, conf, probs, lb)

    except KeyboardInterrupt:
        print("\n👋 ออกจากโปรแกรมแล้ว")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
