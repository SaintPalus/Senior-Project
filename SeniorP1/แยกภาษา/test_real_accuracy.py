import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🌟 ส่วนที่ 1: โหลด Model และ Label เดิม
# ==========================================
print("⏳ กำลังเตรียมข้อมูลและโหลดโมเดล...")
feature_dir = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\extracted_features"
model_path = "asian_emotion_cnn_model.h5"
MAX_TIME_STEPS = 130 

# โหลดโมเดลที่เซฟไว้
model = tf.keras.models.load_model(model_path)

# --- สร้าง Label Encoder ใหม่ให้ตรงกับตอนเทรน ---
y_all = []
X_all = []
for file in os.listdir(feature_dir):
    if file.endswith('.npy'):
        parts = file.split('_')
        if len(parts) >= 2:
            y_all.append(parts[1]) # ดึงอารมณ์จากชื่อไฟล์
            X_all.append(file) # เก็บแค่ชื่อไว้ทำสัดส่วน

label_encoder = LabelEncoder()
label_encoder.fit(y_all)
classes = label_encoder.classes_

print(f"✅ โหลดโมเดลสำเร็จ! อารมณ์ที่ AI รู้จัก: {classes}")

# ==========================================
# 🌟 ส่วนที่ 2: ระบบทดสอบด้วยไฟล์เสียงจริง (Real Inference)
# ==========================================
def predict_emotion_from_audio(wav_path):
    print(f"\n🎤 กำลังวิเคราะห์ไฟล์เสียง: {wav_path}")
    
    # 1. โหลดและแปลงเสียงเป็น Spectrogram แบบเดียวกับตอนเทรน
    y, sr = librosa.load(wav_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. ปรับขนาด (Padding/Truncating) ให้เท่ากับ MAX_TIME_STEPS (130)
    if mel_spec_db.shape[1] < MAX_TIME_STEPS:
        pad_width = MAX_TIME_STEPS - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :MAX_TIME_STEPS]
        
    # 3. จัด Shape ให้เข้า CNN ได้ (1, 128, 130, 1)
    X_input = mel_spec_db[np.newaxis, ..., np.newaxis]
    
    # 4. ให้ AI ทำนายผล
    predictions = model.predict(X_input, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
    
    print(f"🎯 ผลลัพธ์: AI ทายว่าเป็นอารมณ์ **{predicted_emotion}** (มั่นใจ {confidence:.2f}%)")
    
    # โชว์ความมั่นใจของทุกอารมณ์
    print("รายละเอียดความมั่นใจ:")
    for i, prob in enumerate(predictions[0]):
        print(f"   - {classes[i]}: {prob*100:.2f}%")

# ==========================================
# 🌟 วิธีใช้งาน: เอาไฟล์เสียงที่คุณอัดเอง หรือโหลดมาใหม่ มาใส่ตรงนี้!
# ==========================================
# เปลี่ยน Path ตรงนี้เป็นไฟล์เสียงที่คุณต้องการทดสอบ
test_audio_file = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset\Chinese\Chinese ESD\0003_000708.wav"

if os.path.exists(test_audio_file):
    predict_emotion_from_audio(test_audio_file)
else:
    print(f"\n⚠️ ไม่พบไฟล์ทดสอบ: {test_audio_file}\n👉 (กรุณาเปลี่ยน Path ด้านล่างสุดของโค้ดให้ชี้ไปที่ไฟล์ .wav ที่มีอยู่จริงเพื่อทดสอบระบบครับ)")

# --- แถม: โค้ดดึง Classification Report ของจริง ---
# (หมายเหตุ: ในการใช้งานจริง ควรโหลด X_test, y_test กลับมาประเมิน 
# แต่เพื่อความรวดเร็ว คุณสามารถนำสคริปต์นี้ไปประยุกต์ใช้ดูได้ครับ)
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Speech Emotion Recognition')
    plt.ylabel('True Emotion (เฉลย)')
    plt.xlabel('Predicted Emotion (AI ทาย)')
    plt.show()