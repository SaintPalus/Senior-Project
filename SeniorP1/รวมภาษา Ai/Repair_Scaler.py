import os
import librosa
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler

# ตั้งค่า Encoding
sys.stdout.reconfigure(encoding='utf-8')

# 🚨 แก้ Path ให้ตรง
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_SAVE_PATH = os.path.join(CURRENT_DIR, "super_scaler.pkl")

# Config
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

EMOTION_KEYWORDS = {'angry': 'angry', 'happy': 'happy', 'sad': 'sad', 'neutral': 'neutral', 'surprise': 'surprise'}

def extract_features(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    return mfcc.T

print("⏳ กำลังสร้างไฟล์ Scaler ย้อนหลัง (ไม่ต้องเทรนใหม่)...")
X_all = []

# อ่านไฟล์มาคำนวณค่าเฉลี่ย
all_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(('.wav', '.mp3')):
            all_files.append(os.path.join(root, file))

selected_files = all_files  # ไม่ต้อง shuffle ไม่ต้อง slice เอามาให้หมด!

print(f"📂 อ่านข้อมูลจาก {len(selected_files)} ไฟล์ เพื่อสร้างมาตรฐาน...")

for i, file_path in enumerate(selected_files):
    try:
        data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        data, _ = librosa.effects.trim(data, top_db=25)
        if len(data) < SAMPLES_PER_TRACK:
            data = np.pad(data, (0, int(SAMPLES_PER_TRACK - len(data))), 'constant')
        else:
            data = data[:int(SAMPLES_PER_TRACK)]
        
        feat = extract_features(data, sr)
        X_all.append(feat)
    except: pass
    
    if (i+1) % 100 == 0:
        print(f"   Reading {i+1}...", end='\r')

X_all = np.array(X_all)
N, T, F = X_all.shape

# สร้าง Scaler
print("\n🧮 กำลังคำนวณค่ามาตรฐาน (Fitting Scaler)...")
scaler = StandardScaler()
scaler.fit(X_all.reshape(N, -1))

# บันทึกไฟล์
with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✅ สร้างไฟล์สำเร็จ: {SCALER_SAVE_PATH}")
print("👉 ไปรันไฟล์ Test_Final.py ได้เลยครับ!")