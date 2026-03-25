import os
import librosa
import numpy as np
import tensorflow as tf
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# ตั้งค่าให้โชว์ภาษาไทยใน Console ได้
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision, regularizers

# ==========================================
# 🚨 1. SETUP & PATH (แก้ตรงนี้ให้ตรงเครื่องคุณ)
# ==========================================
# หาที่อยู่ปัจจุบันของไฟล์นี้ เพื่อเซฟโมเดลไว้ที่เดียวกัน
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "super_model_multilingual.keras")
LABEL_SAVE_PATH = os.path.join(CURRENT_DIR, "super_label_encoder.pkl")

# 📂 โฟลเดอร์ Dataset ของคุณ
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"

# ตั้งค่าเสียง
SAMPLE_RATE = 22050
DURATION = 3  # ตัดเสียงให้เหลือ 3 วินาทีเท่ากันหมด
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ตั้งค่าการเทรน
BATCH_SIZE = 64
EPOCHS = 100

# คำศัพท์อารมณ์ (Keyword) ที่ใช้จับผิดชื่อโฟลเดอร์
EMOTION_KEYWORDS = {
    'angry': 'angry', 
    'happy': 'happy', 
    'sad': 'sad', 
    'neutral': 'neutral', 
    'surprise': 'surprise'
}

# ==========================================
# 🛠️ 2. SMART DATA LOADER (หัวใจสำคัญ)
# ==========================================
def extract_features(data, sr):
    # ใช้ MFCC 40 ตัว (มาตรฐานสากลสำหรับเสียงพูดหลายภาษา)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    return mfcc.T

def load_data_recursive(root_path):
    X, y = [], []
    print(f"🌍 เริ่มปฏิบัติการสแกนไฟล์ทั่วจักรวาลที่: {root_path}")
    
    # 1. รวบรวมรายชื่อไฟล์ทั้งหมดก่อน
    all_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                full_path = os.path.join(root, file)
                all_files.append(full_path)
    
    print(f"   -> พบไฟล์เสียงทั้งหมด: {len(all_files)} ไฟล์ (กำลังคัดแยก...)")
    
    # สุ่มสลับไฟล์ (Shuffle) เพื่อให้การเทรนกระจายตัวดี
    np.random.shuffle(all_files)
    
    # 2. วนลูปเช็คทีละไฟล์
    for i, file_path in enumerate(all_files):
        # แปลง Path เป็นตัวเล็กและใช้ / เพื่อให้เช็คง่าย
        path_check = file_path.lower().replace('\\', '/')
        
        label = None
        
        # 🕵️‍♂️ Super Path Search: เช็คว่าใน Path มีชื่ออารมณ์ซ่อนอยู่ไหม?
        for key, emotion_name in EMOTION_KEYWORDS.items():
            # เช่นถ้าเจอคำว่า /angry/ ใน Path ก็ถือว่าใช่เลย
            # ใส่เครื่องหมาย / ปิดหน้าหลัง เพื่อความชัวร์ (กันไปชนกับชื่อไฟล์อื่น)
            if f"/{key}/" in path_check or f"\\{key}\\" in path_check:
                label = emotion_name
                break
            # เผื่อกรณีชื่อไฟล์บอกอารมณ์ตรงๆ (เช่น 03-01-05-...wav)
            if key in os.path.basename(path_check):
                label = emotion_name
                break

        # ถ้าเจออารมณ์ ก็โหลดเสียงเลย
        if label:
            try:
                # โหลดเสียง
                data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # ✂️ ตัดเสียงเงียบทิ้ง (Silence Removal) สำคัญมาก!
                data, _ = librosa.effects.trim(data, top_db=25)
                
                # ปรับความยาวให้เท่ากัน (Pad/Truncate)
                if len(data) < SAMPLES_PER_TRACK:
                    data = np.pad(data, (0, int(SAMPLES_PER_TRACK - len(data))), 'constant')
                else:
                    data = data[:int(SAMPLES_PER_TRACK)]
                
                # แปลงเป็นตัวเลข (Feature Extraction)
                feat = extract_features(data, sr)
                
                X.append(feat)
                y.append(label)
                
            except Exception as e:
                # ถ้าไฟล์เสีย ให้ข้ามไปเงียบๆ
                pass
        
        # โชว์ความคืบหน้าทุกๆ 100 ไฟล์
        if (i+1) % 100 == 0:
            print(f"   Processed {i+1}/{len(all_files)}... (Collected: {len(X)} samples)", end='\r')
            
    return np.array(X), np.array(y)

# ==========================================
# 🚀 3. PREPARE DATA
# ==========================================
# เปิด GPU (ถ้ามี)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
    print("\n✅ GPU Ready! (RTX 3060 Engine Start)")
except: pass

# เริ่มโหลดข้อมูล
X, y = load_data_recursive(DATA_PATH)

print(f"\n\n📊 สรุปยอดข้อมูลที่ใช้ได้จริง: {len(X)} ตัวอย่าง")
if len(X) == 0:
    print("❌ ไม่พบข้อมูล! (ลองเช็คชื่อโฟลเดอร์หลักดูครับ ต้องมีคำว่า angry, happy ฯลฯ)")
    exit()

# แปลง Label เป็นตัวเลข
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
print(f"🏷️ Classes ที่เจอ: {lb.classes_}")

# บันทึกตัวแปลภาษา (Label Encoder) เก็บไว้ใช้ตอน Test
with open(LABEL_SAVE_PATH, 'wb') as f:
    pickle.dump(lb, f)

# แบ่งข้อมูล Train / Val / Test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# ปรับมาตรฐานข้อมูล (Normalization)
scaler = StandardScaler()
N, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape[0], T, F)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], T, F)

# ==========================================
# 🧠 4. CREATE MODEL (Hybrid CNN + Bi-LSTM)
# ==========================================
model = Sequential()

# CNN Layers (ตาดูรูปคลื่น)
model.add(Conv1D(256, 5, padding='same', activation='relu', input_shape=(T, F)))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

# Bi-LSTM Layers (หูฟังจังหวะ)
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))

# Output Layers (สมองตัดสินใจ)
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(len(lb.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 🛑 5. TRAIN START!
# ==========================================
callbacks = [
    # เซฟเฉพาะตอนที่แม่นยำขึ้นเท่านั้น
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    # ถ้าค่าไม่ดีขึ้น 10 รอบ ให้หยุดพัก (กัน Overfit)
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    # ถ้ากราฟไม่ขยับ ให้ลด Learning Rate ลง
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

print("\n🚀 START TRAINING SUPER MODEL...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 📊 6. FINAL REPORT
# ==========================================
print("\n🏆 โหลดโมเดลที่เทพที่สุดมาทดสอบ...")
model.load_weights(MODEL_SAVE_PATH)

print("\n📊 ผลการสอบ (Test Set):")
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=lb.classes_))

# วาดกราฟ Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title('Final Confusion Matrix (Universal Model)')
plt.show()

print(f"✅ เสร็จสิ้นภารกิจ! ไฟล์โมเดลอยู่ที่: {MODEL_SAVE_PATH}")