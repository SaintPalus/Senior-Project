import os
import tensorflow as tf
import numpy as np
import librosa
import sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ตั้งค่า Encoding
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
# 🚨 1. ระบบเซฟไฟล์นิรภัย (Safety Save System)
# ==========================================
# หาที่อยู่ปัจจุบันของไฟล์นี้ เพื่อบังคับเซฟที่เดียวกัน
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "ultimate_model_rtx.keras")
LABEL_SAVE_PATH = os.path.join(CURRENT_DIR, "label_encoder.pkl")

print(f"📍 ไฟล์โมเดลจะถูกบันทึกที่: {MODEL_SAVE_PATH}")

# ==========================================
# ⚙️ 2. GPU & CONFIG
# ==========================================
print(f"🚀 TensorFlow Version: {tf.__version__}")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Ready: {len(gpus)} devices found! (RTX 3060 Mode)")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    else:
        print("⚠️ Warning: ไม่เจอ GPU")
except Exception as e:
    print(f"⚠️ GPU Error: {e}")

# 🚨 แก้ Path ให้ตรงเหมือนเดิม
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
# ลด Batch Size ลงนิดนึงเพราะ Feature เราใหญ่ขึ้น (MFCC+Mel)
BATCH_SIZE = 16 
EPOCHS = 100 

EMOTION_KEYWORDS = {
    'angry': 'angry', 'anger': 'angry', 'ag': 'angry',
    'happy': 'happy', 'joy': 'happy', 'hap': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'neutral': 'neutral', 'neu': 'neutral',
    'surprise': 'surprise', 'sur': 'surprise'
}
RAVDESS_MAP = {'03': 'happy', '04': 'sad', '05': 'angry', '01': 'neutral'}

# ==========================================
# 🛠️ 3. ADVANCED FEATURES & AUGMENTATION
# ==========================================
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape)

def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def extract_features_ultimate(data, sr):
    # 1. MFCC (128)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)
    
    # 2. Mel Spectrogram (High-Res)
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # รวมร่าง (Stacking) -> ได้ข้อมูลที่ละเอียดมาก
    # ใช้แกน 0 เพื่อซ้อนกันแนวตั้ง (128 + 128 = 256 Features)
    result = np.concatenate((mfcc, mel_db), axis=0)
    return result.T

def process_files_batch(file_list, augment=False):
    X, y = [], []
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
                if k in path_lower: label = v; break
        
        if label:
            try:
                # Load & Pad
                data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                if len(data) < SAMPLES_PER_TRACK:
                    data = np.pad(data, (0, int(SAMPLES_PER_TRACK - len(data))), 'constant')
                else:
                    data = data[:int(SAMPLES_PER_TRACK)]
                
                # 1. Original
                X.append(extract_features_ultimate(data, sr))
                y.append(label)
                
                # 2. Augment (Only for Train)
                if augment:
                    # Noise
                    X.append(extract_features_ultimate(noise(data), sr))
                    y.append(label)
                    # Pitch
                    X.append(extract_features_ultimate(pitch(data, sr), sr))
                    y.append(label)
            except Exception as e:
                print(f"Error: {file_path} - {e}")

        if (i+1) % 50 == 0:
            print(f"   Processing {i+1}/{len(file_list)}...", end='\r')
            
    return np.array(X), np.array(y)

# ==========================================
# 📥 4. LOAD & SPLIT (LEAKAGE PROOF)
# ==========================================
print("📂 กำลังค้นหาไฟล์...")
all_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(('.wav', '.mp3')):
            all_files.append(os.path.join(root, file))

if not all_files:
    print("❌ ไม่พบไฟล์เสียง! เช็ค Path ดีๆ ครับ")
    exit()

# แบ่งไฟล์ก่อนทำอะไรทั้งสิ้น (กันข้อสอบรั่ว)
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

print(f"\n🔄 สร้างข้อมูล Train (High-Res Features + Augmentation)...")
X_train, y_train_raw = process_files_batch(train_files, augment=True)

print(f"\n🔄 สร้างข้อมูล Test (Original Only)...")
X_test, y_test_raw = process_files_batch(test_files, augment=False)

# Encode Labels
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train_raw))
y_test = to_categorical(lb.transform(y_test_raw))
with open(LABEL_SAVE_PATH, 'wb') as f:
    pickle.dump(lb, f)

# Normalize
scaler = StandardScaler()
N, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], T, F)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# ==========================================
# 🧠 5. MODEL (Bi-LSTM + CNN Ultimate)
# ==========================================
model = Sequential()
# Conv Layers (รับข้อมูลขนาดใหญ่ได้)
model.add(Conv1D(256, 5, padding='same', activation='relu', input_shape=(T, F)))
model.add(BatchNormalization()); model.add(MaxPooling1D(2)); model.add(Dropout(0.3))

model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(BatchNormalization()); model.add(MaxPooling1D(2)); model.add(Dropout(0.3))

# Bi-LSTM Layers
model.add(Bidirectional(LSTM(128, return_sequences=True))); model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64))); model.add(Dropout(0.3))

# Output
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(len(lb.classes_), activation='softmax', dtype='float32'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 🛑 6. TRAINING (Safety Save Enabled)
# ==========================================
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

print("\n🚀 START TRAINING (Ultimate Mode)...")
history = model.fit(
    X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1
)

# ==========================================
# 📊 7. REPORT
# ==========================================
# โหลดไฟล์ที่เซฟไว้ (ถ้ามี) หรือใช้สถานะล่าสุด
if os.path.exists(MODEL_SAVE_PATH):
    print("\n🏆 Loading Best Model from Disk...")
    try:
        model.load_weights(MODEL_SAVE_PATH)
    except:
        print("⚠️ โหลดไฟล์ไม่สำเร็จ ใช้โมเดลล่าสุดแทน")

print("\n📊 Final Evaluation (Test Set):")
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=lb.classes_))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Ultimate Model Accuracy')
plt.legend()
plt.show()

print(f"\n✅ เสร็จสิ้น! ไฟล์โมเดลอยู่ที่: {MODEL_SAVE_PATH}")