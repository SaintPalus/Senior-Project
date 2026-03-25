import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import sys
import seaborn as sns

# ตั้งค่า Encoding
sys.stdout.reconfigure(encoding='utf-8')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Bidirectional, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# ⚙️ 1. GPU & SYSTEM SETUP
# ==========================================
print(f"🚀 TensorFlow Version: {tf.__version__}")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Ready: {len(gpus)} devices found!")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("⚡ Mixed Precision: ENABLED")
    else:
        print("⚠️ Warning: ไม่เจอ GPU")
except Exception as e:
    print(f"⚠️ GPU Error: {e}")

# ==========================================
# ⚙️ 2. CONFIGURATION (Updated for Multi-Lang)
# ==========================================
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"
SAMPLE_RATE = 22050
DURATION = 3 
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
BATCH_SIZE = 32
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
# 🔍 3. FEATURE EXTRACTION (Enhanced for Multi-Lang)
# ==========================================
def extract_features(data, sr):
    # 1. MFCC (128 dims)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)
    
    # 2. Mel Spectrogram (128 dims)
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    
    # 3. Spectral Contrast (สำคัญสำหรับภาษาที่มีโทนเสียง/วรรณยุกต์ เช่น ไทย)
    contrast = librosa.feature.spectral_contrast(y=data, sr=sr)
    
    # รวมฟีเจอร์: ผลลัพธ์จะได้มิติที่ใหญ่ขึ้นเพื่อครอบคลุมความต่างของภาษา
    result = np.concatenate((mfcc, mel, contrast), axis=0) 
    return result.T

def process_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, int(SAMPLES_PER_TRACK) - len(y)), 'constant')
        else:
            y = y[:int(SAMPLES_PER_TRACK)]
            
        # สร้าง Augmentation (Original + Noise)
        features = []
        features.append(extract_features(y, sr)) # Original
        
        # เพิ่ม Noise เฉพาะตอน Train (ในฟังก์ชัน load_data จะคุมอีกที)
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y_noise = y + noise_amp * np.random.normal(size=y.shape)
        features.append(extract_features(y_noise, sr))
        
        return features
    except Exception as e:
        return None

# ==========================================
# 📥 4. LOAD DATA (Logic for Language Split)
# ==========================================
def load_data_multilang(root_path):
    X, y_labels = [], []
    
    # ไล่หาจากโฟลเดอร์ภาษา (เช่น dataset/English, dataset/Thai)
    for lang_folder in os.listdir(root_path):
        lang_path = os.path.join(root_path, lang_folder)
        if not os.path.isdir(lang_path): continue
        
        print(f"🌐 Processing Language: {lang_folder}")
        
        for root, dirs, files in os.walk(lang_path):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3')):
                    file_path = os.path.join(root, file)
                    path_lower = file_path.lower().replace('\\', '/')
                    
                    # Logic แยกอารมณ์
                    label = None
                    filename = os.path.basename(path_lower)
                    parts = filename.split('-')
                    
                    if 'ravdess' in path_lower and len(parts) >= 3:
                        label = RAVDESS_MAP.get(parts[2])
                    else:
                        for kw, em in EMOTION_KEYWORDS.items():
                            if kw in path_lower:
                                label = em
                                break
                    
                    if label:
                        feats = process_file(file_path)
                        if feats:
                            for f in feats:
                                X.append(f)
                                # บันทึก Label แบบ "ภาษา_อารมณ์" หรือแค่ "อารมณ์" 
                                # ในที่นี้เราใช้ Emotion เป็นหลัก แต่เอา Data จากทุกภาษามารวมกัน
                                y_labels.append(label)
                                
    return np.array(X), np.array(y_labels)

# --- EXECUTION ---
X, y = load_data_multilang(DATA_PATH)

lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(lb, f)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)

# Scaling
scaler = StandardScaler()
N, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], T, F)

# ==========================================
# 🧠 5. MODEL ARCHITECTURE (PRO VERSION)
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])
model = Sequential([
    Input(shape=input_shape),
    
    # ส่วนสกัดรูปคลื่น
    Conv1D(512, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(4),
    Dropout(0.3),
    
    Conv1D(256, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(4),
    Dropout(0.3),
    
    # ส่วนวิเคราะห์ลำดับเสียง (สำคัญมากสำหรับการแยกภาษา)
    Bidirectional(LSTM(256, return_sequences=True)),
    Bidirectional(LSTM(128)),
    
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(len(lb.classes_), activation='softmax', dtype='float32')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 🛑 6. TRAINING & EVAL
# ==========================================
callbacks = [
    ModelCheckpoint("multilang_emotion_model.keras", save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    validation_split=0.15, callbacks=callbacks)

print("\n✅ Training Complete! Model saved as 'multilang_emotion_model.keras'")