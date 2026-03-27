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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
                                     Bidirectional, Input, MultiHeadAttention, LayerNormalization, Add)
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
        print(f"✅ GPU Ready: {len(gpus)} devices found! (RTX 3060 High-Performance Mode)")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("⚡ Mixed Precision: ENABLED")
    else:
        print("⚠️ Warning: ไม่เจอ GPU")
except Exception as e:
    print(f"⚠️ GPU Error: {e}")

# ==========================================
# ⚙️ 2. ULTIMATE CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "dataset")

SAMPLE_RATE = 22050
DURATION = 3 # วินาที
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ปรับจูน Batch & Epoch
BATCH_SIZE = 32
EPOCHS = 150 

# Keywords
EMOTION_KEYWORDS = {
    'angry': 'angry', 'anger': 'angry', 'ag': 'angry',
    'happy': 'happy', 'joy': 'happy', 'hap': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'neutral': 'neutral', 'neu': 'neutral',
    'surprise': 'surprise', 'sur': 'surprise'
}
RAVDESS_MAP = {'03': 'happy', '04': 'sad', '05': 'angry', '01': 'neutral'}

# ==========================================
# 🛠️ 3. DATA AUGMENTATION (ปั๊มข้อมูล)
# ==========================================
def noise(data):
    """เติม Noise เบาๆ ให้เสียงดูสมจริงขึ้น"""
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape)
    return data

def stretch(data, rate=0.8):
    """ยืดเสียงให้ช้าลงหรือเร็วขึ้น"""
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    """เลื่อนคลื่นเสียงซ้ายขวา"""
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    """ปรับเสียงสูงต่ำ"""
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# ==========================================
# 🔍 4. ADVANCED FEATURE EXTRACTION
# ==========================================
def extract_features(data, sr):
    # 1. MFCC 128 coefficients — จับลักษณะเสียงพูดหลัก
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)

    # 2. Mel Spectrogram — ความถี่ตามการได้ยินของมนุษย์
    mel = librosa.feature.melspectrogram(y=data, sr=sr)

    # 3. ZCR — ความถี่ที่สัญญาณตัดแกน 0 (บอกความแหลม/ทุ้มของเสียง)
    zcr = librosa.feature.zero_crossing_rate(data)

    # 4. RMS Energy — ความดังเสียงในแต่ละช่วงเวลา
    rms = librosa.feature.rms(y=data)

    # 5. Spectral Centroid — จุดศูนย์กลางความถี่ (เสียงสว่าง/มืด)
    sc = librosa.feature.spectral_centroid(y=data, sr=sr)

    # 6. Spectral Rolloff — ความถี่ที่พลังงาน 85% อยู่ใต้จุดนี้
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)

    # รวม: MFCC(128) + Mel(128) + ZCR(1) + RMS(1) + SC(1) + Rolloff(1) = 260 features
    result = np.concatenate((mfcc, mel, zcr, rms, sc, rolloff), axis=0)
    return result.T  # shape: (Time, 260)

def process_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Padding/Truncating
        if len(y) < SAMPLES_PER_TRACK:
            padding = int(SAMPLES_PER_TRACK) - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            y = y[:int(SAMPLES_PER_TRACK)]
            
        features = []
        
        # 1. ข้อมูลดิบ (Original)
        features.append(extract_features(y, sr))
        
        # 2. สร้างข้อมูลเทียม (Augmentation) -> เพิ่มข้อมูล 2 แบบ
        # แบบ A: เติม Noise
        noise_data = noise(y)
        features.append(extract_features(noise_data, sr))
        
        # แบบ B: ปรับ Pitch (เสียงสูงขึ้น)
        pitch_data = pitch(y, sr)
        features.append(extract_features(pitch_data, sr))
        
        return features # ส่งกลับมาเป็น List (1 ไฟล์ ได้ 3 ข้อมูล)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ==========================================
# 📥 5. LOAD & PREPARE DATA
# ==========================================
def load_data_ultimate(root_path):
    X, y = [], []
    print(f"⏳ เริ่มกระบวนการสแกนและสร้างข้อมูลเทียม (Augmentation Mode)...")
    
    all_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                all_files.append(os.path.join(root, file))
    
    print(f"   -> พบไฟล์ต้นฉบับ {len(all_files)} ไฟล์ (จะคูณ 3 ด้วย Augmentation)")
    
    for i, file_path in enumerate(all_files):
        # หา Label
        path_lower = file_path.lower().replace('\\', '/')
        filename = os.path.basename(path_lower)
        parts = filename.split('-')
        label = None
        
        if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
            label = RAVDESS_MAP[parts[2]]
        else:
            for keyword, emotion in EMOTION_KEYWORDS.items():
                if keyword in path_lower:
                    label = emotion
                    break
        
        if label is not None:
            # ได้ข้อมูลกลับมา 3 ชุด (Original, Noise, Pitch)
            feats_list = process_file(file_path) 
            if feats_list:
                for feat in feats_list:
                    X.append(feat)
                    y.append(label)
        
        if (i+1) % 50 == 0:
            print(f"   Processed {i+1}/{len(all_files)} original files...", end='\r')

    print(f"\n✅ เสร็จสิ้น! ได้ข้อมูล Training ทั้งหมด: {len(X)} samples (จากเดิม {len(all_files)})")
    return np.array(X), np.array(y)

# Load Data
X, y = load_data_ultimate(DATA_PATH)

if len(X) == 0:
    print("❌ ไม่พบข้อมูล!")
    exit()

# Encode Label
lb = LabelEncoder()
y_encoded = to_categorical(lb.fit_transform(y))
print(f"🏷️ Classes: {lb.classes_}")
with open(os.path.join(CURRENT_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(lb, f)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Normalize Data (สำคัญมากสำหรับ Deep Learning)
# ต้อง Reshape เป็น 2D เพื่อ Fit Scaler แล้วคืนรูปเดิม
scaler = StandardScaler()
N, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)
with open(os.path.join(CURRENT_DIR, 'rtx_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
# ใช้ Scaler ตัวเดิมกับ Test/Val (ห้าม Fit ใหม่)
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape[0], T, F)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], T, F)

# ==========================================
# 🧠 6. MODEL ARCHITECTURE (CNN + BiLSTM + Self-Attention)
# ==========================================
input_shape = (X_train.shape[1], X_train.shape[2])

# --- Functional API (รองรับ Attention layer) ---
inputs = Input(shape=input_shape)

# CNN Block 1 — สกัด Pattern ระยะสั้น
x = Conv1D(256, kernel_size=5, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

# CNN Block 2 — สกัด Pattern ระดับกลาง
x = Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

# BiLSTM — จับบริบทเวลาจากซ้ายและขวา
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.3)(x)

# Self-Attention Block — โฟกัสเฉพาะช่วงเวลาที่สำคัญของอารมณ์
attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = LayerNormalization()(Add()([x, attn_out]))  # Residual + Norm
x = Dropout(0.2)(x)

# BiLSTM ชั้นสุดท้าย — สรุปข้อมูล
x = Bidirectional(LSTM(64))(x)
x = Dropout(0.3)(x)

# Dense Output
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
outputs = Dense(len(lb.classes_), activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 🛑 7. TRAINING
# ==========================================
callbacks = [
    ModelCheckpoint(os.path.join(CURRENT_DIR, "ultimate_model_rtx.keras"), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
]

print(f"\n🚀 START TRAINING (Ultimate Mode)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# 📊 8. EVALUATION
# ==========================================
model.load_weights(os.path.join(CURRENT_DIR, "ultimate_model_rtx.keras"))
print("\n🏆 Loading Best Model...")

# Accuracy & Loss Graph
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Real Accuracy (with Augmentation)')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend(); plt.grid(True)
plt.show()

# Confusion Matrix สวยๆ
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title('Final Confusion Matrix (The Truth)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("\n📊 Final Report:")
print(classification_report(y_true, y_pred_classes, target_names=lb.classes_))