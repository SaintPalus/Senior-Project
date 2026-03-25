import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import sys
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# ⚙️ 1. SETUP
# ==========================================
# 🚨 แก้ Path ให้ตรงเหมือนเดิม
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
# 🛠️ 2. FUNCTIONS
# ==========================================
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sr):
    # ใช้ MFCC + Mel Spectrogram
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40) # ลดเหลือ 40 เพื่อความเสถียร
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Resize Mel ให้เท่า MFCC เพื่อ stack
    # (ในที่นี้เราจะใช้แค่ MFCC เป็นหลักก่อนเพื่อลดความซับซ้อนและ Memory)
    return mfcc.T

def get_label(file_path):
    path_lower = file_path.lower().replace('\\', '/')
    filename = os.path.basename(path_lower)
    parts = filename.split('-')
    if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
        return RAVDESS_MAP[parts[2]]
    for k, v in EMOTION_KEYWORDS.items():
        if k in path_lower: return v
    return None

def process_files(file_list, augment=False):
    X, y = [], []
    for i, file_path in enumerate(file_list):
        label = get_label(file_path)
        if label is None: continue
            
        try:
            # Load Audio
            data, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            if len(data) < SAMPLES_PER_TRACK:
                data = np.pad(data, (0, int(SAMPLES_PER_TRACK - len(data))), 'constant')
            else:
                data = data[:int(SAMPLES_PER_TRACK)]
            
            # 1. Original Data
            X.append(extract_features(data, sr))
            y.append(label)
            
            # 2. Augmentation (ทำเฉพาะ Train Set)
            if augment:
                # Noise
                X.append(extract_features(noise(data), sr))
                y.append(label)
                # Pitch
                X.append(extract_features(pitch(data, sr), sr))
                y.append(label)
                
        except Exception as e:
            print(f"❌ Error: {file_path}")
            
        if (i+1) % 50 == 0:
            print(f"   Processing {i+1}/{len(file_list)} files...", end='\r')
            
    return np.array(X), np.array(y)

# ==========================================
# 🚀 3. LOAD & SPLIT (CORRECT WAY)
# ==========================================
# 1. หาไฟล์ทั้งหมดก่อน
all_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(('.wav', '.mp3')):
            all_files.append(os.path.join(root, file))

print(f"📂 พบไฟล์ทั้งหมด: {len(all_files)}")

# 2. แบ่งไฟล์เป็น Train/Test *ก่อน* ทำอะไรทั้งสิ้น (กันข้อสอบรั่ว)
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

print(f"🔄 กำลังประมวลผล Train Set (พร้อม Augmentation)...")
X_train, y_train_raw = process_files(train_files, augment=True) # ปั๊มข้อมูลเฉพาะ Train

print(f"\n🔄 กำลังประมวลผล Test Set (ห้าม Augmentation)...")
X_test, y_test_raw = process_files(test_files, augment=False) # Test ของจริงต้องไม่แต่งเติม

# Encode Labels
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train_raw))
y_test = to_categorical(lb.transform(y_test_raw)) # ใช้ตัวแปลงเดียวกับ Train
print(f"\n🏷️ Classes: {lb.classes_}")

# Normalize
scaler = StandardScaler()
N, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)
# ใช้ Scaler ของ Train มาปรับ Test (ห้าม Fit ใหม่)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape[0], T, F)

# Split Validation จาก Train อีกที
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# ==========================================
# 🧠 4. MODEL (Bi-LSTM Optimized)
# ==========================================
model = Sequential()
model.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(T, F)))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(len(lb.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 🛑 5. TRAIN & EVALUATE
# ==========================================
callbacks = [
    ModelCheckpoint("real_acc_model.keras", save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]

print("\n🚀 Starting Training (Real Accuracy Mode)...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Report
model.load_weights("real_acc_model.keras")
print("\n📊 Real World Evaluation:")
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=lb.classes_))

# Graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy (Real)')
plt.legend()
plt.show()