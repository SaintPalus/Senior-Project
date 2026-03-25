import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ==========================================
# 🌟 ส่วนที่ 1: เช็คและตั้งค่าการใช้งาน GPU ให้ชัวร์!
# ==========================================
print("🔍 กำลังตรวจสอบการ์ดจอ (GPU)...")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ ยืนยัน! ระบบพบ GPU จำนวน {len(gpus)} ตัว พร้อมลุยครับ:")
    for i, gpu in enumerate(gpus):
        print(f"   💻 GPU {i+1}: {gpu.name}")
        try:
            # ทริค: ตั้งค่าให้ TensorFlow ค่อยๆ ดึง VRAM ของการ์ดจอไปใช้เท่าที่จำเป็น
            # ป้องกันปัญหาแย่งแรมกับโปรแกรมอื่นจนโมเดลพัง (Out of Memory)
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"   ⚠️ ตั้งค่า Memory Growth ไม่สำเร็จ: {e}")
else:
    print("⚠️ คำเตือน: ไม่พบ GPU! ระบบจะบังคับใช้ CPU ในการ Train ซึ่งจะช้ากว่าปกติมาก")
    print("   👉 แนะนำให้เช็คว่าลง CUDA Toolkit และ cuDNN ตรงกับเวอร์ชันของ TensorFlow หรือยัง")
print("==========================================\n")

# ==========================================
# 🌟 ส่วนที่ 2: โหลดข้อมูลและเตรียม Train
# ==========================================
feature_dir = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\extracted_features"
MAX_TIME_STEPS = 130 

X = []
y = []

print("📂 กำลังโหลดข้อมูล .npy และปรับขนาด (Padding/Truncating)...")
for file in os.listdir(feature_dir):
    if file.endswith('.npy'):
        mel_spec = np.load(os.path.join(feature_dir, file))
        
        if mel_spec.shape[1] < MAX_TIME_STEPS:
            pad_width = MAX_TIME_STEPS - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :MAX_TIME_STEPS]
            
        X.append(mel_spec)
        
        # สมมติชื่อไฟล์ Thai_Angry_001.npy ให้ดึงคำว่า Angry มาเป็น Label
        parts = file.split('_')
        if len(parts) >= 2:
            emotion = parts[1]
            y.append(emotion)

X = np.array(X)
X = X[..., np.newaxis] 

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"✅ ข้อมูลพร้อมเทรน! รูปแบบ X: {X.shape}, รูปแบบ y: {y_categorical.shape}")
print(f"🎯 อารมณ์ที่ AI จะเรียนรู้: {label_encoder.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ==========================================
# 🌟 ส่วนที่ 3: สร้างและ Train AI
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    
    layers.Dense(len(label_encoder.classes_), activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 เริ่มต้นการฝึกสอน AI (Training)... สังเกตการทำงานของการ์ดจอได้เลย!")
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n🏆 ความแม่นยำ (Test Accuracy): {test_acc*100:.2f}%")

model.save("asian_emotion_cnn_model.h5")
print("💾 บันทึกโมเดลเสร็จสิ้น (asian_emotion_cnn_model.h5)")