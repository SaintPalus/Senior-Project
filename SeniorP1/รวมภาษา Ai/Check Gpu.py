import tensorflow as tf

print("------------------------------------------------")
print("TensorFlow Version:", tf.__version__)
print("------------------------------------------------")

# เช็กรายการอุปกรณ์
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ เย้! เจอ GPU ทั้งหมด {len(gpus)} ตัว")
    for gpu in gpus:
        print(f"   Name: {gpu.name}")
    print("🚀 พร้อมรันแบบ Turbo แล้ว!")
else:
    print("❌ ไม่เจอ GPU เลย (ระบบกำลังใช้ CPU)")
    print("   คำแนะนำ: เช็ก CUDA/cuDNN หรือลองใช้ Google Colab")
print("------------------------------------------------")