import os
import librosa
import numpy as np

# 1. กำหนด Path หลักของโฟลเดอร์ dataset
base_dataset_path = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"

# 2. กำหนดเฉพาะโฟลเดอร์ภาษาใน Asia ที่เราต้องการใช้งาน
target_languages = ['Chinese', 'Japan', 'Korean', 'Thai']

# 3. โฟลเดอร์สำหรับเก็บไฟล์ Feature (.npy) ที่สกัดเสร็จแล้ว
output_feature_dir = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\extracted_features"
os.makedirs(output_feature_dir, exist_ok=True)

print("🚀 เริ่มกระบวนการสกัด Feature (รองรับ .wav และ .flac)...")

total_processed = 0

for lang_folder in target_languages:
    lang_path = os.path.join(base_dataset_path, lang_folder)
    
    if not os.path.exists(lang_path):
        print(f"⚠️ ไม่พบโฟลเดอร์: {lang_folder} ข้ามไป...")
        continue
        
    print(f"\n📂 กำลังจัดการภาษา: {lang_folder}")
    
    for root, dirs, files in os.walk(lang_path):
        for file in files:
            # 🌟 จุดที่แก้ไข 1: เช็คว่าไฟล์ลงท้ายด้วย .wav หรือ .flac (แปลงเป็นตัวเล็กก่อนเผื่อเจอ .FLAC)
            if file.lower().endswith(('.wav', '.flac')):
                file_path = os.path.join(root, file)
                
                try:
                    # โหลดไฟล์เสียง (.flac หรือ .wav ก็โหลดด้วยคำสั่งเดียวกันได้เลย)
                    y, sr = librosa.load(file_path, sr=16000)
                    
                    # สกัด Mel-Spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # 🌟 จุดที่แก้ไข 2: ใช้ os.path.splitext เพื่อตัดนามสกุลไฟล์เก่าทิ้งอย่างปลอดภัย
                    # สมมติไฟล์ชื่อ "audio.flac" จะได้ base_name = "audio"
                    base_name, _ = os.path.splitext(file)
                    output_filename = f"{lang_folder}_{base_name}.npy"
                    
                    output_path = os.path.join(output_feature_dir, output_filename)
                    
                    # บันทึกเป็นไฟล์ .npy
                    np.save(output_path, mel_spec_db)
                    total_processed += 1
                    
                    if total_processed % 50 == 0:
                        print(f"   -> ประมวลผลไปแล้ว {total_processed} ไฟล์...")
                        
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {file}: {e}")

print(f"\n✅ เสร็จสิ้น! ประมวลผลไฟล์เสียงทั้งหมด {total_processed} ไฟล์")