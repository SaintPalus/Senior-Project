from datasets import load_dataset, Audio
import soundfile as sf
import os
import io

# ---------------------------------------------------------
# ตั้งค่า: เลือกภาษาที่จะโหลด
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_name = "dataset/korean-voice-emotion-dataset"
output_folder = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset", "korean_drama")

# ---------------------------------------------------------
# เริ่มกระบวนการโหลดและแยกไฟล์
# ---------------------------------------------------------
print(f"⏳ กำลังดาวน์โหลดข้อมูลจาก {dataset_name} (Mode: Manual Decode)...")

# 1. โหลดแบบ Streaming
ds = load_dataset(dataset_name, split="train", streaming=True)

# ✅ KEY FIX: สั่งปิดการถอดรหัสออโต้ (เพื่อเลี่ยง Error torchcodec)
print("🔧 กำลังตั้งค่าเพื่อ bypass ตัวถอดรหัส...")
ds = ds.cast_column("audio", Audio(decode=False))

print("✅ เริ่มทยอยโหลดและบันทึกไฟล์...")

for i, item in enumerate(ds):
    try:
        # 2. ดึงข้อมูลอารมณ์
        # เช็กชื่อคอลัมน์อารมณ์
        if 'emotion' in item:
            emotion = item['emotion']
        elif 'label' in item:
            emotion = item['label']
        else:
            emotion = "unknown"

        # 3. ถอดรหัสเสียงด้วยตัวเอง (Manual Decode)
        audio_data = item['audio']
        audio_bytes = audio_data['bytes'] # รับมาเป็นก้อนข้อมูลดิบ
        
        # ใช้ soundfile แปลงก้อนข้อมูลดิบเป็นเสียง
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        
        # 4. สร้างโฟลเดอร์ตามชื่ออารมณ์
        target_dir = os.path.join(output_folder, str(emotion).lower())
        os.makedirs(target_dir, exist_ok=True)
        
        # 5. บันทึกไฟล์เสียง .wav
        filename = f"kor_clip_{i:04d}.wav"
        file_path = os.path.join(target_dir, filename)
        
        sf.write(file_path, data, samplerate)
        
        # แสดงความคืบหน้า
        if i % 50 == 0:
            print(f"   Saved {i} files...", end='\r')
            
    except Exception as e:
        # บางไฟล์อาจเสีย ข้ามไปเลย
        pass

print(f"\n🎉 เสร็จเรียบร้อย! ไฟล์ถูกแยกไปที่: {output_folder}")