import os
import sys

# ตั้งค่าให้โชว์ภาษาไทยได้
sys.stdout.reconfigure(encoding='utf-8')

# 🚨 แก้ Path ให้ตรงกับเครื่องคุณ
DATA_PATH = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset"

# คำศัพท์ที่ใช้จับอารมณ์จากชื่อโฟลเดอร์
EMOTION_KEYWORDS = {
    'angry': 'angry', 
    'happy': 'happy', 
    'sad': 'sad', 
    'neutral': 'neutral', 
    'surprise': 'surprise'
}

print(f"🕵️‍♂️ กำลังสแกนโฟลเดอร์: {DATA_PATH}\n")

count_stats = {k: 0 for k in EMOTION_KEYWORDS.values()}
total_files = 0

for root, dirs, files in os.walk(DATA_PATH):
    # ข้ามโฟลเดอร์ที่ไม่เกี่ยวกับเสียง
    audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    
    if not audio_files:
        continue
        
    # เช็ค Path ของโฟลเดอร์ปัจจุบัน
    current_path = root.lower().replace('\\', '/')
    found_label = None
    
    # ดูว่า Path นี้มีคำว่า angry, happy, etc. ซ่อนอยู่ไหม
    for key, emotion in EMOTION_KEYWORDS.items():
        # เช็คว่ามีชื่ออารมณ์อยู่ใน Path ไหม (เช่น .../dataset/angry/...)
        if f"/{key}/" in current_path or f"\\{key}\\" in current_path or current_path.endswith(key):
            found_label = emotion
            break
            
    # ถ้าเจออารมณ์ ให้บวกจำนวนไฟล์เข้าไป
    if found_label:
        count = len(audio_files)
        count_stats[found_label] += count
        total_files += count
        print(f"✅ เจอ {found_label.upper()} จำนวน {count} ไฟล์ \tที่: {root}")
    else:
        # ถ้าไม่เจออารมณ์ในชื่อโฟลเดอร์ ลองดูชื่อไฟล์ทีละตัว (เผื่อไว้)
        pass

print("\n" + "="*40)
print("📊 สรุปยอดไฟล์ที่อ่านได้ (พร้อมเทรน):")
print("="*40)
for emotion, count in count_stats.items():
    print(f"  • {emotion.upper()}: \t{count} ไฟล์")
    
print("-" * 40)
print(f"🔥 รวมทั้งหมด: \t{total_files} ไฟล์")
print("="*40)

if total_files == 0:
    print("❌ ไม่เจอไฟล์เลย! (ลองเช็คชื่อโฟลเดอร์หลักดูครับ ต้องชื่อ angry, happy, sad, neutral, surprise)")
else:
    print("✅ ข้อมูลพร้อมมาก! รันไฟล์ Train_Universal_Super.py ได้เลยครับ")