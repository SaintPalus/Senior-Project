import sounddevice as sd
from scipy.io.wavfile import write
import time
import os

# ตั้งค่ามาตรฐานสำหรับ AI
fs = 16000  # Sample Rate 16kHz
seconds = 4  # ระยะเวลาอัดเสียง (4 วินาที กำลังพอดีสำหรับ 1 ประโยค)

print("🎙️ เตรียมตัวพูดประโยคอะไรก็ได้ พร้อมใส่อารมณ์ (โกรธ, ดีใจ, เศร้า, หรือตกใจ)...")
time.sleep(2)

print("⏳ จะเริ่มอัดเสียงใน...")
for i in range(3, 0, -1):
    print(f"   {i}...")
    time.sleep(1)

print("\n🔴 เริ่มอัดเสียง! (พูดเลย!!)")
# สั่งเปิดไมค์อัดเสียง
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # รอจนกว่าจะครบ 4 วินาที
print("✅ อัดเสียงเสร็จเรียบร้อย!")

# กำหนดโฟลเดอร์ที่จะเซฟไฟล์ (ผมตั้งให้ไปเซฟในโฟลเดอร์ Thai ของคุณ)
output_dir = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset\Thai"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "my_test_voice.wav")

# เซฟเป็นไฟล์ .wav
write(output_path, fs, myrecording)

print(f"💾 บันทึกไฟล์เสียงทดสอบไว้ที่:\n👉 {output_path}")
print("\n🎯 ขั้นตอนต่อไป: นำ Path ด้านบนนี้ไปใส่ในไฟล์ test_real_accuracy.py เพื่อให้ AI ทายอารมณ์ได้เลยครับ!")