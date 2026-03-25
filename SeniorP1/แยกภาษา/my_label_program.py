import os
import shutil
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import soundfile as sf

# ==========================================
# ⚙️ ตั้งค่าโฟลเดอร์
# ==========================================
UNLABELED_DIR = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset" 
LABELED_DIR = r"C:\Users\THINK_01\66070131_SeniorP1\SeniorP1\dataset\Cleaned_Data"

# อารมณ์หลักที่ใช้ในการสแกนอัตโนมัติ
CORE_EMOTIONS = ["Angry", "Happy", "Sad", "Neutral", "Surprise", "Fear"]
# ปุ่มสำหรับ GUI
GUI_EMOTIONS = CORE_EMOTIONS + ["🗑️ ลบทิ้ง (เสียงเสีย)"]

# ==========================================
# 🤖 ส่วนที่ 1: ระบบตรวจสอบและย้ายอัตโนมัติ (Auto-Sorter)
# ==========================================
def auto_sort_files():
    manual_files = []
    auto_moved = 0
    
    # สร้างโฟลเดอร์รอไว้เลย
    os.makedirs(LABELED_DIR, exist_ok=True)
    for emo in CORE_EMOTIONS:
        os.makedirs(os.path.join(LABELED_DIR, emo), exist_ok=True)
        
    print("⏳ [1/2] กำลังสแกนหาไฟล์ที่ระบุอารมณ์ไว้แล้ว (จะข้ามโฟลเดอร์ English)...")
    
    for root_dir, dirs, files in os.walk(UNLABELED_DIR):
        # 🌟 จุดที่อัปเดต: ข้ามโฟลเดอร์ที่จัดระเบียบเสร็จแล้ว และ ข้ามโฟลเดอร์ English
        if "Cleaned_Data" in root_dir or "English" in root_dir:
            continue
            
        for file in files:
            if file.lower().endswith(('.wav', '.flac')):
                full_path = os.path.join(root_dir, file)
                
                # ตัด Path หลักทิ้ง เพื่อดูแค่ชื่อโฟลเดอร์ย่อยและชื่อไฟล์
                rel_path = os.path.relpath(full_path, UNLABELED_DIR).lower()
                
                # ค้นหาว่ามีชื่ออารมณ์ซ่อนอยู่ไหม?
                found_emotions = []
                for emo in CORE_EMOTIONS:
                    if emo.lower() in rel_path:
                        found_emotions.append(emo)
                        
                # ถ้าระบุชัดเจนว่าเป็น "อารมณ์เดียว" -> ย้ายอัตโนมัติ!
                if len(found_emotions) == 1:
                    target_emo = found_emotions[0]
                    
                    # ตั้งชื่อไฟล์ใหม่ให้ฉลาดขึ้น (เอาชื่อโฟลเดอร์พ่อมาแปะ ป้องกันไฟล์ชื่อซ้ำกัน)
                    parent_folder = os.path.basename(os.path.dirname(full_path))
                    new_filename = f"{target_emo}_{parent_folder}_{file}"
                    
                    new_path = os.path.join(LABELED_DIR, target_emo, new_filename)
                    
                    # กันเหนียว: ถ้าชื่อซ้ำให้เติม _copy
                    if os.path.exists(new_path):
                        base, ext = os.path.splitext(new_filename)
                        new_path = os.path.join(LABELED_DIR, target_emo, f"{base}_copy{ext}")
                    
                    shutil.move(full_path, new_path)
                    auto_moved += 1
                else:
                    # ถ้าหาไม่เจอ หรือเจอมันเขียนตีกัน -> โยนเข้าหมวดให้คนฟัง
                    manual_files.append(full_path)
                    
    return manual_files, auto_moved

# ==========================================
# 🧑‍💻 ส่วนที่ 2: หน้าต่าง GUI สำหรับไฟล์ที่ต้องทำมือ
# ==========================================
class AudioLabelerApp:
    def __init__(self, root, audio_files):
        self.root = root
        self.root.title("🎙️ Senior Project: Manual Audio Labeler")
        self.root.geometry("600x450")
        
        self.audio_files = audio_files
        self.current_index = 0

        self.info_label = tk.Label(root, text="กำลังโหลดข้อมูล...", font=("Arial", 12), wraplength=550)
        self.info_label.pack(pady=20)

        self.play_btn = tk.Button(root, text="▶️ เล่นเสียงซ้ำ (Play)", command=self.play_audio, font=("Arial", 12), bg="lightblue")
        self.play_btn.pack(pady=10)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=20)

        for emo in GUI_EMOTIONS:
            btn = tk.Button(self.btn_frame, text=emo, width=15, font=("Arial", 10),
                            command=lambda e=emo: self.label_and_next(e))
            btn.pack(side=tk.TOP, pady=2)
            
        self.load_next_file()

    def load_next_file(self):
        sd.stop()
        if self.current_index < len(self.audio_files):
            self.current_filepath = self.audio_files[self.current_index]
            filename_only = os.path.basename(self.current_filepath)
            self.info_label.config(text=f"กำลังเล่นไฟล์: {filename_only}\n({self.current_index + 1} / {len(self.audio_files)})")
            self.play_audio()
        else:
            self.info_label.config(text="🎉 ทำ Label ครบทุกไฟล์แล้ว! ยอดเยี่ยมมาก ปิดโปรแกรมได้เลย")
            self.play_btn.config(state=tk.DISABLED)

    def play_audio(self):
        try:
            data, fs = sf.read(self.current_filepath)
            sd.play(data, fs)
        except Exception as e:
            print(f"ข้ามไฟล์ที่เล่นไม่ได้: {e}")

    def label_and_next(self, emotion):
        sd.stop()
        old_path = self.current_filepath
        filename_only = os.path.basename(old_path)
        
        if "ลบทิ้ง" in emotion:
            os.remove(old_path)
        else:
            new_filename = f"{emotion}_manual_{filename_only}"
            new_path = os.path.join(LABELED_DIR, emotion, new_filename)
            shutil.move(old_path, new_path)
            
        self.current_index += 1
        self.load_next_file()

# ==========================================
# 🚀 รันโปรแกรมหลัก
# ==========================================
if __name__ == "__main__":
    files_to_manual_label, auto_moved_count = auto_sort_files()
    
    print(f"\n✅ [เสร็จสิ้นขั้นตอนที่ 1] AI ย้ายไฟล์ที่มี Label อัตโนมัติไปแล้วทั้งหมด: {auto_moved_count} ไฟล์!")
    
    if len(files_to_manual_label) > 0:
        print(f"⚠️ [ขั้นตอนที่ 2] พบไฟล์ที่ระบุอารมณ์ไม่ได้ {len(files_to_manual_label)} ไฟล์... กำลังเปิดหน้าต่างให้คุณฟังเสียง!")
        root = tk.Tk()
        app = AudioLabelerApp(root, files_to_manual_label)
        root.mainloop()
    else:
        print("🎉 ไม่มีไฟล์ที่ต้องทำมือเลย! Data ของคุณสะอาดและถูกจัดหมวดหมู่ 100% แล้วครับ!")