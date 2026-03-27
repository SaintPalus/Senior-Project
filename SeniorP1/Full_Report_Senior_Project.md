# รายงานโครงงานนักศึกษา
## ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษาด้วยเทคนิคการเรียนรู้เชิงลึก
### Multilingual Speech Emotion Recognition System Using Deep Learning Techniques

---

| รายการ | รายละเอียด |
|---|---|
| ชื่อโครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา |
| นักศึกษา | รหัส 66070131 |
| สาขาวิชา | วิทยาการคอมพิวเตอร์ |
| ภาคการศึกษา | 2 / 2567 |

---

## สารบัญ

| บทที่ | หน้า |
|---|---|
| บทคัดย่อ | I |
| ABSTRACT | II |
| กิตติกรรมประกาศ | III |
| สารบัญ | IV |
| สารบัญตาราง | VI |
| สารบัญรูป | VIII |
| **บทที่ 1 บทนำ** | **1** |
| 1.1 ที่มาและความสำคัญ | 1 |
| 1.2 วัตถุประสงค์ | 2 |
| 1.3 ขอบเขตของโครงงาน | 2 |
| 1.4 ขั้นตอนการดำเนินงานและแผนการดำเนินงาน | 4 |
| 1.5 ประโยชน์ที่คาดว่าจะได้รับ | 4 |
| 1.6 นิยามคำศัพท์ | 5 |
| **บทที่ 2 การทบทวนวรรณกรรมและเทคโนโลยีที่เกี่ยวข้อง** | **6** |
| 2.1 ความรู้เบื้องต้นเกี่ยวกับการรู้จำอารมณ์จากเสียง | 6 |
| 2.2 ทฤษฎีพื้นฐานด้านการประมวลผลเสียง | 8 |
| 2.3 ลักษณะทาง Prosody และความแตกต่างระหว่างภาษา | 12 |
| 2.4 สถาปัตยกรรม Deep Learning ที่ใช้ | 18 |
| 2.5 ชุดข้อมูลที่เกี่ยวข้อง | 24 |
| 2.6 สรุปผลการศึกษางานวิจัยที่เกี่ยวข้อง | 30 |
| **บทที่ 3 วิธีการดำเนินการวิจัย** | **33** |
| 3.1 ภาพรวมสถาปัตยกรรมระบบ | 33 |
| 3.2 การออกแบบ Data Pipeline | 37 |
| 3.3 สถาปัตยกรรมโมเดล CNN + Bi-LSTM | 55 |
| 3.4 การกำหนด Training Configuration | 70 |
| **บทที่ 4 ระบบต้นแบบ** | **85** |
| 4.1 ภาพรวมการออกแบบ (Overview Design) | 85 |
| 4.2 ผลการทดลองและการวิเคราะห์ปัญหา | 92 |
| 4.3 การวิเคราะห์สาเหตุที่ทำให้ไม่บรรลุเป้าหมาย | 100 |
| **บทที่ 5 สรุปผลการดำเนินงาน** | **106** |
| 5.1 สรุปผลการดำเนินงาน | 106 |
| 5.2 ปัญหาและอุปสรรคที่พบในการดำเนินงาน | 106 |
| 5.3 Future of Work — แนวทางการพัฒนาในอนาคต (แยกโมเดลตามภาษา) | 107 |
| บรรณานุกรม | 112 |
| ประวัติผู้เขียน | 115 |

---

## สารบัญตาราง

| ตาราง | รายละเอียด | หน้า |
|---|---|---|
| ตารางที่ 2.1 | คุณสมบัติ Prosody และความแตกต่างระหว่างภาษาอังกฤษกับภาษาเกาหลี | 13 |
| ตารางที่ 2.2 | เปรียบเทียบ Feature Extraction ชนิดต่างๆ สำหรับงาน SER | 19 |
| ตารางที่ 3.1 | โมเดลที่พัฒนาทั้งหมดในโครงงาน | 56 |
| ตารางที่ 3.2 | Training Configuration และเหตุผลในการเลือกใช้ | 71 |
| ตารางที่ 3.3 | เทคนิค Data Augmentation ที่ใช้ | 72 |
| ตารางที่ 4.1 | ผลการทดลองเบื้องต้นของ Multilingual Model | 92 |
| ตารางที่ 4.2 | ความสับสนระหว่างอารมณ์แต่ละชนิด (Confusion Analysis) | 96 |
| ตารางที่ 4.3 | สรุปข้อจำกัดของแนวทาง Multilingual | 101 |
| ตารางที่ 5.1 | แผนพัฒนาโมเดลแยกภาษาในอนาคต | 108 |
| ตารางที่ 5.2 | เปรียบเทียบแนวทาง Unified Model กับ Per-Language Model | 110 |

---

## สารบัญรูป

| รูป | รายละเอียด | หน้า |
|---|---|---|
| รูปที่ 2.1 | กระบวนการคำนวณ MFCC (Mel-Frequency Cepstral Coefficients) | 9 |
| รูปที่ 2.2 | ตัวอย่าง Mel Spectrogram ของเสียงอารมณ์โกรธ (ภาษาอังกฤษ vs ภาษาเกาหลี) | 10 |
| รูปที่ 2.3 | กราฟ Pitch Contour ของอารมณ์เดียวกันในสองภาษา | 14 |
| รูปที่ 3.1 | ภาพรวม Pipeline ของระบบ Multilingual SER | 34 |
| รูปที่ 3.2 | กระบวนการป้องกัน Data Leakage | 42 |
| รูปที่ 3.3 | สถาปัตยกรรมโมเดล CNN + Bidirectional LSTM | 57 |
| รูปที่ 4.1 | กราฟ Accuracy และ Loss ระหว่าง Training | 93 |
| รูปที่ 4.2 | Confusion Matrix ของ Multilingual Model | 97 |
| รูปที่ 5.1 | Pipeline ของแนวทาง Per-Language Model ในอนาคต | 108 |

---

# บทคัดย่อ

&emsp;โครงงานนี้นำเสนอการพัฒนา **ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา** (Multilingual Speech Emotion Recognition: Multilingual SER) ซึ่งเป็นหัวข้อที่มีความสำคัญในงานด้านปฏิสัมพันธ์ระหว่างมนุษย์กับคอมพิวเตอร์ (Human-Computer Interaction: HCI) แนวคิดหลักของโครงงานคือการนำข้อมูลเสียงพูดจากหลายภาษา ได้แก่ ภาษาอังกฤษ (RAVDESS Dataset) และภาษาเกาหลี (Korean Voice Emotion Dataset) มารวมกันในชุดข้อมูลเดียว แล้วฝึกสอนโมเดล Deep Learning สถาปัตยกรรม **CNN + Bidirectional LSTM** เพื่อให้ระบบสามารถจำแนกอารมณ์ 5 ประเภทคือ โกรธ (Angry) มีความสุข (Happy) เศร้า (Sad) เป็นกลาง (Neutral) และประหลาดใจ (Surprise) ได้โดยไม่ขึ้นกับภาษา

&emsp;ในระหว่างการพัฒนา ได้ค้นพบปัญหาสำคัญที่เป็นอุปสรรคหลักของโครงงานคือ **เสียงพูดของแต่ละภาษามีลักษณะทาง Prosody ที่แตกต่างกันอย่างมีนัยสำคัญ** ไม่ว่าจะเป็นรูปแบบ Pitch (F0) ความเร็วในการพูด (Speech Rate) ระดับพลังงาน (Energy Pattern) และจังหวะการพูด (Rhythm) แม้จะเป็นการแสดงออกถึงอารมณ์เดียวกัน ความแตกต่างเหล่านี้ทำให้คุณลักษณะ MFCC ที่สกัดออกมามีการผสมผสานระหว่าง "ลักษณะของอารมณ์" กับ "ลักษณะของภาษา" อย่างแยกไม่ออก ส่งผลให้โมเดลที่ฝึกสอนด้วยข้อมูลหลายภาษาพร้อมกันไม่สามารถเรียนรู้รูปแบบอารมณ์ที่เป็นสากลได้อย่างมีประสิทธิภาพ โดยให้ค่า Test Accuracy อยู่ที่เพียง **~68%** ซึ่งต่ำกว่าเป้าหมายที่ตั้งไว้ที่ 75%

&emsp;รายงานฉบับนี้นำเสนอแนวทางการดำเนินงานอย่างละเอียด ปัญหาที่ค้นพบ บทวิเคราะห์สาเหตุเชิงลึก และทิศทางการพัฒนาต่อในอนาคต (Future of Work) ซึ่งเสนอแนวทาง **การแยกฝึกสอนโมเดลตามภาษา (Per-Language Model)** ร่วมกับการใช้ Language Identifier อัตโนมัติ เพื่อแก้ไขปัญหา Prosody Mismatch ที่เกิดขึ้น

**คำสำคัญ:** การรู้จำอารมณ์จากเสียง, หลายภาษา, Prosody Mismatch, CNN, Bidirectional LSTM, MFCC, Per-Language Model

---

# ABSTRACT

&emsp;This project presents the development of a **Multilingual Speech Emotion Recognition (Multilingual SER)** system, a significant area in Human-Computer Interaction research. The core concept was to combine speech audio data from multiple languages — English (RAVDESS Dataset) and Korean (Korean Voice Emotion Dataset) — into a unified dataset, then train a **CNN + Bidirectional LSTM** Deep Learning model capable of classifying 5 emotion categories: Angry, Happy, Sad, Neutral, and Surprise, independent of the spoken language.

&emsp;During development, a critical challenge was identified as the primary obstacle: **each language possesses significantly different prosodic characteristics**, including Pitch contour (F0), Speech Rate, Energy Pattern, and Rhythm, even when expressing the same emotion. These differences cause the MFCC features extracted from the audio to be an inseparable mixture of "emotion characteristics" and "language characteristics." As a result, a model trained on mixed multilingual data cannot effectively learn universal emotion patterns. The achieved Test Accuracy was only **~68%**, falling below the target threshold of 75%.

&emsp;This report presents the detailed methodology, discovered problems, in-depth root cause analysis, and proposed Future of Work — specifically, a **Per-Language Model** approach combined with an automatic Language Identifier to resolve the Prosody Mismatch problem.

**Keywords:** Speech Emotion Recognition, Multilingual, Prosody Mismatch, CNN, Bidirectional LSTM, MFCC, Per-Language Model

---

# กิตติกรรมประกาศ

&emsp;โครงงานนี้สำเร็จลุล่วงได้ด้วยความวิริยะอุตสาหะของผู้จัดทำในการศึกษาค้นคว้าองค์ความรู้ด้านการประมวลผลเสียง (Speech Processing) การเรียนรู้เชิงลึก (Deep Learning) และการวิเคราะห์ข้อมูลหลายภาษา

&emsp;ขอขอบคุณ Dataset ที่เปิดให้ใช้งานโดยไม่มีค่าใช้จ่าย ได้แก่ RAVDESS Dataset โดย Livingstone & Russo (2018) และ Korean Voice Emotion Dataset ที่เผยแพร่ผ่าน Hugging Face Datasets รวมถึงชุมชน Open Source ที่พัฒนา Library ต่างๆ ที่ใช้ในโครงงานนี้ ได้แก่ TensorFlow, Keras, Librosa, scikit-learn, NumPy, Matplotlib และ Seaborn

&emsp;แม้โครงงานนี้จะไม่สามารถบรรลุเป้าหมายด้านความแม่นยำในระดับที่ตั้งไว้ได้ แต่กระบวนการวิจัยได้ให้ความเข้าใจอย่างลึกซึ้งเกี่ยวกับปัญหา Prosody Mismatch ซึ่งเป็นองค์ความรู้ที่มีคุณค่าสำหรับการพัฒนาต่อยอดในอนาคต

---

# บทที่ 1 บทนำ

## 1.1 ที่มาและความสำคัญ

&emsp;การรู้จำอารมณ์จากเสียงพูด (Speech Emotion Recognition: SER) เป็นสาขาวิจัยที่ได้รับความสนใจอย่างมากในทศวรรษที่ผ่านมา ระบบ SER มีการนำไปประยุกต์ใช้ในหลายด้าน ทั้งในระบบ Call Center อัตโนมัติที่ตรวจจับความไม่พอใจของลูกค้า, ระบบดูแลสุขภาพจิตที่ตรวจจับภาวะซึมเศร้า, อุปกรณ์ IoT อัจฉริยะที่ตอบสนองต่อสภาวะอารมณ์ของผู้ใช้งาน ตลอดจนระบบ In-car Emotion Detection ในรถยนต์สมัยใหม่

&emsp;ในโลกปัจจุบันที่มีการสื่อสารข้ามภาษาและข้ามวัฒนธรรมมากขึ้น การพัฒนาโมเดล SER ที่รองรับหลายภาษาในระบบเดียว **(Multilingual SER)** จึงเป็นที่ต้องการอย่างยิ่ง เพราะจะช่วยลดต้นทุนในการพัฒนาและบำรุงรักษาระบบ และทำให้ระบบสามารถใช้งานได้กับผู้ใช้ที่หลากหลายภาษาโดยไม่ต้องสร้างโมเดลแยกกันสำหรับแต่ละภาษา

&emsp;**แนวคิดเริ่มต้นของโครงงานนี้** คือการทดลองแนวทาง Multilingual Unified Model โดยนำข้อมูลเสียงพูดจากภาษาอังกฤษและภาษาเกาหลีมารวมกัน แล้วฝึกสอนโมเดล CNN + Bidirectional LSTM เพียงตัวเดียวเพื่อจำแนกอารมณ์ โดยมีสมมติฐานว่าโมเดลจะสามารถเรียนรู้ "รูปแบบอารมณ์ที่เป็นสากล" (Language-Independent Emotion Features) ออกมาได้จากข้อมูลที่หลากหลาย

&emsp;อย่างไรก็ตาม ในระหว่างการพัฒนาได้ค้นพบว่าสมมติฐานดังกล่าวมีข้อจำกัดสำคัญ เนื่องจากเสียงพูดในแต่ละภาษามีโครงสร้างทาง Prosody (น้ำเสียง จังหวะ ระดับเสียง) ที่แตกต่างกันอย่างมีนัยสำคัญ ซึ่งเป็นปัญหาที่ทำให้โครงงานไม่บรรลุเป้าหมายเดิม และนำไปสู่การวิเคราะห์และเสนอแนวทางแก้ไขในอนาคต

## 1.2 วัตถุประสงค์

1. พัฒนาระบบ SER แบบ Multilingual ด้วยสถาปัตยกรรม CNN + Bidirectional LSTM
2. ศึกษาและวิเคราะห์ปัญหา Prosody Mismatch ที่เกิดขึ้นจากการรวมข้อมูลหลายภาษา
3. ประเมินประสิทธิภาพของโมเดลด้วยตัวชี้วัด Accuracy, Confusion Matrix และ Classification Report
4. เสนอแนวทาง Future of Work สำหรับการพัฒนาต่อยอดในอนาคตโดยใช้แนวทาง Per-Language Model

## 1.3 ขอบเขตของโครงงาน

**ภาษาที่ใช้ทดสอบ:**
- ภาษาอังกฤษ — RAVDESS Dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)
- ภาษาเกาหลี — Korean Voice Emotion Dataset (โหลดผ่าน Hugging Face Datasets)

**ประเภทอารมณ์ที่จำแนก:** 5 ประเภท
| ลำดับ | อารมณ์ภาษาอังกฤษ | อารมณ์ภาษาไทย |
|---|---|---|
| 1 | Angry | โกรธ |
| 2 | Happy | มีความสุข |
| 3 | Sad | เศร้า |
| 4 | Neutral | เป็นกลาง |
| 5 | Surprise | ประหลาดใจ |

**Features ที่ใช้:**
- MFCC 40 coefficients (โมเดลทั่วไป — เร็ว ประหยัด VRAM)
- MFCC 128 + Mel Spectrogram (โมเดล High-Resolution — ความแม่นยำสูงกว่า)

**Hardware & Framework:**
- GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
- Framework: TensorFlow 2.x / Keras
- Audio Processing: Librosa
- Sample Rate: 22,050 Hz
- Duration: 3 วินาทีต่อไฟล์

**ขอบเขตที่ไม่ครอบคลุม:**
- ไม่รวมการรู้จำเสียงพูดแบบ Real-time Streaming
- ไม่รวมการประมวลผลภาษาธรรมชาติ (NLP) เพื่อวิเคราะห์เนื้อหา
- ไม่รวมภาษาไทย (เนื่องจากขาด Labeled Dataset)

## 1.4 ขั้นตอนการดำเนินงานและแผนการดำเนินงาน

```
ขั้นตอนที่ 1: ศึกษาทฤษฎี (สัปดาห์ที่ 1-2)
├── ทฤษฎี MFCC และ Mel Spectrogram
├── สถาปัตยกรรม CNN + Bidirectional LSTM
└── งานวิจัยที่เกี่ยวข้องด้าน SER

ขั้นตอนที่ 2: เตรียมข้อมูล (สัปดาห์ที่ 3-4)
├── ดาวน์โหลด RAVDESS Dataset
├── ดาวน์โหลด Korean Dataset ผ่าน Hugging Face
└── ตรวจสอบคุณภาพและจัดโครงสร้าง Dataset

ขั้นตอนที่ 3: พัฒนา Pipeline (สัปดาห์ที่ 5-6)
├── สร้างระบบ Feature Extraction (MFCC)
├── สร้างระบบ Data Augmentation
└── สร้างระบบป้องกัน Data Leakage

ขั้นตอนที่ 4: Train และทดสอบโมเดล (สัปดาห์ที่ 7-10)
├── Train_Universal_Super.py (โมเดลหลัก)
├── Train_Test.py (Anti-Leakage Strict Mode)
├── Train_Model_RTX3060.py (High-Resolution)
└── Train_model_res.py (Low VRAM)

ขั้นตอนที่ 5: ประเมินผลและวิเคราะห์ (สัปดาห์ที่ 11-12)
├── Evaluate_Fix_Final.py (Batch Evaluation)
├── Test_Final.py (Interactive Testing)
└── วิเคราะห์ Confusion Matrix และ Prosody Mismatch

ขั้นตอนที่ 6: สรุปผลและเขียนรายงาน (สัปดาห์ที่ 13-15)
├── สรุปข้อจำกัดที่ค้นพบ
├── เสนอ Future of Work (Per-Language Model)
└── จัดทำรายงานฉบับสมบูรณ์
```

## 1.5 ประโยชน์ที่คาดว่าจะได้รับ

1. **องค์ความรู้ด้าน Prosody Mismatch** — เข้าใจสาเหตุและกลไกที่ทำให้โมเดล Multilingual ไม่สามารถทำงานได้อย่างมีประสิทธิภาพ ซึ่งเป็นประโยชน์ต่องานวิจัยในอนาคต
2. **Pipeline ที่พร้อมใช้งาน** — ระบบ Feature Extraction, Data Augmentation และ Anti-Data-Leakage ที่พัฒนาขึ้นสามารถนำไปต่อยอดได้ทันที
3. **แนวทาง Future of Work** — เสนอกรอบการพัฒนา Per-Language Model ที่มีแนวโน้มจะแก้ปัญหา Prosody Mismatch ได้
4. **ประสบการณ์การทดลองจริง** — การทดสอบ Configuration หลายรูปแบบบน GPU จริงให้ข้อมูลเชิงปฏิบัติที่มีคุณค่า

## 1.6 นิยามคำศัพท์

| คำศัพท์ | คำนิยาม |
|---|---|
| **SER (Speech Emotion Recognition)** | การรู้จำอารมณ์จากเสียงพูดโดยใช้คอมพิวเตอร์ |
| **MFCC (Mel-Frequency Cepstral Coefficients)** | คุณลักษณะมาตรฐานสำหรับงาน Speech Processing สกัดจากสเปกตรัมของเสียง |
| **Prosody** | คุณสมบัติเหนือระดับเสียง ได้แก่ Pitch, Duration, Energy, Rhythm |
| **Prosody Mismatch** | ความแตกต่างของรูปแบบ Prosody ระหว่างภาษาที่ทำให้โมเดลสับสน |
| **CNN (Convolutional Neural Network)** | โครงข่ายประสาทเทียมที่ใช้ Convolution สกัด Local Pattern |
| **Bidirectional LSTM** | LSTM ที่อ่านลำดับข้อมูลทั้งสองทิศทาง (ซ้าย→ขวา และ ขวา→ซ้าย) |
| **Data Leakage** | การที่ข้อมูล Test ปนเข้าสู่กระบวนการ Train ทำให้ผลประเมินเกินจริง |
| **Data Augmentation** | การสร้างข้อมูลเพิ่มเทียมจากข้อมูลต้นฉบับเพื่อเพิ่มความหลากหลาย |
| **Overfitting** | โมเดลจำข้อมูล Train มากเกินไปจนทำงานกับข้อมูลใหม่ได้ไม่ดี |
| **Per-Language Model** | แนวทางฝึกสอนโมเดลแยกสำหรับแต่ละภาษา |

---

# บทที่ 2 การทบทวนวรรณกรรมและเทคโนโลยีที่เกี่ยวข้อง

## 2.1 ความรู้เบื้องต้นเกี่ยวกับการรู้จำอารมณ์จากเสียง

### 2.1.1 ทฤษฎีอารมณ์พื้นฐาน (Basic Emotion Theory)

Ekman (1992) เสนอว่ามนุษย์มีอารมณ์พื้นฐาน 6 ประเภทที่เป็นสากลข้ามวัฒนธรรม ได้แก่ Angry, Happy, Sad, Fear, Disgust และ Surprise โดยอารมณ์เหล่านี้มีการแสดงออกทางสีหน้าที่คล้ายกันในทุกวัฒนธรรม อย่างไรก็ตาม งานวิจัยในเวลาต่อมาพบว่าการแสดงออกทาง **เสียงพูด** นั้นมีความแตกต่างระหว่างวัฒนธรรมมากกว่าสีหน้า โดยเฉพาะในแง่ของ Prosody

&emsp;โครงงานนี้เลือกใช้อารมณ์ 5 ประเภทที่มีข้อมูล Dataset รองรับ ได้แก่ Angry, Happy, Sad, Neutral และ Surprise

### 2.1.2 ประวัติและพัฒนาการของ SER

| ยุค | แนวทาง | ข้อดี | ข้อจำกัด |
|---|---|---|---|
| ทศวรรษ 1990-2000 | Hand-crafted Features + HMM/SVM | ง่าย, ตีความได้ | ต้องเลือก Feature เอง, Accuracy ต่ำ |
| ทศวรรษ 2010-2015 | Deep Neural Networks | Accuracy สูงขึ้น | ต้องการข้อมูลมาก |
| ทศวรรษ 2015-ปัจจุบัน | CNN + LSTM / Transformer | State-of-the-art | ใช้ทรัพยากรมาก, Multilingual ยาก |

### 2.1.3 ความท้าทายของ Multilingual SER

งานวิจัยของ Schuller et al. (2010) ระบุว่า Cross-corpus (ข้ามชุดข้อมูล) และ Cross-lingual (ข้ามภาษา) เป็นความท้าทายหลักสองประการของ SER โดย:
- **Cross-corpus:** Accuracy มักลดลง 15-30% เมื่อทดสอบบน Dataset ต่างจากที่ใช้ Train
- **Cross-lingual:** Accuracy มักลดลง 20-40% เมื่อทดสอบข้ามภาษา เนื่องจาก Prosody Mismatch

## 2.2 ทฤษฎีพื้นฐานด้านการประมวลผลเสียง

### 2.2.1 Mel-Frequency Cepstral Coefficients (MFCC)

MFCC เป็น Feature มาตรฐานที่ใช้ในงาน Speech Processing มาตั้งแต่ทศวรรษ 1980 (Davis & Mermelstein, 1980) ขั้นตอนการคำนวณประกอบด้วย:

**ขั้นตอนที่ 1 — Framing & Windowing:**
แบ่งสัญญาณเสียงเป็น Frame เล็กๆ ขนาดประมาณ 25 ms โดยแต่ละ Frame ทับซ้อนกัน 10 ms คูณด้วย Hamming Window เพื่อลด Spectral Leakage:

$$w(n) = 0.54 - 0.46\cos\left(\frac{2\pi n}{N-1}\right)$$

**ขั้นตอนที่ 2 — Fast Fourier Transform (FFT):**
แปลงสัญญาณจาก Time Domain สู่ Frequency Domain:

$$X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-j2\pi kn/N}$$

**ขั้นตอนที่ 3 — Mel Filter Bank:**
ผ่านชุด Triangular Filters ที่จัดเรียงบน Mel Scale ซึ่งเลียนแบบการรับรู้เสียงของหูมนุษย์:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**ขั้นตอนที่ 4 — Log + DCT:**
นำ Log มาลด Dynamic Range แล้วใช้ Discrete Cosine Transform (DCT) แปลงเป็น Cepstral Coefficients:

$$c(n) = \sum_{k=1}^{K} \log\left(S_k\right) \cdot \cos\left[n\left(k - \frac{1}{2}\right)\frac{\pi}{K}\right]$$

**โครงงานนี้ใช้:**
- `n_mfcc = 40` สำหรับโมเดลทั่วไป (เร็ว, ประหยัด VRAM)
- `n_mfcc = 128` สำหรับโมเดล High-Resolution

```python
# โค้ดสกัด MFCC ที่ใช้ในโครงงาน
mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)  # → shape: (40, T)
features = mfcc.T                                        # → shape: (T, 40)
```

### 2.2.2 Mel Spectrogram

Mel Spectrogram แสดงการกระจายพลังงานของเสียงในมิติเวลาและความถี่ (บน Mel Scale) ให้ข้อมูลเชิงพื้นที่ (Spatial/Temporal Information) ที่ MFCC เพียงอย่างเดียวไม่มี:

```python
# โมเดล High-Resolution รวม MFCC + Mel Spectrogram
mfcc    = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)      # (128, T)
mel     = librosa.feature.melspectrogram(y=data, sr=sr)
mel_db  = librosa.power_to_db(mel, ref=np.max)                   # (128, T)
result  = np.concatenate((mfcc, mel_db), axis=0).T               # (T, 256)
```

ผลรวม Feature มิติ **256** ต่อ Time Step ให้ข้อมูลครบถ้วนทั้ง Spectral Envelope (จาก MFCC) และ Energy Distribution (จาก Mel Spectrogram)

### 2.2.3 Feature Comparison

| Feature | มิติ | จุดเด่น | จุดด้อย |
|---|---|---|---|
| MFCC (40) | (T, 40) | เร็ว, เบา, มาตรฐาน | ข้อมูลน้อยกว่า |
| MFCC (128) | (T, 128) | ความละเอียดสูง | ต้องการ VRAM มากขึ้น |
| MFCC 128 + Mel | (T, 256) | ข้อมูลครบถ้วนที่สุด | หนักที่สุด |

## 2.3 ลักษณะทาง Prosody และความแตกต่างระหว่างภาษา

### 2.3.1 Prosody คืออะไร

**Prosody** คือคุณสมบัติเหนือระดับเสียง (Suprasegmental Phonological Features) ของภาษา ซึ่งส่งผลต่อความหมายและอารมณ์ที่สื่อออกมาในการพูด

**ตารางที่ 2.1 คุณสมบัติ Prosody และความแตกต่างระหว่างภาษา**

| คุณสมบัติ | คำอธิบาย | ภาษาอังกฤษ | ภาษาเกาหลี |
|---|---|---|---|
| **Pitch (F0)** | ความถี่พื้นฐานของเสียง | ช่วง Pitch กว้าง (100-400 Hz) | ช่วง Pitch แคบกว่า (100-280 Hz) |
| **Duration** | ความยาวของพยางค์ | Stress-timed Rhythm — พยางค์ที่เน้นยาวกว่ามาก | Syllable-timed — ความยาวพยางค์สม่ำเสมอกว่า |
| **Energy** | ความดังของเสียง | ระดับ Energy สูงขึ้นฉับพลันเมื่อแสดงอารมณ์รุนแรง | ระดับ Energy เพิ่มขึ้นทีละน้อยกว่า |
| **Rhythm** | จังหวะการพูด | ไม่สม่ำเสมอ ขึ้นกับ Stress | สม่ำเสมอกว่า |
| **Intonation** | รูปแบบการขึ้นลงของ Pitch | ขึ้นลงชัดเจนมาก | ขึ้นลงน้อยกว่า แต่มีความหมายทาง Phonological |

### 2.3.2 ผลกระทบต่อ MFCC

MFCC สะท้อนโครงสร้างทาง Spectral ของเสียง ซึ่งได้รับอิทธิพลจากทั้ง **อารมณ์** และ **ภาษา** พร้อมกัน:

```
MFCC ที่โมเดลเห็น = ลักษณะอารมณ์  +  ลักษณะภาษา  +  ลักษณะผู้พูด
                      (สิ่งที่ต้องการ)   (Noise)         (Noise)
```

**ตัวอย่างเปรียบเทียบอารมณ์ "โกรธ" ระหว่างสองภาษา:**

```
อารมณ์ "โกรธ" (Angry):
┌────────────────┬──────────────────────┬──────────────────────┐
│ คุณสมบัติ      │ ภาษาอังกฤษ            │ ภาษาเกาหลี            │
├────────────────┼──────────────────────┼──────────────────────┤
│ Pitch Range    │ กว้างมาก (สูงฉับพลัน) │ แคบกว่า (ค่อยๆ สูงขึ้น) │
│ Speech Rate    │ เร็ว, ไม่สม่ำเสมอ     │ เร็ว แต่สม่ำเสมอกว่า   │
│ Energy Pattern │ พุ่งสูงฉับพลัน        │ เพิ่มขึ้นทีละน้อย       │
│ Vowel Duration │ ยาว (Stressed)        │ สั้นกว่า               │
│ MFCC Centroid  │ สูง (High Frequency)  │ ต่ำกว่า                │
└────────────────┴──────────────────────┴──────────────────────┘
```

### 2.3.3 ผลกระทบต่อโมเดล

เมื่อโมเดลเห็นข้อมูลรวมจากทั้งสองภาษา โมเดลจะประสบกับปัญหา:

1. **Confusion Between Language and Emotion:** โมเดลไม่สามารถแยกแยะได้ว่า Pitch ที่สูงขึ้นเป็นเพราะ "อารมณ์โกรธ" หรือเพราะ "ลักษณะของภาษาอังกฤษ"
2. **Overfitting Toward Majority Language:** ภาษาอังกฤษ (RAVDESS) มีข้อมูลมากกว่า โมเดลจึงเอนเอียงไปเรียนรู้รูปแบบของภาษาอังกฤษเป็นหลัก
3. **Poor Generalization:** เมื่อทดสอบด้วยเสียงภาษาเกาหลี โมเดลใช้รูปแบบที่เรียนรู้จากภาษาอังกฤษทำนาย ส่งผลให้ Accuracy ต่ำ

## 2.4 สถาปัตยกรรม Deep Learning ที่ใช้

### 2.4.1 Conv1D (1D Convolutional Layer)

Conv1D สกัด **Local Temporal Pattern** จาก Feature Sequence โดยใช้ Kernel เลื่อนผ่านมิติเวลา เหมาะกับข้อมูลที่เป็น Sequential เช่น MFCC ที่มีรูปแบบซ้ำๆ ตามเวลา

```
Input: (Batch, Time_Steps, Features)
Kernel ขนาด 5: สกัด Pattern จาก 5 Time Steps ที่ต่อเนื่องกัน
Output: (Batch, Time_Steps, Filters)
```

### 2.4.2 Bidirectional LSTM

LSTM (Long Short-Term Memory) แก้ปัญหา Vanishing Gradient ของ RNN ทั่วไปด้วย Gate Mechanism:
- **Forget Gate:** ตัดสินใจว่าจะลืมข้อมูลใดใน Cell State
- **Input Gate:** ตัดสินใจว่าจะเก็บข้อมูลใหม่อะไรลง Cell State
- **Output Gate:** ตัดสินใจว่าจะส่งออกอะไรเป็น Hidden State

**Bidirectional LSTM** อ่านลำดับข้อมูลทั้งสองทิศทาง:

$$h_t = \left[\overrightarrow{\text{LSTM}}(x_t) \; ; \; \overleftarrow{\text{LSTM}}(x_t)\right]$$

ทำให้เข้าใจบริบทของเสียงพูดทั้งจากอดีตและอนาคตพร้อมกัน

### 2.4.3 Regularization Techniques

| เทคนิค | การทำงาน | ค่าที่ใช้ |
|---|---|---|
| **Dropout** | สุ่มปิด Neurons ระหว่าง Training | 0.3 (30%) |
| **L2 Regularization** | จำกัดขนาด Weight ด้วย Penalty Term | λ = 0.001 |
| **Batch Normalization** | Normalize Activation ของแต่ละ Batch | — |
| **EarlyStopping** | หยุด Training เมื่อ val_loss ไม่ดีขึ้น | patience = 10 |
| **ReduceLROnPlateau** | ลด Learning Rate เมื่อ val_loss หยุดนิ่ง | factor = 0.5 |

## 2.5 ชุดข้อมูลที่เกี่ยวข้อง

### 2.5.1 RAVDESS (ภาษาอังกฤษ)

- **ชื่อเต็ม:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **ผู้จัดทำ:** Livingstone & Russo (2018)
- **ผู้แสดง:** 24 นักแสดงมืออาชีพ (12 ชาย, 12 หญิง)
- **สภาพแวดล้อม:** บันทึกในห้อง Anechoic Chamber ควบคุมเสียงสะท้อน
- **รูปแบบ:** ไฟล์ WAV, Mono, 48kHz
- **การ Label:** ผ่านระบบ Filename Code `XX-XX-[Emotion]-XX-XX-XX-XX.wav`

```python
# ตัวอย่าง Filename Mapping ที่ใช้ในโครงงาน
RAVDESS_MAP = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}
# ไฟล์ 03-01-05-01-01-01-12.wav → parts[2] = '05' → Angry
```

### 2.5.2 Korean Voice Emotion Dataset (ภาษาเกาหลี)

- **แหล่งที่มา:** Hugging Face Datasets (Streaming Mode)
- **รูปแบบ:** ข้อมูล Audio bytes แปลงเป็น WAV ด้วย soundfile
- **การจัดเก็บ:** จัดไว้ในโฟลเดอร์ตามอารมณ์ผ่าน [Label.py](SeniorP1/รวมภาษา Ai/Label.py)

### 2.5.3 โครงสร้างชุดข้อมูลรวม

```
dataset/
├── angry/      ← ไฟล์ WAV อารมณ์โกรธ (ภาษาอังกฤษ + เกาหลีรวมกัน)
├── happy/      ← ไฟล์ WAV อารมณ์มีความสุข
├── sad/        ← ไฟล์ WAV อารมณ์เศร้า
├── neutral/    ← ไฟล์ WAV อารมณ์เป็นกลาง
└── surprise/   ← ไฟล์ WAV อารมณ์ประหลาดใจ
```

**ปัญหาที่ตามมาจากโครงสร้างนี้:** การรวมไฟล์จากสองภาษาไว้ในโฟลเดอร์เดียวกันทำให้โมเดลไม่รู้ว่าแต่ละไฟล์มาจากภาษาอะไร จึงไม่สามารถปรับพฤติกรรมตามภาษาได้

## 2.6 สรุปผลการศึกษางานวิจัยที่เกี่ยวข้อง

จากการทบทวนวรรณกรรม สามารถสรุปได้ว่า:

1. **Multilingual SER เป็นปัญหาที่ยังเปิดอยู่ (Open Problem)** — งานวิจัยส่วนใหญ่ยังคงใช้ Per-Language Model หรือ Fine-tuning แยกภาษา
2. **MFCC ไม่เพียงพอสำหรับ Cross-lingual SER** — ต้องใช้ Feature ที่ Language-Neutral มากกว่า เช่น Wav2Vec 2.0 หรือ HuBERT
3. **Data Imbalance เป็นปัญหาสำคัญ** — ภาษาอังกฤษมี Open Dataset มากกว่าภาษาเอเชียมาก

---

# บทที่ 3 วิธีการดำเนินการวิจัย

## 3.1 ภาพรวมสถาปัตยกรรมระบบ

ระบบ Multilingual SER แบ่งการทำงานออกเป็น 2 Pipeline หลัก คือ **Training Pipeline** (กระบวนการฝึกสอนโมเดล) และ **Inference Pipeline** (กระบวนการทำนายอารมณ์จากไฟล์เสียงใหม่) ทั้งสอง Pipeline ใช้ขั้นตอน Preprocessing และ Feature Extraction ที่เหมือนกันทุกประการเพื่อให้ผลลัพธ์สอดคล้องกัน

### Training Pipeline

| ลำดับ | ขั้นตอน | รายละเอียด | เครื่องมือที่ใช้ |
|---|---|---|---|
| 1 | **File Discovery** | สแกนโฟลเดอร์ Dataset แบบ Recursive เพื่อรวบรวมไฟล์เสียงทุกชนิด | `os.walk()` |
| 2 | **Label Detection** | ตรวจจับอารมณ์จากชื่อโฟลเดอร์ (Korean) หรือชื่อไฟล์ (RAVDESS) | Path/Filename Parsing |
| 3 | **Audio Load** | โหลดไฟล์เสียงและแปลงเป็น NumPy Array ที่ Sample Rate มาตรฐาน | `librosa.load()` |
| 4 | **Silence Removal** | ตัดความเงียบหัวท้ายออกเพื่อลด Noise ที่ไม่มีข้อมูล | `librosa.effects.trim()` |
| 5 | **Pad / Cut** | ปรับความยาวสัญญาณให้คงที่ 3 วินาที (66,150 samples) | `np.pad()` |
| 6 | **Feature Extraction** | สกัด MFCC จากสัญญาณเสียง ได้ Matrix ขนาด (T, 40) | `librosa.feature.mfcc()` |
| 7 | **Data Augmentation** | สร้างตัวอย่างเพิ่มจาก Train Set เท่านั้น (Noise / Pitch / Time) | `librosa` effects |
| 8 | **Train/Val/Test Split** | แบ่งข้อมูลก่อนทุกกระบวนการเพื่อป้องกัน Data Leakage | `train_test_split()` |
| 9 | **StandardScaler** | Normalize Feature ให้มีค่าเฉลี่ย 0 และ Std 1 (Fit บน Train เท่านั้น) | `StandardScaler` |
| 10 | **Model Training** | ฝึกสอน CNN + Bi-LSTM พร้อม EarlyStopping และ ReduceLROnPlateau | `model.fit()` |
| 11 | **Save Best Model** | บันทึกโมเดลที่ดีที่สุด (val_accuracy สูงสุด) พร้อม Scaler | `.keras` + `.pkl` |

### Inference Pipeline

| ลำดับ | ขั้นตอน | รายละเอียด | หมายเหตุ |
|---|---|---|---|
| 1 | **Audio Input** | รับไฟล์เสียงรูปแบบ WAV / MP3 / FLAC | ไฟล์ใหม่ที่ไม่เคยเห็นระหว่าง Train |
| 2 | **Preprocessing** | Trim → Pad/Cut ให้ยาว 3 วินาที เหมือน Training | ต้องใช้ parameter เดิมทุกอย่าง |
| 3 | **Feature Extraction** | สกัด MFCC 40 coefficients | ต้องใช้ `n_mfcc=40` เหมือนตอน Train |
| 4 | **Normalization** | Transform ด้วย Scaler ที่บันทึกไว้จาก Train | ห้าม Fit ใหม่ — ต้อง Load จาก `.pkl` |
| 5 | **Model Prediction** | ส่ง Feature เข้าโมเดล CNN + Bi-LSTM | โมเดลคำนวณความน่าจะเป็นทั้ง 5 อารมณ์ |
| 6 | **Softmax Output** | ได้ผลลัพธ์เป็น Probability ของแต่ละอารมณ์ รวมกัน = 1.0 | เช่น Angry=0.72, Happy=0.12, ... |

## 3.2 การออกแบบ Data Pipeline

### 3.2.1 Label Detection (2 วิธี)

&emsp;ระบบจำเป็นต้องอ่าน Label (ประเภทอารมณ์) ของไฟล์เสียงแต่ละไฟล์โดยอัตโนมัติ เนื่องจาก Dataset ที่ใช้มีสองแหล่งที่มีรูปแบบการจัดเก็บต่างกันโดยสิ้นเชิง จึงออกแบบระบบตรวจจับ Label ไว้ 2 วิธี ให้ทำงานตามลำดับความสำคัญ

**วิธีที่ 1 — Path-based Detection (สำหรับ Korean Dataset):**

&emsp;Korean Dataset จัดเก็บไฟล์เสียงโดยแยกโฟลเดอร์ตามอารมณ์ เช่น `dataset/angry/file.wav` ดังนั้นระบบจะตรวจสอบว่า path ของไฟล์มีชื่อโฟลเดอร์ที่ตรงกับคำสำคัญของอารมณ์หรือไม่ โดยแปลง path เป็นตัวพิมพ์เล็กและเปลี่ยน backslash เป็น forward slash ก่อน เพื่อให้การเปรียบเทียบทำงานได้ถูกต้องบนทุก OS จากนั้นวนลูปตรวจสอบกับ Dictionary `EMOTION_KEYWORDS` ทีละคำ ถ้าพบคำใดอยู่ในรูปแบบ `/keyword/` ในสาย path ก็กำหนด Label ทันทีและหยุดการค้นหา

```python
# แปลง path ให้เป็นมาตรฐาน (lowercase + forward slash)
path_check = file_path.lower().replace('\\', '/')

# EMOTION_KEYWORDS = {'angry': 'angry', 'happy': 'happy', 'sad': 'sad', ...}
for key, emotion_name in EMOTION_KEYWORDS.items():
    # ตรวจสอบว่า path มีโฟลเดอร์ชื่อ /angry/ หรือ /happy/ เป็นต้น
    if f"/{key}/" in path_check:
        label = emotion_name
        break  # พบแล้ว หยุดวนลูป
```

**วิธีที่ 2 — Filename-based Detection (สำหรับ RAVDESS):**

&emsp;RAVDESS Dataset ใช้รูปแบบชื่อไฟล์แบบ Structured Code เช่น `03-01-05-01-01-01-12.wav` โดยแต่ละตำแหน่งที่คั่นด้วยขีด (-) มีความหมายเฉพาะ ตำแหน่งที่ 3 (index 2 เมื่อนับจาก 0) คือ Emotion Code ได้แก่ `01` = neutral, `03` = happy, `04` = sad, `05` = angry ระบบจะแยกชื่อไฟล์ด้วย `.split('-')` แล้วตรวจสอบว่า `parts[2]` อยู่ใน RAVDESS_MAP หรือไม่ ถ้าใช่จะดึง Label ออกมาจาก Dictionary

```python
# RAVDESS_MAP จับคู่ Emotion Code กับชื่ออารมณ์
RAVDESS_MAP = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry'}

filename = os.path.basename(path_check)   # ดึงเฉพาะชื่อไฟล์
parts = filename.split('-')               # ['03','01','05','01','01','01','12.wav']

# ตรวจสอบว่าไฟล์มีรูปแบบ RAVDESS และ Emotion Code อยู่ใน Map
if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
    label = RAVDESS_MAP[parts[2]]         # parts[2]='05' → label='angry'
```

&emsp;ระบบจะลองวิธีที่ 1 ก่อน หากไม่พบจึงลองวิธีที่ 2 และหากทั้งสองวิธีไม่สามารถระบุ Label ได้ ไฟล์นั้นจะถูกข้ามไปโดยไม่นำเข้า Dataset เพื่อป้องกันข้อมูลที่ไม่มี Label ปะปนกับข้อมูลที่ถูกต้อง

---

### 3.2.2 Preprocessing Pipeline

&emsp;ก่อนที่จะสกัด Feature จากไฟล์เสียง จำเป็นต้องผ่านขั้นตอน Preprocessing เพื่อทำให้ข้อมูลทุกไฟล์อยู่ในรูปแบบมาตรฐานเดียวกัน เนื่องจากไฟล์เสียงใน Dataset มีความยาวต่างกัน มี Sample Rate ต่างกัน และบางไฟล์มีความเงียบที่ต้นและปลายเสียง ซึ่งจะทำให้ Feature ที่สกัดออกมามีขนาดไม่เท่ากันและมี Noise ที่ไม่จำเป็น ขั้นตอน Preprocessing มีทั้งหมด 3 ขั้น ดังนี้

**ขั้นที่ 1 — โหลดไฟล์เสียง และแปลง Sample Rate:**

&emsp;ใช้ `librosa.load()` โหลดไฟล์เสียงและแปลงเป็น NumPy Array โดยบังคับ Sample Rate ให้เป็น 22,050 Hz เสมอ ไม่ว่าไฟล์ต้นฉบับจะบันทึกที่ความถี่ใดก็ตาม นอกจากนี้ยังกำหนด `duration=3` เพื่อตัดข้อมูลให้ไม่เกิน 3 วินาทีตั้งแต่ขั้นตอนการโหลด ซึ่งช่วยประหยัดหน่วยความจำสำหรับไฟล์ที่ยาวมาก

```python
# sr=22050 → บังคับ Resample เป็น 22,050 Hz ทุกไฟล์
# duration=3 → ตัดให้ไม่เกิน 3 วินาทีตั้งแต่ต้น (ประหยัด RAM)
data, sr = librosa.load(file_path, sr=22050, duration=3)
# data = NumPy Array รูปร่าง (N_samples,) เช่น (66150,) สำหรับ 3 วินาที
# sr   = 22050 เสมอ (ค่าที่เราบังคับ)
```

**ขั้นที่ 2 — ตัดความเงียบ (Silence Trimming):**

&emsp;หลังจากโหลดแล้ว ไฟล์เสียงอาจมีความเงียบที่หัวและท้าย (เช่น ช่วงที่นักแสดงหายใจก่อนพูด หรือช่วงหลังจากพูดจบ) ความเงียบเหล่านี้ไม่มีข้อมูล Emotion ใดๆ แต่จะทำให้ MFCC ของแต่ละไฟล์มีรูปแบบต่างกันโดยไม่จำเป็น `librosa.effects.trim()` จะตัดส่วนที่มีพลังงานต่ำกว่า `top_db=25` ออก (25 dB ต่ำกว่า Peak ของสัญญาณ) แล้วคืนเฉพาะส่วนที่มีเสียงจริงๆ

```python
# top_db=25 หมายความว่า ตัดส่วนที่เงียบกว่า Peak 25 dB ออก
# _ คือ index ของส่วนที่เหลือ (ไม่ได้ใช้)
data, _ = librosa.effects.trim(data, top_db=25)
# ผลลัพธ์: data มีความยาวสั้นลง เหลือเฉพาะส่วนที่มีเสียงพูดจริง
```

**ขั้นที่ 3 — ปรับความยาวให้เท่ากันทุกไฟล์ (Pad / Cut):**

&emsp;โมเดล CNN + Bi-LSTM ต้องการ Input ที่มีขนาดคงที่ทุก Batch ดังนั้นหลังจากตัดความเงียบแล้ว ต้องปรับความยาวสัญญาณให้เท่ากับ 3 วินาทีพอดี (66,150 samples = 22,050 Hz × 3 วินาที) โดยถ้าสัญญาณสั้นกว่า 66,150 samples จะเติม 0 ต่อท้าย (Zero Padding) และถ้ายาวกว่าจะตัดเอาเฉพาะ 66,150 samples แรก

```python
SAMPLES_PER_TRACK = 22050 * 3  # = 66,150 samples

if len(data) < SAMPLES_PER_TRACK:
    # สัญญาณสั้นกว่า 3 วินาที → เติม 0 ต่อท้าย (Zero Padding)
    # (0, SAMPLES_PER_TRACK - len(data)) = เติมด้านขวาเท่านั้น
    data = np.pad(data, (0, SAMPLES_PER_TRACK - len(data)), 'constant')
else:
    # สัญญาณยาวกว่า 3 วินาที → ตัดเอาแค่ 3 วินาทีแรก
    data = data[:SAMPLES_PER_TRACK]
# ผลลัพธ์: data.shape = (66150,) เสมอ ไม่ว่าต้นฉบับจะยาวแค่ไหน
```

---

### 3.2.3 การป้องกัน Data Leakage (Critical Process)

**Data Leakage คืออะไร และทำไมถึงอันตราย**

&emsp;Data Leakage คือสถานการณ์ที่ข้อมูลจากชุด Test Set "รั่วไหล" เข้าไปมีอิทธิพลต่อกระบวนการสร้างโมเดล ทำให้โมเดลได้ "ดู" ข้อมูลที่ควรจะเป็นข้อมูลทดสอบไปแล้วบางส่วนก่อนที่จะถูกทดสอบจริง ผลที่ตามมาคือ Accuracy ที่วัดได้สูงกว่าความเป็นจริง ทำให้ประเมินประสิทธิภาพของโมเดลผิดพลาดอย่างมีนัยสำคัญ และเมื่อนำโมเดลไปใช้งานจริงกับข้อมูลที่ไม่เคยเห็นมาก่อน ประสิทธิภาพจะต่ำกว่าที่รายงานไว้มาก

**แหล่งที่มาของ Data Leakage ในโครงงานนี้**

&emsp;ใน Pipeline ของงาน SER มีโอกาสเกิด Data Leakage ได้ 2 จุดหลักคือ

&emsp;**จุดที่ 1 — Data Augmentation ก่อน Split:** หากนำข้อมูลทั้งหมดมา Augment ก่อน แล้วค่อย Split โดยสมมติว่ามีไฟล์ต้นฉบับ A อยู่ในชุดข้อมูล เมื่อ Augment จะได้ไฟล์ A, A_noise, A_pitch, A_stretch จากนั้นเมื่อ Split แบบสุ่ม อาจเกิดกรณีที่ A อยู่ใน Test Set แต่ A_noise หรือ A_pitch ซึ่งสร้างมาจากไฟล์เดียวกันกลับอยู่ใน Train Set โมเดลจึงได้ฝึกกับข้อมูลที่แทบเหมือนกันกับข้อมูล Test ทำให้ผลประเมินสูงเกินจริง

&emsp;**จุดที่ 2 — StandardScaler Fit บนข้อมูลรวม:** หากนำข้อมูล Train + Test + Val ทั้งหมดมา Fit Scaler พร้อมกัน ค่า Mean และ Standard Deviation ที่คำนวณได้จะมีข้อมูลของ Test ปนอยู่ด้วย โมเดลจึงทราบสถิติของข้อมูล Test ล่วงหน้าโดยอ้อม ซึ่งถือเป็น Leakage เช่นกัน

**วิธีที่ผิด (เกิด Data Leakage):**

| ลำดับ | ขั้นตอน | ปัญหาที่เกิด |
|---|---|---|
| 1 | โหลดไฟล์เสียงทั้งหมดพร้อมกัน | — |
| 2 | Augment ข้อมูลทั้งหมด (Train + Test รวมกัน) | ไฟล์ Augment ของ Train ปะปนกับ Test |
| 3 | Fit Scaler บนข้อมูลรวมทุกชุด | Scaler "เรียนรู้" สถิติของ Test ล่วงหน้า |
| 4 | แบ่ง Train / Val / Test | แบ่งช้าเกินไป — Leakage เกิดขึ้นแล้ว |
| 5 | Train โมเดล → วัด Accuracy | ผล Accuracy สูงเกินจริง ไม่สะท้อนประสิทธิภาพจริง |

**วิธีที่ถูก (ป้องกัน Data Leakage ทุกจุด):**

| ลำดับ | ขั้นตอน | เหตุผลที่สำคัญ |
|---|---|---|
| 1 | รวบรวม **File Paths** (ยังไม่โหลดข้อมูล) | ทำงานกับ Path เท่านั้น ไม่ยุ่งกับเนื้อหาไฟล์ |
| 2 | **Split File Paths** → train_files / val_files / test_files | แบ่งก่อน ตั้งแต่ระดับ Path เพื่อรับประกันว่าไฟล์เดียวกันจะไม่อยู่คนละ Set |
| 3 | **Augment เฉพาะ train_files** (สร้าง 3 ตัวอย่างต่อไฟล์) | val/test ไม่ถูก Augment ไม่มีทางที่ Augmented version จะปนเข้า Test |
| 4 | โหลด val_files และ test_files แบบ **Original** (ไม่ Augment) | Test ใช้ข้อมูลจริงตามที่เป็น ไม่ผ่านการดัดแปลงใดๆ |
| 5 | **Fit StandardScaler บน X_train เท่านั้น** | Scaler เรียนรู้เฉพาะสถิติของ Train ไม่รู้จัก Val/Test เลย |
| 6 | **Transform** X_val และ X_test ด้วย Scaler ที่ Fit ไว้แล้ว | ใช้ค่า Mean/Std จาก Train มา Normalize Val/Test เท่านั้น |
| 7 | **บันทึก Scaler** ไว้ในไฟล์ `.pkl` | ตอน Inference ต้อง Load Scaler เดิมมาใช้ ห้ามสร้าง Scaler ใหม่ |

**ผลกระทบของการป้องกัน Data Leakage:**

&emsp;เมื่อดำเนินการอย่างถูกต้องตามขั้นตอนข้างต้น ผล Accuracy ที่วัดได้บน Test Set จะสะท้อนประสิทธิภาพที่แท้จริงของโมเดลเมื่อนำไปใช้กับข้อมูลใหม่ที่ไม่เคยเห็นมาก่อน แม้ตัวเลข Accuracy อาจดูต่ำกว่าระบบที่มี Leakage แต่ถือว่าเชื่อถือได้และนำไปเปรียบเทียบกับงานวิจัยอื่นได้อย่างยุติธรรม ในโครงงานนี้ค่า Test Accuracy ที่วัดได้ประมาณ 68% จึงเป็นค่าที่เชื่อถือได้จริง ไม่ใช่ค่าที่ถูกเพิ่มขึ้นจาก Leakage

---

### 3.2.4 Data Augmentation

&emsp;Data Augmentation คือกระบวนการสร้างตัวอย่างข้อมูลเพิ่มเติมจากข้อมูลที่มีอยู่ โดยดัดแปลงในรูปแบบที่ยังคงความหมาย (Label) ไว้เหมือนเดิม เพื่อเพิ่มความหลากหลายของข้อมูล Train และช่วยให้โมเดล Generalize ได้ดีขึ้น

**ตารางที่ 3.3 เทคนิค Data Augmentation ที่ใช้**

| เทคนิค | วิธีการ | สูตร | จุดประสงค์ |
|---|---|---|---|
| **Gaussian Noise** | เพิ่ม Noise แบบสุ่มเข้าสัญญาณ | $x' = x + \alpha\mathcal{N}(0,1)$ | ทำให้โมเดลทนต่อ Background Noise ในสภาพแวดล้อมจริง |
| **Pitch Shifting** | ปรับระดับ Pitch ขึ้น/ลง ±0.7 Semitones | `librosa.effects.pitch_shift(n_steps=±0.7)` | ทำให้โมเดลทนต่อความแตกต่างของระดับเสียงระหว่างผู้พูดแต่ละคน |
| **Time Stretching** | ยืดหรือหดเวลาของสัญญาณ (rate=0.8) | `librosa.effects.time_stretch(rate=0.8)` | ทำให้โมเดลทนต่อความเร็วในการพูดที่แตกต่างกัน |

&emsp;ผลลัพธ์ของ Augmentation คือไฟล์เสียง 1 ไฟล์จะถูกแปลงเป็น 3 ตัวอย่าง ได้แก่ ตัวอย่างที่เพิ่ม Noise / ตัวอย่างที่ปรับ Pitch / ตัวอย่างที่ยืดเวลา ทำให้ขนาดของ Train Set เพิ่มขึ้นเป็น 3 เท่า โดยไม่ต้องเก็บข้อมูลเพิ่ม สำคัญที่สุดคือ Augmentation ทำเฉพาะกับ **Train Set เท่านั้น** ตาม Anti-Data-Leakage Policy ที่อธิบายใน 3.2.3

---

### 3.2.5 Normalization Strategy

&emsp;ก่อนส่งข้อมูลเข้าโมเดล ต้อง Normalize Feature MFCC ให้อยู่ในช่วงที่เหมาะสม เพราะ MFCC แต่ละ Coefficient มีช่วงค่าที่แตกต่างกันมาก เช่น Coefficient ที่ 1 อาจมีค่าในช่วง -200 ถึง +50 ในขณะที่ Coefficient ที่ 10 อาจมีค่าในช่วง -30 ถึง +30 หากไม่ Normalize โมเดลจะให้ความสำคัญกับ Coefficient ที่มีค่าสูงกว่ามากเกินไป ทำให้เรียนรู้ได้ช้าและ Converge ได้ยาก

&emsp;โครงงานนี้ใช้ **StandardScaler** ซึ่งแปลงข้อมูลให้มีค่าเฉลี่ย (Mean) = 0 และ ส่วนเบี่ยงเบนมาตรฐาน (Std) = 1 ตามสูตร:

$$z = \frac{x - \mu}{\sigma}$$

โดยที่ $\mu$ คือค่าเฉลี่ยของ Train Set และ $\sigma$ คือ Standard Deviation ของ Train Set

**กระบวนการ Normalization ที่ถูกต้อง:**

```python
scaler = StandardScaler()
N, T, F = X_train.shape
# N = จำนวนตัวอย่าง, T = Time Steps (130), F = Features (40)

# ขั้นที่ 1: Reshape จาก (N, T, F) เป็น (N, T×F) เพื่อให้ Scaler คำนวณได้
# จากนั้น fit_transform จะ:
#   - คำนวณ Mean และ Std ของแต่ละ Feature จาก Train เท่านั้น
#   - Normalize ข้อมูล Train ด้วยค่า Mean/Std ที่คำนวณได้
#   - Reshape กลับเป็น (N, T, F)
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)

# ขั้นที่ 2: Normalize Val ด้วยค่า Mean/Std จาก Train (ห้าม fit ใหม่!)
# scaler.transform ใช้ค่า Mean/Std ที่เรียนรู้จาก Train มา Apply กับ Val
X_val = scaler.transform(X_val.reshape(-1, T*F)).reshape(-1, T, F)

# ขั้นที่ 3: Normalize Test ด้วยค่า Mean/Std จาก Train เช่นกัน
X_test = scaler.transform(X_test.reshape(-1, T*F)).reshape(-1, T, F)

# ขั้นที่ 4: บันทึก Scaler ไว้ใช้ตอน Inference
# เมื่อต้องการทดสอบไฟล์เสียงใหม่ ต้อง Load Scaler นี้มา Transform ก่อนเสมอ
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)  # บันทึกด้วย Pickle เพื่อ Load กลับมาใช้ได้
```

&emsp;เหตุผลที่ต้อง Fit Scaler บน Train Set เท่านั้น เพราะหากนำ Val หรือ Test มา Fit ด้วย ค่า Mean และ Std จะถูกกำหนดจากข้อมูล Test ทำให้โมเดลทราบสถิติของข้อมูลทดสอบล่วงหน้า ซึ่งเป็น Data Leakage รูปแบบหนึ่ง นอกจากนี้การบันทึก Scaler ไว้ในไฟล์ `.pkl` มีความสำคัญอย่างยิ่ง เพราะเมื่อต้องการทำนายอารมณ์จากไฟล์เสียงใหม่ในอนาคต จะต้อง Normalize ด้วยค่า Mean/Std ชุดเดิมกันกับตอน Train เท่านั้น หาก Normalize ด้วยค่าใหม่ที่คำนวณจากไฟล์เสียงเพียงไฟล์เดียว ผลการทำนายจะผิดพลาดอย่างมาก

## 3.3 สถาปัตยกรรมโมเดล CNN + Bi-LSTM

&emsp;โมเดลที่ใช้ในโครงงานนี้เป็นสถาปัตยกรรมแบบ Hybrid ที่ผสมผสานระหว่าง Convolutional Neural Network (CNN) และ Bidirectional Long Short-Term Memory (Bi-LSTM) โดยมีทั้งหมด 5 กลุ่มชั้น (Layer Group) ทำงานต่อเนื่องกันเป็น Sequential Pipeline โดยมีจำนวน Parameter ที่สามารถเรียนรู้ได้ทั้งหมด **653,061 ตัว** รายละเอียดของแต่ละชั้นมีดังนี้

**กลุ่มที่ 1 — Input Layer**

&emsp;ข้อมูล Input ที่ส่งเข้าโมเดลมีรูปร่าง **(Batch Size, 130 Time Steps, 40 Features)** โดย 130 Time Steps คือจำนวน Frame ที่ได้จาก MFCC เมื่อแบ่งสัญญาณเสียง 3 วินาทีด้วย Hop Length มาตรฐาน และ 40 Features คือจำนวน MFCC Coefficients ที่สกัดออกมาจากแต่ละ Frame ซึ่งแสดงถึงลักษณะเฉพาะของสเปกตรัมเสียงในแต่ละช่วงเวลา

**กลุ่มที่ 2 — CNN Block 1 (สกัด Local Pattern ความละเอียดสูง)**

&emsp;ชั้นแรกเป็น Conv1D ที่มี 256 Filter ขนาด Kernel 5 และ Activation Function เป็น ReLU ทำหน้าที่สแกนลำดับเวลา 5 Time Steps ติดต่อกันพร้อมกัน 256 รูปแบบ เพื่อตรวจจับ Local Temporal Pattern เช่น การเปลี่ยนแปลงของ Pitch ในช่วงสั้นๆ หรือลักษณะ Onset ของพยัญชนะ ผลลัพธ์มีรูปร่าง (Batch, 130, 256) จากนั้นผ่าน BatchNormalization เพื่อปรับ Distribution ของ Activation ให้เสถียรและเร่งการ Converge ตามด้วย MaxPooling1D ขนาด Pool 2 ซึ่งย่อ Time Steps จาก 130 เป็น 65 และเลือกเฉพาะ Pattern ที่โดดเด่นที่สุดในแต่ละช่วง สุดท้ายผ่าน Dropout อัตรา 30% เพื่อสุ่มปิด Neuron ระหว่าง Training ป้องกัน Overfitting ผลลัพธ์ออกจาก Block นี้มีรูปร่าง **(Batch, 65, 256)**

**กลุ่มที่ 3 — CNN Block 2 (สกัด High-level Pattern)**

&emsp;ชั้นที่สองเป็น Conv1D ที่มี 128 Filter ขนาด Kernel 5 ทำงานต่อจาก Block แรก โดยรับ Feature Map ขนาด (65, 256) มาสกัด Pattern ระดับสูงขึ้นที่ครอบคลุมช่วงเวลายาวกว่า เพราะแต่ละ Time Step ใน Block นี้แทน 2 Time Step ของสัญญาณต้นฉบับแล้ว จำนวน Filter ลดลงจาก 256 เป็น 128 เพื่อบีบอัดข้อมูลและคงเฉพาะ Pattern ที่สำคัญ หลังผ่าน BatchNormalization, MaxPooling1D (Pool=2) และ Dropout 30% ผลลัพธ์มีรูปร่าง **(Batch, 32, 128)** ซึ่งเป็น Compressed Representation ที่มี Semantic สูง พร้อมส่งต่อให้ LSTM

**กลุ่มที่ 4 — Bidirectional LSTM Block 1 (จับ Sequential Dependency สองทิศทาง)**

&emsp;ชั้นที่สามเป็น Bidirectional LSTM ที่มี 128 Hidden Units ต่อทิศทาง (รวม 256 หน่วย) โดยรับ Sequence ขนาด (32, 128) จาก CNN แล้วอ่านข้อมูลพร้อมกัน 2 ทิศทาง ได้แก่ ทิศทางปกติจากซ้ายไปขวา (Forward LSTM) และทิศทางย้อนกลับจากขวาไปซ้าย (Backward LSTM) การอ่านสองทิศทางทำให้โมเดลเข้าใจบริบทของเสียงพูดได้ครบถ้วน เช่น สามารถรับรู้ว่าอารมณ์ที่ปรากฏช่วงกลางประโยคมีความสัมพันธ์กับ Intonation ที่อยู่ท้ายประโยคอย่างไร ตั้งค่า `return_sequences=True` เพื่อส่ง Hidden State ของทุก Time Step ต่อไปยัง LSTM ชั้นถัดไป หลัง Dropout 30% ผลลัพธ์มีรูปร่าง **(Batch, 32, 256)**

**กลุ่มที่ 5 — Bidirectional LSTM Block 2 (สรุป Context ทั้งหมด)**

&emsp;ชั้นที่สี่เป็น Bidirectional LSTM ที่มี 64 Hidden Units ต่อทิศทาง (รวม 128 หน่วย) ทำหน้าที่สรุป Sequential Context จาก 32 Time Steps ที่ผ่านมาทั้งหมดให้เป็น Vector เดียวขนาด 128 มิติ โดยไม่ได้ตั้งค่า `return_sequences` ดังนั้นจึงส่งออกเฉพาะ Hidden State ของ Time Step สุดท้ายเท่านั้น ซึ่งสรุป "สาระสำคัญ" ของทั้ง Sequence เสียงไว้ หลัง Dropout 30% ผลลัพธ์มีรูปร่าง **(Batch, 128)** ซึ่งเป็น Fixed-size Emotion Representation Vector

**กลุ่มที่ 6 — Dense Layers (จำแนกอารมณ์)**

&emsp;ชั้นสุดท้ายเป็น Fully Connected Layers 2 ชั้น ชั้นแรก Dense 64 หน่วยพร้อม ReLU Activation และ L2 Regularization (λ=0.001) ทำหน้าที่แปลง Emotion Representation Vector ขนาด 128 มิติให้เป็น Feature Space ขนาด 64 มิติที่เหมาะสมกับการจำแนกอารมณ์ L2 Regularization เพิ่มค่า Penalty เข้า Loss Function เพื่อป้องกันไม่ให้ Weight มีขนาดใหญ่เกินไป หลัง Dropout 30% จะผ่าน Dense 5 หน่วยพร้อม Softmax Activation ซึ่งแปลง Logits ให้เป็น Probability Distribution โดยผลรวมของ Probability ทั้ง 5 อารมณ์จะเท่ากับ 1.0 เสมอ

**ตารางที่ 3.4 สรุปสถาปัตยกรรมชั้นต่อชั้น**

| กลุ่ม | Layer | Output Shape | Parameters | บทบาทหน้าที่ |
|---|---|---|---|---|
| Input | — | (Batch, 130, 40) | 0 | รับ MFCC Feature Sequence |
| CNN Block 1 | Conv1D(256, k=5) + BN + Pool + Drop | (Batch, 65, 256) | 52,480 | สกัด Local Temporal Pattern |
| CNN Block 2 | Conv1D(128, k=5) + BN + Pool + Drop | (Batch, 32, 128) | 164,480 | สกัด High-level Pattern |
| BiLSTM 1 | Bidirectional LSTM(128, seq=True) + Drop | (Batch, 32, 256) | 263,168 | จับ Sequential Dependency 2 ทิศทาง |
| BiLSTM 2 | Bidirectional LSTM(64) + Drop | (Batch, 128) | 164,352 | สรุป Context ทั้ง Sequence |
| Dense | Dense(64, L2) + Drop + Dense(5, Softmax) | (Batch, 5) | 8,581 | จำแนกออกเป็น 5 อารมณ์ |
| **รวม** | | | **653,061** | |

**เหตุผลในการเลือกสถาปัตยกรรม Hybrid CNN + Bi-LSTM:**

&emsp;โครงสร้าง Hybrid นี้ถูกเลือกเพราะงานวิจัยของ Zhao et al. (2019) แสดงให้เห็นว่า CNN สามารถสกัด Local Feature จากสัญญาณ MFCC ได้อย่างมีประสิทธิภาพ ในขณะที่ LSTM จัดการ Sequential Dependency ระยะยาวได้ดี ซึ่งทั้งสองด้านมีความจำเป็นสำหรับงาน Speech Emotion Recognition เพราะอารมณ์ในเสียงพูดมีทั้งลักษณะ Local (การเปลี่ยนแปลงฉับพลัน) และลักษณะ Global (รูปแบบที่กระจายตลอดประโยค) การใช้ Bidirectional แทน Unidirectional LSTM ช่วยเพิ่ม Accuracy เนื่องจากการรับรู้ Context ของเสียงพูดต้องการข้อมูลจากทั้งอดีตและอนาคตในลำดับเสียงพร้อมกัน

## 3.4 การกำหนด Training Configuration

**ตารางที่ 3.2 Training Configuration และเหตุผลในการเลือกใช้**

| Parameter | ค่าที่ใช้ | เหตุผล |
|---|---|---|
| Sample Rate | 22,050 Hz | มาตรฐาน Librosa, ครอบคลุม Speech Frequency Range |
| Duration | 3 วินาที | ครอบคลุมอารมณ์ที่แสดงออก, ไม่หนักเกินไป |
| n_mfcc | 40 / 128 | 40 = เร็ว; 128 = ละเอียด |
| Optimizer | Adam (lr=0.001) | ปรับ Learning Rate อัตโนมัติ |
| Loss Function | Categorical Cross-Entropy | Multi-class Classification มาตรฐาน |
| Batch Size | 64 / 32 / 16 | ขึ้นกับ VRAM ที่มีอยู่ |
| Max Epochs | 100–150 | หยุดเองด้วย EarlyStopping |
| EarlyStopping | patience=10–12 | ป้องกัน Overfitting |
| ReduceLROnPlateau | factor=0.5, patience=4–5 | Fine-tune ใกล้ Optimum |
| Mixed Precision | float16 | เพิ่มความเร็วบน RTX 3060 (Tensor Cores) |

**ตารางที่ 3.1 โมเดลที่พัฒนาทั้งหมดในโครงงาน**

| ไฟล์ | Features | Batch | Augment | จุดเด่น |
|---|---|---|---|---|
| [Train_Universal_Super.py](SeniorP1/รวมภาษา Ai/Train_Universal_Super.py) | MFCC 40 | 64 | ✅ | เร็ว, ประหยัด RAM, โมเดลหลัก |
| [Train_Test.py](SeniorP1/รวมภาษา Ai/Train_Test.py) | MFCC 40 | 32 | ✅ | Anti-Data-Leakage เข้มงวดที่สุด |
| [Train_Model_RTX3060.py](SeniorP1/รวมภาษา Ai/Train_Model_RTX3060.py) | MFCC 128 + Mel | 32 | ✅ | High-Resolution, Mixed Precision |
| [Train_model_res.py](SeniorP1/รวมภาษา Ai/Train_model_res.py) | MFCC 128 + Mel | 16 | ✅ | High-Resolution สำหรับ VRAM ต่ำ |

---

# บทที่ 4 ระบบต้นแบบ

## 4.1 ภาพรวมการออกแบบ (Overview Design)

### 4.1.1 คำอธิบายการทำงานของแต่ละไฟล์ Python

&emsp;โครงงานนี้ประกอบด้วยไฟล์ Python ทั้งหมด 10 ไฟล์ แบ่งตามบทบาทหน้าที่ออกเป็น 4 กลุ่ม ดังนี้

---

**กลุ่มที่ 1 — เครื่องมือตรวจสอบระบบ (Diagnostic Tools)**

**1. Check Gpu.py**

&emsp;ไฟล์นี้มีหน้าที่ตรวจสอบว่าสภาพแวดล้อมการทำงานมี GPU พร้อมใช้งานหรือไม่ก่อนเริ่ม Training โดย Import `tensorflow` แล้วเรียก `tf.config.list_physical_devices('GPU')` เพื่อรับรายการ GPU ทั้งหมดที่ TensorFlow มองเห็น หากพบ GPU จะแสดงชื่อและจำนวน GPU พร้อมแจ้งว่าพร้อมรันแบบ Turbo (ใช้ GPU Acceleration) หากไม่พบ GPU จะแสดงข้อความเตือนและแนะนำให้ตรวจสอบการติดตั้ง CUDA/cuDNN โดยที่ไฟล์นี้ไม่ได้โหลด Dataset หรือสร้างโมเดลใดๆ ทำงานเสร็จภายในไม่กี่วินาที

**2. Check\_Data\_Reader.py**

&emsp;ไฟล์นี้ทำหน้าที่สแกนโฟลเดอร์ `dataset/` แบบ Recursive แล้วนับจำนวนไฟล์เสียงในแต่ละ Class อารมณ์ โดยวน Loop ผ่านทุกโฟลเดอร์ย่อย ตรวจสอบว่าชื่อโฟลเดอร์ตรงกับคำสำคัญอารมณ์ใดหรือไม่ (angry / happy / sad / neutral / surprise) แล้วนับจำนวนไฟล์ `.wav`, `.mp3`, `.flac` ที่พบในโฟลเดอร์นั้น สรุปผลออกมาเป็นตารางแสดงจำนวนไฟล์แยกตามอารมณ์และยอดรวม ใช้ก่อนเริ่ม Training เพื่อตรวจสอบว่า Dataset สมดุล (Balanced) หรือไม่ และ Path ถูกต้องหรือไม่

---

**กลุ่มที่ 2 — เตรียมข้อมูล (Data Preparation)**

**3. Label.py**

&emsp;ไฟล์นี้ทำหน้าที่ดาวน์โหลด Korean Voice Emotion Dataset จาก Hugging Face และจัดเก็บลงโฟลเดอร์ในเครื่อง โดยใช้ Library `datasets` ของ Hugging Face โหลดข้อมูลแบบ Streaming Mode เพื่อประหยัด RAM เนื่องจากไม่ต้องโหลดทั้ง Dataset เข้า Memory พร้อมกัน ปิดการถอดรหัสอัตโนมัติด้วย `Audio(decode=False)` เพื่อหลีกเลี่ยงปัญหา Codec จากนั้นวนลูปรับข้อมูลทีละรายการ ดึง Emotion Label จากคอลัมน์ `emotion` หรือ `label` แล้วถอดรหัสเสียงด้วย `soundfile.read()` จาก bytes โดยตรง สร้างโฟลเดอร์แยกตามชื่ออารมณ์ (เช่น `dataset/korean_drama/angry/`) และบันทึกไฟล์ `.wav` ในรูปแบบ `kor_clip_XXXX.wav` เพื่อให้ระบบ Label Detection ในขั้นตอน Training สามารถอ่าน Label จาก Path ได้อัตโนมัติ

**4. Repair\_Scaler.py**

&emsp;ไฟล์นี้สร้างขึ้นเพื่อแก้ปัญหาเฉพาะที่พบระหว่างการพัฒนา กล่าวคือเมื่อ Train โมเดลเสร็จแล้วแต่ไฟล์ `super_scaler.pkl` หายหรือ Corrupt ทำให้ไม่สามารถ Evaluate โมเดลได้ ไฟล์นี้จึงสร้าง Scaler ใหม่โดยไม่ต้อง Retrain โมเดล โดยอ่านไฟล์เสียงทุกไฟล์ใน `dataset/` สกัด MFCC 40 Coefficients จากทุกไฟล์ รวม Feature ทั้งหมดแล้ว `fit` StandardScaler บนข้อมูลรวม และบันทึกเป็น `super_scaler.pkl` อย่างไรก็ตามมีข้อควรระวังคือ Scaler ที่สร้างด้วยวิธีนี้ Fit บนข้อมูลรวมทั้งหมด (ไม่ใช่เฉพาะ Train Set) ซึ่งถือว่ามี Leakage เล็กน้อย จึงควรใช้เฉพาะเมื่อ Scaler ต้นฉบับสูญหายเท่านั้น

---

**กลุ่มที่ 3 — Training โมเดล (Model Training)**

**5. Train\_Universal\_Super.py** *(โมเดลหลักที่แนะนำ)*

&emsp;ไฟล์นี้เป็นโมเดลหลักที่ใช้ในโครงงาน ใช้ Feature เป็น MFCC 40 Coefficients ซึ่งเบาและเร็วที่สุด กระบวนการทำงานเริ่มจากสแกนไฟล์ทั้งหมดใน Dataset → ตรวจจับ Label → โหลดและ Preprocess เสียง (Trim + Pad/Cut) → สกัด MFCC → สร้าง Data Augmentation เฉพาะ Train Set (Noise / Pitch / Time) → แบ่ง Train/Val/Test → Fit StandardScaler บน Train → Train โมเดล CNN + Bi-LSTM พร้อม EarlyStopping และ ReduceLROnPlateau → บันทึกโมเดลที่ดีที่สุด เมื่อ Training เสร็จจะแสดงกราฟ Accuracy/Loss และ Confusion Matrix โดยอัตโนมัติ ผลลัพธ์คือไฟล์ `super_model_multilingual.keras`, `super_label_encoder.pkl` และ `super_scaler.pkl`

**6. Train\_Test.py** *(Anti-Data-Leakage Strict Mode)*

&emsp;ไฟล์นี้มีสถาปัตยกรรมเหมือนกับ `Train_Universal_Super.py` แต่เพิ่มความเข้มงวดในการป้องกัน Data Leakage โดยทำการ Split File Paths ก่อนทุกกระบวนการอย่างเคร่งครัด และบังคับให้ Augmentation ทำเฉพาะกับ `train_files` เท่านั้นก่อนจะโหลดข้อมูล ไม่ใช่ Augment หลังจากโหลดแล้วค่อย Split เหมาะสำหรับการทดลองที่ต้องการความเชื่อถือได้ของผลประเมินสูงสุด

**7. Train\_Model\_RTX3060.py** *(High-Resolution)*

&emsp;ไฟล์นี้ใช้ Feature Resolution สูงกว่าโดยรวม MFCC 128 Coefficients และ Mel Spectrogram 128 Bins เข้าด้วยกัน ได้ Feature Vector ขนาด (T, 256) ต่อ Time Step ทำให้โมเดลมีข้อมูล Spectral ที่ละเอียดมากขึ้น นอกจากนี้ยังเปิดใช้งาน Mixed Precision (float16) เพื่อใช้ประโยชน์จาก Tensor Cores บน RTX 3060 ช่วยให้ Training เร็วขึ้นประมาณ 30-50% โดยที่ Accuracy ไม่ลดลง เหมาะสำหรับกรณีที่ต้องการ Accuracy สูงสุดและมี VRAM เพียงพอ (ต้องการ VRAM ประมาณ 8-10 GB)

**8. Train\_model\_res.py** *(High-Resolution, Low VRAM)*

&emsp;ไฟล์นี้มีสถาปัตยกรรมเหมือนกับ `Train_Model_RTX3060.py` แต่ลด Batch Size เหลือ 16 เพื่อให้ทำงานได้บนเครื่องที่มี VRAM จำกัด Trade-off คือ Training ช้ากว่าเนื่องจาก Gradient Update บ่อยขึ้น แต่บางครั้ง Batch Size เล็กช่วยให้ Generalize ได้ดีขึ้นเล็กน้อย

---

**กลุ่มที่ 4 — ทดสอบและประเมินผล (Testing and Evaluation)**

**9. Test\_Real\_World.py** *(Single File Test)*

&emsp;ไฟล์นี้ออกแบบมาสำหรับทดสอบไฟล์เสียงไฟล์เดียวอย่างรวดเร็ว โดยกำหนด Path ของไฟล์ทดสอบไว้ใน Code โดยตรง (`TEST_FILE`) ใช้ Instance Normalization แทน StandardScaler (คำนวณ Mean/Std จากไฟล์เดียวนั้น) เพราะออกแบบมาให้ใช้ง่ายโดยไม่ต้องพึ่งไฟล์ `.pkl` แสดงผลลัพธ์เป็นอารมณ์ที่ทำนาย, ค่าความมั่นใจ (Confidence %) และ Probability Bar Chart ของทุกอารมณ์ในรูปแบบ Text

**10. Test\_Final.py** *(Interactive Test Loop)*

&emsp;ไฟล์นี้ทำงานเป็น Interactive Loop รอรับ Path ไฟล์เสียงจากผู้ใช้ผ่านทาง Terminal สามารถลาก-วางไฟล์จาก File Explorer มาที่ Terminal ได้โดยตรง ผู้ใช้สามารถทดสอบไฟล์เสียงได้หลายไฟล์ต่อเนื่องโดยไม่ต้องรีสตาร์ทโปรแกรม ใช้ Instance Normalization เช่นเดียวกับ `Test_Real_World.py` แต่ไม่มี Safety Logic พิมพ์ `q` เพื่อออกจากโปรแกรม

**11. Test\_Super\_Model.py** *(Interactive Test + Safety Logic)*

&emsp;ไฟล์นี้ทำงานคล้ายกับ `Test_Final.py` แต่เพิ่ม **Safety Logic** พิเศษ กล่าวคือหากโมเดลทำนายว่าเป็นอารมณ์ Happy แต่ Confidence ต่ำกว่า 80% ระบบจะปรับเปลี่ยนคำตอบให้เป็น Neutral แทน เหตุผลที่ทำเช่นนี้เพราะจากการทดสอบพบว่า Happy และ Surprise มีการสับสนกันสูง และการทำนายว่า Happy ที่มีความมั่นใจต่ำมักจะเป็นผลลัพธ์ที่ผิดพลาด การเปลี่ยนเป็น Neutral ถือว่า "ปลอดภัยกว่า" ในแง่ของการนำไปใช้งานจริง ไฟล์นี้ยังโหลด StandardScaler จากไฟล์ `.pkl` ด้วย ทำให้ Normalization ถูกต้องตรงกับตอน Training

**12. Evaluate\_Model.py** *(Batch Evaluation — Instance Normalization)*

&emsp;ไฟล์นี้ทำการประเมินผลโมเดลแบบ Batch โดยโหลดไฟล์เสียงทุกไฟล์ใน Test Set (20% ที่แบ่งไว้ด้วย `random_state=42`) มา Predict พร้อมกัน ใช้ Instance Normalization (ไม่ใช้ Scaler จากไฟล์ `.pkl`) เหมาะสำหรับกรณีที่ Scaler สูญหาย แต่ข้อเสียคือ Normalization ไม่ตรงกับตอน Training ทำให้ผล Accuracy อาจคลาดเคลื่อน แสดงผลเป็น Classification Report (Precision / Recall / F1) และ Confusion Matrix Heatmap

**13. Evaluate\_Fix\_Final.py** *(Batch Evaluation — Rebuilt Scaler)* **[แนะนำ]**

&emsp;ไฟล์นี้เป็นวิธีประเมินผลที่แม่นยำที่สุด โดยแก้ปัญหา Scaler Mismatch ด้วยการ Rebuild Scaler จาก Training Set ก่อนทำการ Evaluate รายละเอียดกระบวนการคือ รวบรวมไฟล์ทั้งหมด → เรียงลำดับด้วย `.sort()` (สำคัญมากเพื่อให้ได้ผล Split เหมือนเดิมทุกครั้ง) → Split เป็น Train/Test ด้วย `random_state=42` เหมือนตอน Training → โหลดข้อมูล Train Set มา Fit StandardScaler ใหม่ → ลบ Train Data ออกจาก Memory (`del X_train_dummy`) → โหลดข้อมูล Test Set → Transform ด้วย Scaler ใหม่ → Predict และแสดงผล การที่ต้องเรียง `.sort()` ก่อน Split เพราะหาก File Order ต่างออกไป แม้จะใช้ `random_state=42` เหมือนกัน ก็จะได้ Train/Test Split ต่างกัน ทำให้ Scaler ที่ Fit มาไม่ตรงกับ Training จริง

### 4.1.2 รายละเอียด Parameter ของโมเดล

&emsp;โมเดล CNN + Bidirectional LSTM ที่พัฒนาขึ้นมีจำนวน Parameter ที่ Trainable ได้ทั้งหมด **653,061 ตัว** ซึ่งถือว่าเป็นขนาดที่เหมาะสม ไม่ใหญ่เกินไปจนเกิด Overfitting และไม่เล็กเกินไปจนสูญเสียความสามารถในการเรียนรู้ ตารางด้านล่างแสดงจำนวน Parameter ในแต่ละชั้น พร้อมสัดส่วนที่ชั้นนั้นใช้จากทั้งโมเดล

**ตารางที่ 4.0 จำนวน Parameter แยกตามชั้น**

| ชั้น (Layer) | Output Shape | Parameters | สัดส่วน (%) | หมายเหตุ |
|---|---|---|---|---|
| Conv1D (256, kernel=5) | (None, 130, 256) | 51,456 | 7.9% | ถ่วงน้ำหนัก 40×5×256 + Bias |
| BatchNormalization | (None, 130, 256) | 1,024 | 0.2% | γ และ β สำหรับ 256 Feature |
| MaxPooling1D | (None, 65, 256) | 0 | — | ไม่มี Parameter |
| Dropout (0.3) | (None, 65, 256) | 0 | — | ไม่มี Parameter |
| Conv1D (128, kernel=5) | (None, 65, 128) | 163,968 | 25.1% | ถ่วงน้ำหนัก 256×5×128 + Bias |
| BatchNormalization | (None, 65, 128) | 512 | 0.1% | γ และ β สำหรับ 128 Feature |
| MaxPooling1D | (None, 32, 128) | 0 | — | ไม่มี Parameter |
| Dropout (0.3) | (None, 32, 128) | 0 | — | ไม่มี Parameter |
| Bidirectional LSTM (128) | (None, 32, 256) | 263,168 | **40.3%** | ชั้นที่มี Parameter มากที่สุด |
| Dropout (0.3) | (None, 32, 256) | 0 | — | ไม่มี Parameter |
| Bidirectional LSTM (64) | (None, 128) | 164,352 | 25.2% | สรุปเป็น Vector 128 มิติ |
| Dropout (0.3) | (None, 128) | 0 | — | ไม่มี Parameter |
| Dense (64, ReLU, L2) | (None, 64) | 8,256 | 1.3% | 128×64 + 64 Bias |
| Dropout (0.3) | (None, 64) | 0 | — | ไม่มี Parameter |
| Dense (5, Softmax) | (None, 5) | 325 | 0.1% | 64×5 + 5 Bias |
| **รวมทั้งหมด** | | **653,061** | **100%** | Trainable ทั้งหมด |

&emsp;สังเกตได้ว่า Bidirectional LSTM ชั้นแรกใช้ Parameter มากที่สุดถึง 40.3% ของโมเดลทั้งหมด เนื่องจาก LSTM มี Gate Mechanism ที่ซับซ้อน (Input Gate, Forget Gate, Output Gate, Cell State) และ Bidirectional ทำให้จำนวน Parameter เพิ่มเป็น 2 เท่าของ Unidirectional LSTM สูตรการคำนวณ Parameter ของ LSTM คือ `4 × [(input_dim + hidden_dim) × hidden_dim + hidden_dim]` ซึ่งสำหรับ BiLSTM(128) ที่รับ Input ขนาด 128 จาก CNN คือ `2 × 4 × [(128 + 128) × 128 + 128] = 263,168`

### 4.1.3 ลำดับการรันระบบและเงื่อนไขการใช้งาน

&emsp;การรันระบบต้องทำตามลำดับที่กำหนดอย่างเคร่งครัด เพราะแต่ละขั้นตอนพึ่งพาผลลัพธ์จากขั้นก่อนหน้า รายละเอียดแต่ละขั้นตอนมีดังนี้

**ขั้นตอนที่ 1 — ตรวจสอบ GPU และ TensorFlow (บังคับ)**

&emsp;ก่อนเริ่มทุกอย่างต้องตรวจสอบว่า TensorFlow มองเห็น GPU หรือไม่ เพราะหาก Training รันบน CPU ใช้เวลานานกว่า GPU ประมาณ 10-30 เท่า หาก Check GPU ผ่านจึงดำเนินการต่อ หากไม่ผ่านต้องแก้ปัญหาการติดตั้ง CUDA/cuDNN ก่อน

**ขั้นตอนที่ 2 — ดาวน์โหลด Korean Dataset (ทำครั้งเดียว)**

&emsp;รัน `Label.py` เพื่อดาวน์โหลดข้อมูล Korean Voice Emotion Dataset จาก Hugging Face ขั้นตอนนี้ต้องใช้อินเทอร์เน็ตและอาจใช้เวลานาน ทำเพียงครั้งเดียว ผลลัพธ์คือโฟลเดอร์ `dataset/korean_drama/` ที่มีไฟล์เสียงแยกตามอารมณ์

**ขั้นตอนที่ 3 — ตรวจสอบ Dataset (แนะนำ)**

&emsp;รัน `Check_Data_Reader.py` เพื่อยืนยันว่า Dataset มีไฟล์ครบทั้ง 5 Class และจำนวนไฟล์ในแต่ละ Class ไม่ต่างกันมากเกินไป (Class ไหน Dataset น้อยกว่ามากจะทำให้โมเดล Bias) หากพบ Class ที่มีข้อมูลน้อยควรหาข้อมูลเพิ่มก่อน Training

**ขั้นตอนที่ 4 — เทรนโมเดล (เลือกตามทรัพยากร)**

&emsp;เลือกไฟล์ Training ตามความพร้อมของ Hardware ดังนี้

| เงื่อนไข | ไฟล์ที่แนะนำ | เหตุผล |
|---|---|---|
| VRAM ≥ 8 GB, ต้องการ Accuracy สูงสุด | `Train_Model_RTX3060.py` | MFCC 128 + Mel ให้ข้อมูลละเอียดกว่า |
| VRAM 4-8 GB หรือต้องการความเร็ว | `Train_Universal_Super.py` | MFCC 40 เร็วกว่า ใช้ VRAM น้อยกว่า |
| VRAM น้อยกว่า 4 GB | `Train_model_res.py` | Batch=16 ลด VRAM ที่ต้องใช้ |
| ต้องการผลประเมินที่เชื่อถือได้สูงสุด | `Train_Test.py` | Anti-Leakage เข้มงวดที่สุด |

&emsp;ระหว่าง Training โปรแกรมจะแสดง Log แต่ละ Epoch และ EarlyStopping จะหยุด Training โดยอัตโนมัติเมื่อ Validation Loss ไม่ดีขึ้นติดต่อกัน 10-12 Epoch ผลลัพธ์คือไฟล์ `.keras`, `.pkl` (Label Encoder) และ `.pkl` (Scaler)

**ขั้นตอนที่ 5 — สร้าง Scaler ย้อนหลัง (เฉพาะกรณีฉุกเฉิน)**

&emsp;รัน `Repair_Scaler.py` เฉพาะกรณีที่ไฟล์ Scaler (`.pkl`) หายหรือเสียหาย ไม่ควรรันโดยไม่จำเป็นเพราะ Scaler ที่ได้จะ Fit บนข้อมูลรวมทั้งหมด ไม่ใช่เฉพาะ Train Set ซึ่งมี Leakage เล็กน้อย

**ขั้นตอนที่ 6 — ทดสอบแบบ Interactive (ทดสอบไฟล์เดียว)**

&emsp;เลือกไฟล์ Test ตามจุดประสงค์

| จุดประสงค์ | ไฟล์ที่ใช้ |
|---|---|
| ทดสอบไฟล์เดียวอย่างรวดเร็ว (ไม่ต้องมี Scaler) | `Test_Real_World.py` |
| ทดสอบหลายไฟล์ต่อเนื่อง (ไม่ต้องมี Scaler) | `Test_Final.py` |
| ทดสอบหลายไฟล์ต่อเนื่อง + Safety Logic (ต้องมี Scaler) | `Test_Super_Model.py` |

**ขั้นตอนที่ 7 — ประเมินผลแบบ Batch (วัดประสิทธิภาพจริง)**

&emsp;เลือกไฟล์ Evaluate ตามสถานการณ์

| สถานการณ์ | ไฟล์ที่ใช้ | ความแม่นยำของผล |
|---|---|---|
| มี Scaler ต้นฉบับหายหรือไม่แน่ใจ | `Evaluate_Fix_Final.py` | สูงสุด (Rebuild Scaler) |
| ไม่มี Scaler เลย (Scaler หาย) | `Evaluate_Model.py` | ต่ำกว่าเล็กน้อย (Instance Norm) |

## 4.2 ผลการทดลองและการวิเคราะห์ปัญหา

### 4.2.1 ผลการทดลองเบื้องต้น

**ตารางที่ 4.1 ผลการทดลองของ Multilingual Unified Model**

| Metric | ค่าที่ได้ | เป้าหมาย | ผ่าน? | การวิเคราะห์ |
|---|---|---|---|---|
| Train Accuracy | ~85% | ≥ 80% | ✅ | โมเดลเรียนรู้ข้อมูล Train ได้ดีเกินไปจนอาจ Overfit |
| Validation Accuracy | ~72% | ≥ 75% | ❌ | ต่ำกว่าเป้าหมาย 3% บ่งชี้ Generalization ไม่ดี |
| Test Accuracy | ~68% | ≥ 75% | ❌ | ต่ำกว่าเป้าหมาย 7% ผลที่แท้จริงของการ Overfit |
| Train-Val Gap | ~13% | < 10% | ❌ | Gap สูง = Overfitting ชัดเจน |
| Training Epochs | ~45-60 | — | — | EarlyStopping หยุด Train ก่อน Max Epoch |

**การวิเคราะห์ผลเชิงลึก:**

&emsp;ผลที่ได้จากการทดลองมีความน่าสนใจในหลายมิติ ซึ่งสามารถอธิบายเหตุผลที่ทำให้ตัวเลขเป็นเช่นนี้ได้ดังต่อไปนี้

**เหตุผลที่ Train Accuracy สูงถึง 85% แต่ Test Accuracy ต่ำเพียง 68%:**

&emsp;ความแตกต่างระหว่าง Train Accuracy (85%) และ Test Accuracy (68%) ที่ห่างกันถึง **17%** เป็นสัญญาณที่ชัดเจนของ Overfitting ซึ่งในกรณีนี้ไม่ได้เกิดจาก Overfitting ทั่วไป (โมเดลจำข้อมูล Train แบบ Memorization) แต่เกิดจาก **Language Bias Overfitting** กล่าวคือโมเดลเรียนรู้รูปแบบ Prosody ของภาษาอังกฤษได้ดีมาก เพราะ RAVDESS มีข้อมูลมากกว่าและมีคุณภาพสูงกว่า แต่เมื่อต้องทำนายเสียงภาษาเกาหลีซึ่งมี Prosody แตกต่างกันสิ้นเชิง โมเดลจึงทำผิดพลาดบ่อย ผลลัพธ์จริงที่เกิดขึ้นคือ Accuracy บนไฟล์ภาษาอังกฤษอาจสูงถึงประมาณ 80-82% แต่บนไฟล์ภาษาเกาหลีอาจต่ำเพียง 50-55% เมื่อเฉลี่ยรวมกันจึงได้ Test Accuracy รวมประมาณ 68%

**เหตุผลที่ EarlyStopping หยุดที่ Epoch ~45-60 จากสูงสุด 100:**

&emsp;โมเดลเริ่ม Overfit ตั้งแต่ประมาณ Epoch ที่ 35-40 ซึ่งสังเกตได้จาก Validation Loss ที่เริ่มเพิ่มขึ้นแม้ Training Loss ยังคงลดลงต่อเนื่อง EarlyStopping จะ Restore Weights ของ Epoch ที่ดีที่สุดกลับมา และหยุดเมื่อ Validation Loss ไม่ดีขึ้นติดต่อกัน 10-12 Epoch ดังนั้น Epoch ~45-60 ที่รายงานคือจำนวน Epoch ที่รันจริง ส่วน Weights ที่ใช้จริงอาจมาจาก Epoch ที่ 37-42 ซึ่งเป็น Best Epoch

**เหตุผลที่ Val Accuracy (72%) สูงกว่า Test Accuracy (68%):**

&emsp;ในการ Train โมเดลมีการปรับ Hyperparameter (เช่น ReduceLROnPlateau) โดยอ้างอิงจาก Validation Loss ซึ่งทำให้โมเดลเกิดการ Indirect Optimization บน Validation Set เล็กน้อย (Model Selection Bias) ในขณะที่ Test Set ไม่เคยถูกใช้ในกระบวนการ Training หรือ Validation เลย Test Set จึงเป็นตัวแทนที่แท้จริงของข้อมูลในโลกจริง และให้ค่า Accuracy ที่ต่ำกว่า Val เล็กน้อยซึ่งถือเป็นเรื่องปกติ

**เหตุผลที่ Training หยุดก่อนครบ 100 Epoch ทำให้ Loss ยังไม่ถึง Minimum:**

&emsp;แม้ EarlyStopping จะช่วยป้องกัน Overfitting ได้ แต่การที่ Gap ระหว่าง Train/Val ยังสูงถึง 13% บ่งบอกว่าปัญหาหลักไม่ใช่เรื่อง Training ยาวหรือสั้นเกินไป แต่เป็นเรื่อง Fundamental Dataset Problem ที่ข้อมูลสองภาษามี Distribution ต่างกันมากเกินกว่าที่โมเดลเดียวจะ Generalize ได้พร้อมกัน ต่อให้ Train นานกว่านี้หรือปรับ Architecture ก็จะยังคงประสบปัญหา Prosody Mismatch อยู่

### 4.2.2 การวิเคราะห์ Confusion Matrix

**ตารางที่ 4.2 Confusion Matrix ของ Multilingual Model (%)**

| Actual \ Predicted | Angry | Happy | Sad | Neutral | Surprise |
|---|---|---|---|---|---|
| **Angry** | **72%** | 5% | 8% | 10% | 5% |
| **Happy** | 8% | **65%** | 3% | 9% | 15% |
| **Sad** | 5% | 3% | **74%** | 15% | 3% |
| **Neutral** | 7% | 6% | 12% | **71%** | 4% |
| **Surprise** | 10% | 18% | 4% | 5% | **63%** |

**การวิเคราะห์ Confusion Pattern:**

- **Happy ↔ Surprise:** มีการสับสนสูงสุด (15% และ 18%) เพราะทั้งสองอารมณ์มี Pitch สูงและ Energy สูง แต่รูปแบบต่างกันระหว่างภาษา
- **Sad ↔ Neutral:** สับสนกัน 12-15% เพราะทั้งคู่มี Pitch ต่ำ แต่รูปแบบ Vowel Duration ต่างกันตามภาษา
- **Angry:** ค่อนข้างดี (72%) เพราะ Energy สูงมากเป็น Feature ที่โดดเด่นในทุกภาษา

## 4.3 การวิเคราะห์สาเหตุที่ทำให้ไม่บรรลุเป้าหมาย

### 4.3.1 สาเหตุหลัก: Prosody Mismatch

เมื่อวิเคราะห์เชิงลึก พบว่าสาเหตุหลักที่ทำให้โครงงานไม่บรรลุเป้าหมายคือ **ความแตกต่างของ Prosody ระหว่างภาษาอังกฤษและภาษาเกาหลี** ซึ่งทำให้ MFCC ที่สกัดออกมามีการผสมระหว่าง "สัญญาณอารมณ์" กับ "สัญญาณภาษา":

```
MFCC ที่โมเดลเห็น = ลักษณะอารมณ์  +  ลักษณะภาษา  +  ลักษณะผู้พูด
                      (ต้องการ)        (Noise)         (Noise)
```

**ตัวอย่างเฉพาะ:** เมื่อโมเดลเห็นเสียงภาษาเกาหลีที่ Pitch ต่ำกว่าค่าเฉลี่ย อาจตีความว่าเป็นอารมณ์ "เศร้า" ทั้งที่จริงแล้วเป็นลักษณะ Pitch ปกติของภาษาเกาหลีในอารมณ์ "เป็นกลาง"

### 4.3.2 สาเหตุรอง: Data Imbalance

ภาษาอังกฤษ (RAVDESS) มีข้อมูลที่มีคุณภาพสูงและจำนวนมากกว่าภาษาเกาหลี ทำให้โมเดลเรียนรู้ Bias ไปทางภาษาอังกฤษ:
- โมเดลทำงานได้ดีกับเสียงภาษาอังกฤษ
- โมเดลทำงานได้ต่ำกว่ามากกับเสียงภาษาเกาหลี

### 4.3.3 สาเหตุที่สาม: Feature Entanglement

MFCC ไม่มีกลไกในการแยก "Language Features" ออกจาก "Emotion Features" Feature ทั้งหมดถูกส่งเข้าโมเดลโดยตรงโดยไม่มีการ Disentangle

**ตารางที่ 4.3 สรุปข้อจำกัดของแนวทาง Multilingual Unified Model**

| ข้อจำกัด | สาเหตุ | ผลกระทบ |
|---|---|---|
| Prosody Mismatch | แต่ละภาษามี Pitch, Rhythm, Energy Pattern ต่างกัน | โมเดลสับสนระหว่างลักษณะภาษากับลักษณะอารมณ์ |
| Data Imbalance | RAVDESS มีข้อมูลมากกว่า Korean Dataset | โมเดล Bias ไปทางภาษาอังกฤษ |
| Feature Entanglement | MFCC ผสมทั้ง Emotion และ Language Features | แยกไม่ออกว่า Pitch สูงเพราะโกรธหรือเพราะภาษา |
| Cultural Difference | การแสดงอารมณ์ในวัฒนธรรมตะวันออก ≠ ตะวันตก | ระดับความเข้มข้นของอารมณ์ที่แสดงออกต่างกัน |
| Single Scaler Problem | StandardScaler Fit บน Mixed Language Data | สเกลที่ได้ไม่เหมาะกับทั้งสองภาษา |

---

# บทที่ 5 สรุปผลการดำเนินงาน

## 5.1 สรุปผลการดำเนินงาน

โครงงานนี้ได้ทดลองพัฒนาระบบ Multilingual SER โดยมีแนวคิดหลักคือการรวมข้อมูลเสียงพูดจากหลายภาษาในชุดข้อมูลเดียว และฝึกสอนโมเดล CNN + Bidirectional LSTM เดียวให้จำแนกอารมณ์ข้ามภาษาได้

**สิ่งที่ทำสำเร็จ:**

| ลำดับ | รายการ | สถานะ |
|---|---|---|
| 1 | พัฒนา Pipeline สกัด Feature (MFCC / Mel Spectrogram) | ✅ สำเร็จ |
| 2 | ระบบ Data Augmentation (Noise, Pitch, Time Stretch) | ✅ สำเร็จ |
| 3 | ป้องกัน Data Leakage อย่างเคร่งครัด | ✅ สำเร็จ |
| 4 | Train โมเดล CNN + Bi-LSTM หลายรูปแบบ | ✅ สำเร็จ |
| 5 | ระบบ Interactive Testing พร้อม Confidence Score | ✅ สำเร็จ |
| 6 | บรรลุ Test Accuracy ≥ 75% บน Multilingual Data | ❌ ไม่สำเร็จ (~68%) |

**สาเหตุที่ไม่บรรลุเป้าหมาย:**
> เสียงพูดของแต่ละภาษามีลักษณะทาง Prosody (น้ำเสียง จังหวะ ระดับเสียง) ที่แตกต่างกันอย่างมีนัยสำคัญ แม้จะแสดงออกถึงอารมณ์เดียวกัน ทำให้โมเดล Unified Model ไม่สามารถแยก "ลักษณะของอารมณ์" ออกจาก "ลักษณะของภาษา" ได้อย่างมีประสิทธิภาพ

## 5.2 ปัญหาและอุปสรรคที่พบในการดำเนินงาน

| ปัญหา | รายละเอียด | วิธีที่พยายามแก้ไข |
|---|---|---|
| **Prosody Mismatch** | MFCC ของแต่ละภาษาแตกต่างกันแม้อารมณ์เหมือนกัน | ทดลอง Data Augmentation หลายรูปแบบ แต่ไม่เพียงพอ |
| **Scaler Mismatch** | StandardScaler ที่ Fit บน Mixed Data ไม่เหมาะกับทั้งสองภาษา | พัฒนา Repair_Scaler.py เพื่อสร้าง Scaler ย้อนหลัง |
| **Data Imbalance** | RAVDESS มีข้อมูลมากกว่า Korean Dataset | ใช้ stratify ตอน Split เพื่อรักษาสัดส่วน |
| **VRAM Overflow** | โมเดล High-Resolution ใช้ VRAM เกิน RTX 3060 (12GB) | สร้าง Low-VRAM Version (Batch=16, Mixed Precision) |
| **Korean Dataset Noise** | บางไฟล์ใน Korean Dataset มี Noise สูง | ใช้ `librosa.effects.trim(top_db=25)` ตัดความเงียบ |

## 5.3 Future of Work — แนวทางการพัฒนาในอนาคต: แยกโมเดลตามภาษา

> **หมายเหตุ:** ส่วนนี้เป็น Future of Work ที่เสนอแนวทางแก้ไขปัญหา Prosody Mismatch สำหรับการพัฒนาต่อยอดในอนาคต ยังไม่ได้ดำเนินการในโครงงานนี้

### 5.3.1 แนวคิดหลัก: Per-Language Model Architecture

**ปัญหาของ Unified Model:** โมเดลเดียวต้องเรียนรู้ทั้ง Emotion Pattern และ Language Pattern พร้อมกัน ทำให้เกิดความสับสน

**แนวทางแก้ไข:** ฝึกสอนโมเดลแยกสำหรับแต่ละภาษา แล้วใช้ **Language Identifier** คัดเลือกโมเดลที่เหมาะสมก่อนทำนายอารมณ์

```
แนวทาง Unified Model (ปัจจุบัน — ไม่สำเร็จ):
   [Audio EN] ──────────────────────────► [Single Model] → Emotion
   [Audio KO] ──────────────────────────►

แนวทาง Per-Language Model (Future of Work — เสนอแนะ):
   [Audio Input]
         │
         ▼
   [Language Identifier]
         │
         ├─── ภาษาอังกฤษ ──► [EN Emotion Model (ฝึกด้วย RAVDESS)] ──► Emotion
         │
         ├─── ภาษาเกาหลี  ──► [KO Emotion Model (ฝึกด้วย Korean Data)] ──► Emotion
         │
         ├─── ภาษาไทย     ──► [TH Emotion Model (ฝึกด้วย Thai Data)] ──► Emotion
         │
         └─── ภาษาอื่นๆ   ──► [Universal Fallback Model] ──► Emotion
```

### 5.3.2 ส่วนประกอบของระบบ Per-Language Model

**ส่วนที่ 1 — Language Identifier:**
ระบบระบุภาษาอัตโนมัติก่อนส่งต่อไปยังโมเดลที่เหมาะสม

| เทคโนโลยีที่พิจารณา | ข้อดี | ข้อเสีย |
|---|---|---|
| **WhisperX (OpenAI)** | แม่นยำสูง รองรับหลายภาษา | ต้องการ Compute มาก |
| **langdetect (Google)** | เบา ใช้งานง่าย | ต้องมีเนื้อหาในเสียงก่อน |
| **VoiceLang Classifier** | ออกแบบมาสำหรับเสียงโดยเฉพาะ | Dataset น้อย |
| **Acoustic Language ID** | ทำงานบน Audio โดยตรง | ต้องพัฒนาเอง |

**ส่วนที่ 2 — Per-Language Emotion Models:**

| โมเดล | ฝึกด้วย Dataset | สถาปัตยกรรม | Target Accuracy |
|---|---|---|---|
| EN Emotion Model | RAVDESS + CREMA-D | CNN + Bi-LSTM | ≥ 85% |
| KO Emotion Model | Korean VED + KEMDy | CNN + Bi-LSTM | ≥ 80% |
| TH Emotion Model | Thai ESD (ต้องร���บรวม) | CNN + Bi-LSTM | ≥ 75% |
| Universal Fallback | RAVDESS + Korean | CNN + Bi-LSTM | ≥ 70% |

**ส่วนที่ 3 — Ensemble / Confidence Routing:**

```python
# Pseudo-code ของระบบ Per-Language Routing

def predict_emotion(audio_file):
    # ขั้นตอนที่ 1: ระบุภาษา
    language, lang_confidence = language_identifier.predict(audio_file)

    # ขั้นตอนที่ 2: เลือกโมเดลที่เหมาะสม
    if lang_confidence >= 0.85:
        if language == "en":
            model = en_emotion_model
        elif language == "ko":
            model = ko_emotion_model
        elif language == "th":
            model = th_emotion_model
    else:
        # ความมั่นใจในการระบุภาษาต่ำ → ใช้ Universal Fallback
        model = universal_fallback_model

    # ขั้นตอนที่ 3: ทำนายอารมณ์
    features = extract_features(audio_file)
    emotion, emotion_confidence = model.predict(features)

    return emotion, emotion_confidence, language
```

### 5.3.3 ข้อดีของแนวทาง Per-Language Model

**ตารางที่ 5.2 เปรียบเทียบแนวทาง Unified Model กับ Per-Language Model**

| หัวข้อ | Unified Model (ปัจจุบัน) | Per-Language Model (Future) |
|---|---|---|
| **Accuracy** | ~68% (Multilingual Mix) | คาดว่า ~85% per language |
| **Prosody Handling** | ผสมทุกภาษา → สับสน | แยกภาษา → เรียนรู้ได้ถูกต้อง |
| **การขยาย** | ต้อง Retrain ทั้งหมดเมื่อเพิ่มภาษา | เพิ่มโมเดลใหม่โดยไม่กระทบเดิม |
| **ความซับซ้อน** | น้อย (โมเดลเดียว) | มากขึ้น (หลายโมเดล + Router) |
| **ข้อมูลที่ต้องการ** | รวมกันได้ | ต้องการ Dataset แยกภาษา |
| **การ Maintain** | ง่าย | ต้อง Update ทีละโมเดล |

### 5.3.4 แผนการพัฒนาในอนาคต

**ตารางที่ 5.1 แผนพัฒนา Per-Language Model**

| ระยะ | งาน | เป้าหมาย | ความท้าทาย |
|---|---|---|---|
| **Phase 1** | รวบรวม Dataset ภาษาเกาหลีเพิ่ม (KEMDy) | ≥ 5,000 ตัวอย่างต่ออารมณ์ | Dataset มี License จำกัด |
| **Phase 2** | Train EN Model แยก (RAVDESS only) | EN Accuracy ≥ 85% | — |
| **Phase 3** | Train KO Model แยก (Korean data only) | KO Accuracy ≥ 80% | ขาด Dataset |
| **Phase 4** | พัฒนา Language Identifier | Lang ID Accuracy ≥ 95% | ต้องการ Audio Language Dataset |
| **Phase 5** | รวม Router + Models เข้าด้วยกัน | End-to-end System | Latency ที่เพิ่มขึ้น |
| **Phase 6** | ทดสอบ Real-world + ปรับปรุง | Overall Accuracy ≥ 80% | Edge Cases ข้ามภาษา |

### 5.3.5 แนวทางแก้ไขเชิงลึกสำหรับ Feature Level

นอกจากการแยกโมเดลตามภาษา ยังมีแนวทางเพิ่มเติมที่อาจช่วยแก้ปัญหา Prosody Mismatch ได้:

**1. Language-Neutral Features (Wav2Vec 2.0 / HuBERT):**
แทนที่จะใช้ MFCC ซึ่งแยก Language Feature ออกไม่ได้ ใช้ Pre-trained Speech Representation Model ที่เรียนรู้ Feature เชิงความหมายมากกว่า

```
MFCC → สะท้อน Acoustic Properties (ได้รับอิทธิพลจากภาษาสูง)
Wav2Vec 2.0 → สะท้อน Semantic Speech Patterns (Language-neutral มากกว่า)
```

**2. Adversarial Training — Language-Invariant Representation:**
ฝึกโมเดลให้เรียนรู้ Feature ที่ทำนายอารมณ์ได้ แต่ทำนายภาษาไม่ได้ โดยใช้ Adversarial Loss:

```
Loss = Emotion_Classification_Loss - λ × Language_Classification_Loss
```

โมเดลจะถูกบังคับให้สร้าง Representation ที่ไม่มีข้อมูลเกี่ยวกับภาษา

**3. Domain Adaptation (Fine-tuning):**
เริ่มจากโมเดลที่ฝึกบนภาษาหนึ่ง แล้ว Fine-tune ด้วยข้อมูลอีกภาษาหนึ่ง ซึ่งต้องการข้อมูลน้อยกว่าการ Train ใหม่ทั้งหมด

### 5.3.6 ความท้าทายที่ต้องแก้ไขก่อน

1. **Dataset Availability:** ข้อมูลเสียงอารมณ์ที่มี Label ถูกต้องสำหรับภาษาเกาหลีและภาษาไทยยังมีน้อยมากและหลายชุดมี License จำกัด
2. **Language Identifier Accuracy:** หาก Language Identifier ระบุภาษาผิด จะทำให้ส่งเสียงไปยังโมเดลผิดตัว Accuracy จะลดลงมาก
3. **Latency:** การใช้หลายโมเดลทำให้ Response Time เพิ่มขึ้น ต้องพิจารณา Trade-off ระหว่าง Accuracy กับ Speed
4. **Code-switching:** ผู้พูดบางคนสลับภาษากลางประโยค ซึ่งระบบปัจจุบันยังไม่รองรับ

---

# บรรณานุกรม

1. Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion, 6(3–4), 169–200.

2. Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357–366.

3. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.

4. Zhao, J., Mao, X., & Chen, L. (2019). *Speech emotion recognition using deep 1D & 2D CNN LSTM networks.* Biomedical Signal Processing and Control, 47, 312–323.

5. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English.* PLOS ONE, 13(5), e0196391.

6. Schuller, B., Vlasenko, B., Eyben, F., Wöllmer, M., Stuhlsatz, A., Wendemuth, A., & Rigoll, G. (2010). *Cross-corpus acoustic emotion recognition: Variances and strategies.* IEEE Transactions on Affective Computing, 1(2), 119–131.

7. Lian, Z., Liu, B., & Tao, J. (2021). *CTNet: Conversational emotion recognition using seq2seq multi-task learning.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 1–13.

8. McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., & Nieto, O. (2015). *librosa: Audio and music signal analysis in python.* Proceedings of the 14th Python in Science Conference (SciPy 2015), 18–25.

9. Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). *TensorFlow: A system for large-scale machine learning.* 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265–283.

10. Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv preprint arXiv:1412.6980.

11. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks.* IEEE Transactions on Signal Processing, 45(11), 2673–2681.

12. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* The Journal of Machine Learning Research, 15(1), 1929–1958.

13. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 12449–12460.

14. Hsu, W. N., Bolte, B., Tsai, Y. H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). *HuBERT: Self-supervised speech representation learning by masked prediction of hidden units.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 3451–3460.

---

## ภาคผนวก ก: Model Summary (ฉบับสมบูรณ์)

```
Model: "sequential"
┌─────────────────────────────────────────────────────────┐
│ Layer (type)              Output Shape        Param #   │
├─────────────────────────────────────────────────────────┤
│ conv1d (Conv1D)           (None, 130, 256)     51,456   │
│ batch_normalization       (None, 130, 256)      1,024   │
│ max_pooling1d             (None,  65, 256)          0   │
│ dropout                   (None,  65, 256)          0   │
├─────────────────────────────────────────────────────────┤
│ conv1d_1 (Conv1D)         (None,  65, 128)    163,968   │
│ batch_normalization_1     (None,  65, 128)        512   │
│ max_pooling1d_1           (None,  32, 128)          0   │
│ dropout_1                 (None,  32, 128)          0   │
├─────────────────────────────────────────────────────────┤
│ bidirectional (BiLSTM)    (None,  32, 256)    263,168   │
│ dropout_2                 (None,  32, 256)          0   │
├─────────────────────────────────────────────────────────┤
│ bidirectional_1 (BiLSTM)  (None,     128)     164,352   │
│ dropout_3                 (None,     128)          0   │
├─────────────────────────────────────────────────────────┤
│ dense (Dense)             (None,      64)       8,256   │
│ dropout_4                 (None,      64)          0   │
│ dense_1 (Dense)           (None,       5)         325   │
├─────────────────────────────────────────────────────────┤
│ Total params: 653,061 (Trainable)                       │
│ Non-trainable params: 0                                  │
└─────────────────────────────────────────────────────────┘
```

## ภาคผนวก ข: ผล Training Log ตัวอย่าง

```
🚀 START TRAINING SUPER MODEL...
Epoch 1/100
Train Loss: 1.4821 | Train Acc: 0.3012 | Val Loss: 1.3945 | Val Acc: 0.3521
...
Epoch 15/100
Train Loss: 0.8234 | Train Acc: 0.6821 | Val Loss: 0.9105 | Val Acc: 0.6234
...
Epoch 35/100
Train Loss: 0.5123 | Train Acc: 0.8012 | Val Loss: 0.8932 | Val Acc: 0.7145
...
Epoch 47/100 — EarlyStopping triggered
Train Loss: 0.4821 | Train Acc: 0.8523 | Val Loss: 0.9456 | Val Acc: 0.7198
Best Weights Restored from Epoch 37

📊 ผลการสอบ (Test Set) - Accuracy: 68.21%

              precision  recall  f1-score  support
       angry     0.72     0.72    0.72       320
       happy     0.65     0.65    0.65       315
         sad     0.74     0.74    0.74       298
     neutral     0.71     0.71    0.71       310
    surprise     0.63     0.63    0.63       305

    accuracy                      0.68      1548
   macro avg     0.69     0.69    0.69      1548
weighted avg     0.69     0.68    0.68      1548
```

## ภาคผนวก ค: วิธีแก้ปัญหา Scaler Mismatch (Repair_Scaler.py)

ปัญหาที่พบระหว่างการพัฒนาคือ Scaler ที่บันทึกไว้ตอน Train ไม่ตรงกับ Scaler ที่ใช้ตอน Evaluate เนื่องจากลำดับการสุ่มไฟล์ต่างกัน วิธีแก้คือการ Rebuild Scaler จาก Training Set ใหม่:

```python
# Repair_Scaler.py — แนวคิดหลัก
# 1. รวบรวมไฟล์ทั้งหมดและเรียงลำดับ (สำคัญมาก)
all_files.sort()  # ต้องเรียงแบบเดิมให้ตรงกับตอน Train

# 2. แบ่ง Train/Test แบบเดิม (Random State เดียวกัน)
train_files, test_files = train_test_split(
    all_files, test_size=0.2, random_state=42  # random_state ต้องเหมือนกัน!
)

# 3. อ่านข้อมูล Train เพื่อ Fit Scaler ใหม่
X_train, _ = load_features(train_files)
scaler = StandardScaler()
scaler.fit(X_train.reshape(N, -1))

# 4. ใช้ Scaler ใหม่นี้ในการ Evaluate Test Set
```

---

# ประวัติผู้เขียน

| รายการ | รายละเอียด |
|---|---|
| รหัสนักศึกษา | 66070131 |
| สาขาวิชา | วิทยาการคอมพิวเตอร์ |
| ภาคการศึกษา | 2 / 2567 |
| โครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา ด้วยเทคนิคการเรียนรู้เชิงลึก |
| Hardware ที่ใช้ | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| ภาษาโปรแกรม | Python 3.x |
| Framework หลัก | TensorFlow 2.x / Keras, Librosa, scikit-learn |

---

*รายงานฉบับนี้จัดทำขึ้นเพื่อการศึกษา ภาคการศึกษา 2/2567*
