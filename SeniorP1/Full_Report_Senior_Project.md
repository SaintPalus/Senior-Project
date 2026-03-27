# รายงานโครงงานนักศึกษา
## ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษาด้วยเทคนิคการเรียนรู้เชิงลึก
### Multilingual Speech Emotion Recognition System Using Deep Learning Techniques

---

| รายการ | รายละเอียด |
|---|---|
| ชื่อโครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา |
| นักศึกษา | รหัส 66070131/66070062|
| สาขาวิชา | วิทยาการคอมพิวเตอร์ |
| ภาคการศึกษา | 2 / 2568 |

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

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTILINGUAL SER SYSTEM                       │
│           (ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา)             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
     [Training Pipeline]            [Inference Pipeline]
              │                               │
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  File Discovery  │           │   Audio Input    │
    │  (os.walk)       │           │  (WAV/MP3/FLAC)  │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  Label Detection │           │   Preprocessing  │
    │  Path / Filename │           │  Trim → Pad/Cut  │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Audio Load       │           │Feature Extraction│
    │ librosa.load()   │           │  MFCC (40 coeff) │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Silence Removal  │           │  Normalization   │
    │ trim(top_db=25)  │           │  StandardScaler  │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Feature Extract  │           │  CNN + Bi-LSTM   │
    │ MFCC → (T, 40)  │           │    Model         │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Data Augmentation│           │  Softmax Output  │
    │ Noise/Pitch/Time │           │  Emotion + Conf% │
    └────────┬─────────┘           └──────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Train/Val/Test  │
    │  Split + Scaler  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  CNN + Bi-LSTM   │
    │  Model Training  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Save Best Model │
    │  (.keras + .pkl) │
    └──────────────────┘
```

## 3.2 การออกแบบ Data Pipeline

### 3.2.1 Label Detection (2 วิธี)

ระบบตรวจจับ Label ของไฟล์เสียงได้สองวิธีเพื่อรองรับ Dataset หลายรูปแบบ:

**วิธีที่ 1 — Path-based Detection (สำหรับ Korean Dataset):**
```python
# ถ้า path มีโฟลเดอร์ /angry/ ก็ label = "angry"
for key, emotion_name in EMOTION_KEYWORDS.items():
    if f"/{key}/" in path_check:
        label = emotion_name
        break
```

**วิธีที่ 2 — Filename-based Detection (สำหรับ RAVDESS):**
```python
# ไฟล์ 03-01-05-01-01-01-12.wav → parts[2] = '05' → Angry
RAVDESS_MAP = {'03': 'happy', '04': 'sad', '05': 'angry', '01': 'neutral'}
parts = filename.split('-')
if len(parts) >= 3 and parts[2] in RAVDESS_MAP:
    label = RAVDESS_MAP[parts[2]]
```

### 3.2.2 Preprocessing Pipeline

```python
# โหลดเสียงด้วย Sample Rate มาตรฐาน
data, sr = librosa.load(file_path, sr=22050, duration=3)

# ตัดความเงียบที่ปลายเสียง
data, _ = librosa.effects.trim(data, top_db=25)

# ปรับให้ยาวเท่ากันทุกไฟล์ (3 วินาที = 66,150 samples)
if len(data) < SAMPLES_PER_TRACK:
    data = np.pad(data, (0, SAMPLES_PER_TRACK - len(data)), 'constant')
else:
    data = data[:SAMPLES_PER_TRACK]
```

### 3.2.3 การป้องกัน Data Leakage (Critical Process)

Data Leakage คือปัญหาที่ข้อมูล Test ปนเข้าสู่กระบวนการ Train ทำให้ผลประเมินสูงเกินจริง

```
❌ วิธีผิด (มี Data Leakage):
   1. Load ข้อมูลทั้งหมด
   2. Augment ทั้งหมด          ← ข้อมูล Augment ของ Train ปนกับ Test!
   3. Split Train/Test

✅ วิธีถูก (ไม่มี Data Leakage):
   1. Split File Paths ก่อน (train_files / val_files / test_files)
   2. Augment เฉพาะ train_files เท่านั้น (x3 samples)
   3. Load test_files แบบ Original ไม่ Augment
   4. Fit StandardScaler บน X_train เท่านั้น
   5. Transform X_val และ X_test ด้วย Scaler เดิม (ห้าม Fit ซ้ำ)
```

### 3.2.4 Data Augmentation

**ตารางที่ 3.3 เทคนิค Data Augmentation ที่ใช้**

| เทคนิค | วิธีการ | สูตร | ผลที่ต้องการ |
|---|---|---|---|
| **Gaussian Noise** | เพิ่ม Noise แบบสุ่ม | $x' = x + \alpha\mathcal{N}(0,1)$ | ทนต่อ Background Noise |
| **Pitch Shifting** | ปรับ Pitch ±0.7 Semitones | `librosa.effects.pitch_shift` | ทนต่อความแตกต่างระหว่างผู้พูด |
| **Time Stretching** | ขยาย/หดเวลา (rate=0.8) | `librosa.effects.time_stretch` | ทนต่อความเร็วในการพูด |

ผลลัพธ์: ไฟล์ต้นฉบับ 1 ไฟล์ → ตัวอย่าง 3 ตัวอย่าง (เพิ่ม Dataset 3 เท่า)

### 3.2.5 Normalization Strategy

```python
scaler = StandardScaler()
N, T, F = X_train.shape

# Fit เฉพาะบน Train (ห้าม Fit ซ้ำ!)
X_train = scaler.fit_transform(X_train.reshape(N, -1)).reshape(N, T, F)

# Transform ด้วย Train statistics เท่านั้น
X_val  = scaler.transform(X_val.reshape(-1, T*F)).reshape(-1, T, F)
X_test = scaler.transform(X_test.reshape(-1, T*F)).reshape(-1, T, F)

# บันทึก Scaler ไว้ใช้ตอน Inference
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

## 3.3 สถาปัตยกรรมโมเดล CNN + Bi-LSTM

```
Input Shape: (Batch_Size, 130 time_steps, 40 features)
                          │
          ┌───────────────┴────────────────────┐
          │         CNN Block 1                │
          │  Conv1D(256, kernel=5, ReLU)        │
          │  BatchNormalization                 │
          │  MaxPooling1D(pool=2)              │
          │  Dropout(0.3)                      │
          └───────────────┬────────────────────┘
                          │ (Batch, 65, 256)
          ┌───────────────┴────────────────────┐
          │         CNN Block 2                │
          │  Conv1D(128, kernel=5, ReLU)        │
          │  BatchNormalization                 │
          │  MaxPooling1D(pool=2)              │
          │  Dropout(0.3)                      │
          └───────────────┬────────────────────┘
                          │ (Batch, 32, 128)
          ┌───────────────┴────────────────────┐
          │      Bidirectional LSTM 1          │
          │  BiLSTM(128, return_seq=True)      │
          │  Dropout(0.3)                      │
          └───────────────┬────────────────────┘
                          │ (Batch, 32, 256)
          ┌───────────────┴────────────────────┐
          │      Bidirectional LSTM 2          │
          │  BiLSTM(64)                        │
          │  Dropout(0.3)                      │
          └───────────────┬────────────────────┘
                          │ (Batch, 128)
          ┌───────────────┴────────────────────┐
          │         Dense Layers               │
          │  Dense(64, ReLU, L2=0.001)         │
          │  Dropout(0.3)                      │
          │  Dense(5, Softmax)                 │
          └───────────────┬────────────────────┘
                          │
          Output: [Angry, Happy, Sad, Neutral, Surprise]
                  (ความน่าจะเป็นของแต่ละอารมณ์ รวมกัน = 1.0)
```

**Total Parameters: ~653,061**

**เหตุผลในการเลือกสถาปัตยกรรมนี้:**
- CNN Block สกัด Local Temporal Pattern (เช่น การเปลี่ยน Pitch ระยะสั้น)
- Bidirectional LSTM จับ Long-term Dependency (เช่น รูปแบบอารมณ์ที่กระจายทั้งประโยค)
- การรวมกันของทั้งสองชั้นให้ Accuracy สูงสุดสำหรับงาน SER ตามงานวิจัยของ Zhao et al. (2019)

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

### 4.1.1 โครงสร้างไฟล์โปรแกรม

```
SeniorP1/รวมภาษา Ai/
│
├── [เครื่องมือตรวจสอบ]
│   ├── Check Gpu.py           ← ตรวจสอบ GPU / TensorFlow Version
│   └── Check_Data_Reader.py   ← นับจำนวนไฟล์เสียงในแต่ละ Class
│
├── [เตรียมข้อมูล]
│   ├── Label.py               ← ดาวน์โหลด Korean Dataset จาก Hugging Face
│   └── Repair_Scaler.py       ← สร้าง StandardScaler ย้อนหลังจาก Dataset
│
├── [Training]
│   ├── Train_Universal_Super.py   ← โมเดลหลัก (MFCC 40, เร็วที่สุด)
│   ├── Train_Test.py              ← Anti-Leakage Strict Mode
│   ├── Train_Model_RTX3060.py     ← High-Resolution (MFCC 128 + Mel)
│   └── Train_model_res.py         ← High-Resolution, Low VRAM Version
│
├── [Inference / Testing]
│   └── Test_Real_World.py     ← ทดสอบไฟล์เสียงเดียว (Real-world Test)
│
└── Test modle/
    ├── Test_Final.py          ← Interactive Test (Universal Model)
    ├── Test_Super_Model.py    ← Interactive Test with Safety Logic
    ├── Evaluate_Model.py      ← Batch Evaluation (Instance Normalization)
    └── Evaluate_Fix_Final.py  ← Batch Evaluation (Rebuilt Scaler Method)
```

### 4.1.2 Model Summary

```
Model: CNN + Bidirectional LSTM (Super Universal Model)
Total Parameters: 653,061 (ทั้งหมดเป็น Trainable)
─────────────────────────────────────────────────────────
Layer                      Output Shape         Params
─────────────────────────────────────────────────────────
Conv1D (256, kernel=5)     (None, 130, 256)      51,456
BatchNormalization          (None, 130, 256)       1,024
MaxPooling1D (pool=2)       (None,  65, 256)           0
Dropout (0.3)               (None,  65, 256)           0
─────────────────────────────────────────────────────────
Conv1D (128, kernel=5)      (None,  65, 128)     163,968
BatchNormalization           (None,  65, 128)         512
MaxPooling1D (pool=2)        (None,  32, 128)           0
Dropout (0.3)                (None,  32, 128)           0
─────────────────────────────────────────────────────────
Bidirectional LSTM (128)     (None,  32, 256)     263,168
Dropout (0.3)                (None,  32, 256)           0
─────────────────────────────────────────────────────────
Bidirectional LSTM (64)      (None,      128)     164,352
Dropout (0.3)                (None,      128)           0
─────────────────────────────────────────────────────────
Dense (64, ReLU, L2)         (None,       64)       8,256
Dropout (0.3)                (None,       64)           0
Dense (5, Softmax)            (None,        5)         325
─────────────────────────────────────────────────────────
```

### 4.1.3 ลำดับการรันระบบ

```bash
# 1. ตรวจสอบ GPU และ TensorFlow
python "Check Gpu.py"

# 2. ตรวจสอบ Dataset ว่าครบและสมดุลหรือไม่
python "Check_Data_Reader.py"

# 3. เทรนโมเดล (เลือกอย่างใดอย่างหนึ่งตามทรัพยากร)
python "Train_Universal_Super.py"      # แนะนำ: เร็ว + ประหยัด
python "Train_Model_RTX3060.py"        # สำหรับ Accuracy สูงสุด

# 4. (ถ้าจำเป็น) สร้าง Scaler ย้อนหลัง
python "Repair_Scaler.py"

# 5. ทดสอบแบบ Interactive
python "Test modle/Test_Final.py"

# 6. ประเมินผลจริงแบบ Batch
python "Test modle/Evaluate_Fix_Final.py"
```

## 4.2 ผลการทดลองและการวิเคราะห์ปัญหา

### 4.2.1 ผลการทดลองเบื้องต้น

**ตารางที่ 4.1 ผลการทดลองของ Multilingual Unified Model**

| Metric | ค่าที่ได้ | เป้าหมาย | ผ่าน? | หมายเหตุ |
|---|---|---|---|---|
| Train Accuracy | ~85% | ≥ 80% | ✅ | โมเดลเรียนรู้ข้อมูล Train ได้ดี |
| Validation Accuracy | ~72% | ≥ 75% | ❌ | ต่ำกว่าเป้าหมาย 3% |
| Test Accuracy | ~68% | ≥ 75% | ❌ | ต่ำกว่าเป้าหมาย 7% |
| Train-Val Gap | ~13% | < 10% | ❌ | แสดงอาการ Overfitting |
| Training Epochs | ~45-60 | — | — | EarlyStopping ทำงาน |

**การวิเคราะห์เบื้องต้น:** Gap ระหว่าง Train Accuracy (85%) กับ Test Accuracy (68%) ที่สูงถึง 17% บ่งชี้ว่าโมเดลกำลัง Overfit กับข้อมูลหรือ Domain ใดโดเมนหนึ่งอย่างชัดเจน

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
