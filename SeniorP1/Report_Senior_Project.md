# รายงานโครงงานนักศึกษา
## ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา ด้วยเทคนิคการเรียนรู้เชิงลึก
### Multilingual Speech Emotion Recognition System Using Deep Learning Techniques

---

| รายการ | รายละเอียด |
|---|---|
| ชื่อโครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา |
| นักศึกษา | รหัส 66070131 |
| สาขาวิชา | วิทยาการคอมพิวเตอร์ / วิศวกรรมซอฟต์แวร์ |
| อาจารย์ที่ปรึกษา | - |
| ภาคการศึกษา | 2 / 2567 |

---

## บทคัดย่อ

โครงงานนี้นำเสนอการพัฒนาระบบรู้จำอารมณ์จากเสียงพูดแบบรวมหลายภาษา (Multilingual Speech Emotion Recognition) โดยใช้สถาปัตยกรรม Hybrid CNN + Bidirectional LSTM ซึ่งสามารถจำแนกอารมณ์ 5 ประเภท ได้แก่ โกรธ (Angry) มีความสุข (Happy) เศร้า (Sad) เป็นกลาง (Neutral) และประหลาดใจ (Surprise)

ในระหว่างการพัฒนา พบปัญหาสำคัญคือ **เสียงพูดของแต่ละภาษามีลักษณะทาง Prosody ที่แตกต่างกันอย่างมีนัยสำคัญ** ทำให้โมเดลที่ฝึกสอนด้วยข้อมูลหลายภาษาพร้อมกันไม่สามารถเรียนรู้รูปแบบอารมณ์ได้อย่างมีประสิทธิภาพ รายงานนี้นำเสนอแนวทางที่ดำเนินการ ปัญหาที่พบ บทวิเคราะห์สาเหตุ และทิศทางการพัฒนาต่อในอนาคต

**คำสำคัญ:** การรู้จำอารมณ์จากเสียง, หลายภาษา, Prosody, CNN, Bi-LSTM, MFCC

---

## Abstract

This project presents the development of a Multilingual Speech Emotion Recognition system using a Hybrid CNN + Bidirectional LSTM architecture to classify 5 emotion categories. During development, a critical challenge was discovered: **each language possesses significantly different prosodic characteristics**, making it difficult for a single unified model to effectively learn emotion patterns across languages. This report presents the methodology, identified problems, root cause analysis, and proposed future directions.

**Keywords:** Speech Emotion Recognition, Multilingual, Prosody, CNN, Bi-LSTM, MFCC

---

# บทที่ 1 บทนำ

## 1.1 ความเป็นมาและความสำคัญของปัญหา

การรู้จำอารมณ์จากเสียงพูด (Speech Emotion Recognition: SER) เป็นสาขาวิจัยที่มีความสำคัญในด้านปฏิสัมพันธ์ระหว่างมนุษย์กับคอมพิวเตอร์ ในโลกที่มีการใช้หลายภาษา การพัฒนาโมเดล SER ที่รองรับหลายภาษาในระบบเดียว (Multilingual Model) จะช่วยลดต้นทุนการพัฒนาและทำให้ระบบใช้งานได้กว้างขึ้น

แนวคิดเริ่มต้นของโครงงานนี้คือการรวมข้อมูลเสียงพูดจากหลายภาษา (ภาษาอังกฤษ ภาษาเกาหลี และอื่นๆ) เข้าด้วยกันในชุดข้อมูลเดียว แล้วฝึกสอนโมเดล CNN + Bi-LSTM เพื่อให้เรียนรู้รูปแบบอารมณ์ที่เป็นสากล (Language-Independent Emotion Features)

## 1.2 วัตถุประสงค์ของโครงงาน

1. พัฒนาระบบ SER แบบ Multilingual ด้วย CNN + Bi-LSTM
2. ศึกษาและวิเคราะห์ปัญหาที่เกิดขึ้นจากการรวมข้อมูลหลายภาษา
3. เสนอแนวทางแก้ไขสำหรับการพัฒนาในอนาคต

## 1.3 ขอบเขตของโครงงาน

- **ภาษาที่ใช้ทดสอบ:** ภาษาอังกฤษ (RAVDESS) และภาษาเกาหลี (Korean Voice Emotion Dataset)
- **ประเภทอารมณ์:** 5 ประเภท (Angry, Happy, Sad, Neutral, Surprise)
- **Features:** MFCC 40 coefficients และ MFCC 128 + Mel Spectrogram
- **Framework:** TensorFlow / Keras บน GPU NVIDIA RTX 3060

---

# บทที่ 2 ทฤษฎีและงานวิจัยที่เกี่ยวข้อง

## 2.1 ทฤษฎีพื้นฐานด้านการประมวลผลเสียง

### 2.1.1 Mel-Frequency Cepstral Coefficients (MFCC)

MFCC เป็นคุณลักษณะ (Feature) มาตรฐานสำหรับงาน Speech Processing ขั้นตอนการคำนวณประกอบด้วย:

1. **Framing & Windowing** — แบ่งสัญญาณเป็น Frame ขนาด ~25ms คูณด้วย Hamming Window
2. **FFT** — แปลงสัญญาณเป็น Frequency Domain
3. **Mel Filter Bank** — ผ่าน Triangular Filters บน Mel Scale: $m = 2595 \cdot \log_{10}(1 + f/700)$
4. **Log + DCT** — บีบอัดและแปลงเป็น Cepstral Coefficients

โครงงานนี้ใช้ **n_mfcc = 40** (โมเดลทั่วไป) และ **n_mfcc = 128** (โมเดล High-Resolution)

### 2.1.2 Mel Spectrogram

Mel Spectrogram แสดงการกระจายพลังงานของเสียงในมิติเวลาและความถี่ (Mel Scale) ให้ข้อมูลเชิงพื้นที่ (Spatial Information) ที่ MFCC เพียงอย่างเดียวไม่มี โครงงานนี้รวม MFCC และ Mel Spectrogram เข้าด้วยกัน:

```python
mfcc   = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)        # (128, T)
mel    = librosa.feature.melspectrogram(y=data, sr=sr)
mel_db = librosa.power_to_db(mel, ref=np.max)                    # (128, T)
result = np.concatenate((mfcc, mel_db), axis=0).T                # (T, 256)
```

### 2.1.3 Prosody และความแตกต่างระหว่างภาษา

**Prosody** คือคุณสมบัติเหนือระดับเสียง (Suprasegmental Features) ของภาษา ได้แก่:

| คุณสมบัติ | คำอธิบาย | ตัวอย่างความแตกต่าง |
|---|---|---|
| **Pitch (F0)** | ความถี่พื้นฐานของเสียง | ภาษาเกาหลีมีช่วง Pitch แคบกว่าภาษาอังกฤษ |
| **Duration** | ความยาวของพยางค์ | ภาษาอังกฤษมี Stress-timed, เกาหลีมี Syllable-timed |
| **Energy** | ความดังของเสียง | ระดับ Energy ที่แสดงอารมณ์ "โกรธ" ต่างกันในแต่ละวัฒนธรรม |
| **Rhythm** | จังหวะการพูด | ความเร็วและรูปแบบการหยุดพักต่างกัน |

**ปัญหาที่เกิดขึ้นในโครงงานนี้:** โมเดลที่ฝึกด้วยข้อมูลรวมหลายภาษาพบว่า MFCC จาก Pitch Pattern ของแต่ละภาษามีความแตกต่างกันสูงมาก ทำให้โมเดลสับสนระหว่าง "ลักษณะของภาษา" กับ "ลักษณะของอารมณ์"

## 2.2 สถาปัตยกรรม CNN + Bidirectional LSTM

### 2.2.1 Conv1D Layer
สกัด Local Temporal Patterns จาก Feature Sequence โดยใช้ Kernel เลื่อนผ่านมิติเวลา

### 2.2.2 Bidirectional LSTM
อ่านลำดับข้อมูลทั้งจากซ้ายไปขวา (Forward) และขวาไปซ้าย (Backward) พร้อมกัน ทำให้เข้าใจบริบทของเสียงพูดได้ทั้งสองทิศทาง

$$h_t = [\overrightarrow{\text{LSTM}}(x_t); \overleftarrow{\text{LSTM}}(x_t)]$$

### 2.2.3 Regularization
- **Dropout (0.3):** ป้องกัน Overfitting โดยสุ่มปิด Neurons ระหว่าง Train
- **L2 Regularization (λ=0.001):** จำกัดขนาด Weight ไม่ให้ใหญ่เกินไป
- **Batch Normalization:** ทำให้ Distribution ของ Activation เสถียร

## 2.3 เทคนิค Data Augmentation

เพื่อเพิ่มความหลากหลายของข้อมูลฝึกสอนโดยไม่ต้องเก็บข้อมูลเพิ่ม:

| เทคนิค | วิธีการ | ผลลัพธ์ |
|---|---|---|
| **Gaussian Noise** | เพิ่ม Noise แบบสุ่ม: $x' = x + \alpha\mathcal{N}(0,1)$ | ทำให้โมเดลทนทานต่อ Background Noise |
| **Pitch Shifting** | ปรับ Pitch ±0.7 Semitones | ทนทานต่อความแตกต่างระหว่างผู้พูด |
| **Time Stretching** | ขยาย/หดเวลา rate=0.8 | ทนทานต่อความเร็วในการพูดที่แตกต่าง |

ผลลัพธ์: 1 ไฟล์ต้นฉบับ → 3 ตัวอย่าง (เพิ่ม Dataset 3 เท่า)

## 2.4 ชุดข้อมูลที่ใช้

### RAVDESS (ภาษาอังกฤษ)
- 24 นักแสดงมืออาชีพ (12 ชาย, 12 หญิง)
- บันทึกในสภาพแวดล้อมที่ควบคุม (Anechoic Chamber)
- ระบุอารมณ์ผ่าน Filename Code: `XX-XX-[Emotion]-XX-XX-XX-XX.wav`

### Korean Voice Emotion Dataset (ภาษาเกาหลี)
- โหลดผ่าน Hugging Face Datasets API (Streaming Mode)
- แปลงจาก bytes → WAV ด้วย soundfile
- จัดเก็บในโฟลเดอร์ตามอารมณ์

### โครงสร้างชุดข้อมูล
```
dataset/
├── angry/      ← ไฟล์ WAV อารมณ์โกรธ (ทุกภาษา)
├── happy/      ← ไฟล์ WAV อารมณ์มีความสุข
├── sad/        ← ไฟล์ WAV อารมณ์เศร้า
├── neutral/    ← ไฟล์ WAV อารมณ์เป็นกลาง
└── surprise/   ← ไฟล์ WAV อารมณ์ประหลาดใจ
```

---

# บทที่ 3 การวิเคราะห์และออกแบบระบบ

## 3.1 ภาพรวมสถาปัตยกรรมระบบ

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTILINGUAL SER SYSTEM                      │
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
    │ x3 samples       │           │  Emotion + Conf% │
    └────────┬─────────┘           └──────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Train/Val Split │
    │  StandardScaler  │
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

## 3.2 การออกแบบ Data Pipeline อย่างละเอียด

### 3.2.1 Label Detection (2 วิธี)

```python
# วิธีที่ 1: ดูจากโครงสร้างโฟลเดอร์ (Path-based)
# → ถ้า path มี /angry/ ก็ label = "angry"
if f"/{key}/" in path_check:
    label = emotion_name

# วิธีที่ 2: ดูจากชื่อไฟล์ (RAVDESS Format)
# → filename: 03-01-05-01-01-01-12.wav → parts[2]='05' → angry
RAVDESS_MAP = {'03': 'happy', '04': 'sad', '05': 'angry', '01': 'neutral'}
if parts[2] in RAVDESS_MAP:
    label = RAVDESS_MAP[parts[2]]
```

### 3.2.2 การป้องกัน Data Leakage (สำคัญมาก)

```
❌ วิธีผิด (Data Leakage):
   1. Load All Data
   2. Augment All
   3. Split Train/Test   ← ข้อมูล Augment ของ Train ปนกับ Test!

✅ วิธีถูก (No Leakage):
   1. Split File Paths ก่อน (train_files / test_files)
   2. Augment เฉพาะ train_files (x3)
   3. Load test_files แบบ Original เท่านั้น
   4. Fit Scaler บน X_train เท่านั้น
   5. Transform X_val และ X_test ด้วย Scaler เดิม
```

### 3.2.3 Normalization Strategy

```python
scaler = StandardScaler()
N, T, F = X_train.shape

# Fit เฉพาะบน Train (ห้าม Fit ซ้ำ!)
X_train = scaler.fit_transform(X_train.reshape(N,-1)).reshape(N,T,F)

# Transform ด้วย Train statistics
X_val  = scaler.transform(X_val.reshape(-1,T*F)).reshape(-1,T,F)
X_test = scaler.transform(X_test.reshape(-1,T*F)).reshape(-1,T,F)

# บันทึก Scaler ไว้ใช้ตอน Inference
with open('super_scaler.pkl', 'wb') as f:
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
                  (ความน่าจะเป็นของแต่ละอารมณ์)
```

**Total Parameters: ~653,000**

## 3.4 โมเดลที่พัฒนาทั้งหมด

| ไฟล์ | Features | Batch | Augment | จุดเด่น |
|---|---|---|---|---|
| `Train_Universal_Super.py` | MFCC 40 | 64 | ✅ | เร็ว, ประหยัด RAM |
| `Train_Test.py` | MFCC 40 | 32 | ✅ | Data Leakage-proof เข้มงวดสุด |
| `Train_Model_RTX3060.py` | MFCC 128 + Mel | 32 | ✅ | High-Res, Mixed Precision |
| `Train_model_res.py` | MFCC 128 + Mel | 16 | ✅ | High-Res, VRAM ต่ำ |

## 3.5 Training Configuration

| Parameter | ค่าที่ใช้ | เหตุผล |
|---|---|---|
| Sample Rate | 22,050 Hz | มาตรฐาน Librosa / Speech Range |
| Duration | 3 วินาที | ครอบคลุมอารมณ์ / ไม่หนักเกินไป |
| Optimizer | Adam (lr=0.001) | ปรับ Learning Rate อัตโนมัติ |
| Loss | Categorical Cross-Entropy | Multi-class Classification |
| Epochs | 100–150 | หยุดเองด้วย EarlyStopping |
| EarlyStopping | patience=10–12 | ป้องกัน Overfitting |
| ReduceLROnPlateau | factor=0.5, patience=4–5 | Fine-tune ใกล้ Optimum |
| Mixed Precision | float16 | เพิ่มความเร็วบน RTX 3060 |

---

# บทที่ 4 ผลการทดลองและการวิเคราะห์ปัญหา

## 4.1 ผลการทดลองเบื้องต้น (Multilingual Model)

โมเดลที่ฝึกสอนด้วยข้อมูลรวมหลายภาษา (ภาษาอังกฤษ + ภาษาเกาหลี) ให้ผลดังนี้:

| Metric | ค่าที่ได้ | เป้าหมาย | ผ่าน? |
|---|---|---|---|
| Train Accuracy | ~85% | ≥ 80% | ✅ |
| Validation Accuracy | ~72% | ≥ 75% | ❌ |
| Test Accuracy | ~68% | ≥ 75% | ❌ |
| Gap (Train-Val) | ~13% | < 10% | ❌ (Overfit) |

## 4.2 ปัญหาที่พบ: Prosody Mismatch ข้ามภาษา

### 4.2.1 สาเหตุหลัก

ปัญหาสำคัญที่ค้นพบคือ **ลักษณะทาง Prosody ของแต่ละภาษามีความแตกต่างกันอย่างมีนัยสำคัญ** แม้จะเป็นอารมณ์เดียวกัน:

```
อารมณ์ "โกรธ" (Angry):
┌────────────────┬─────────────────┬──────────────────┐
│ คุณสมบัติ      │ ภาษาอังกฤษ      │ ภาษาเกาหลี       │
├────────────────┼─────────────────┼──────────────────┤
│ Pitch Range    │ กว้าง (สูงมาก)  │ แคบกว่า          │
│ Speech Rate    │ เร็ว, ไม่สม่ำเสมอ│ เร็ว แต่สม่ำเสมอ │
│ Energy Pattern │ เพิ่มขึ้นฉับพลัน │ เพิ่มทีละน้อย    │
│ Vowel Duration │ ยาว (Stress)    │ สั้นกว่า         │
└────────────────┴─────────────────┴──────────────────┘
```

### 4.2.2 ผลกระทบต่อ MFCC Features

MFCC สะท้อนโครงสร้างทาง Spectral ของเสียง ซึ่งได้รับอิทธิพลจากทั้ง **อารมณ์** และ **ภาษา** พร้อมกัน:

```
MFCC ที่โมเดลเห็น = ลักษณะอารมณ์ + ลักษณะภาษา + ลักษณะผู้พูด
                         (ต้องการ)    (noise)       (noise)
```

เมื่อรวมข้อมูลหลายภาษา โมเดลไม่สามารถแยก "ลักษณะอารมณ์" ออกจาก "ลักษณะภาษา" ได้อย่างชัดเจน ทำให้:
- โมเดล Overfit กับภาษาที่มีข้อมูลมากกว่า (ภาษาอังกฤษ)
- Accuracy บนข้อมูลภาษาเกาหลีต่ำกว่ามาก
- Confusion Matrix แสดงให้เห็นว่าโมเดลสับสนเฉพาะตอนอารมณ์ที่ Pitch Pattern ต่างกันระหว่างภาษา

### 4.2.3 การวิเคราะห์ด้วย Confusion Matrix

```
ตัวอย่างความสับสนที่พบบ่อย (Multilingual Model):

Actual\Predicted  Angry  Happy  Sad   Neutral  Surprise
Angry              72%    5%    8%     10%       5%
Happy              8%    65%    3%      9%      15%  ← สับสน Surprise สูง
Sad                5%     3%   74%     15%       3%
Neutral            7%     6%   12%     71%       4%
Surprise          10%    18%   4%       5%      63%  ← สับสน Happy สูง
```

Happy และ Surprise มีความสับสนสูงสุด เพราะ Pitch Contour ของทั้งสองอารมณ์ใกล้เคียงกัน และแตกต่างกันมากระหว่างภาษา

## 4.3 สรุปข้อจำกัดของแนวทาง Multilingual

| ข้อจำกัด | รายละเอียด |
|---|---|
| Prosody Mismatch | แต่ละภาษามี Pitch, Rhythm, Energy Pattern ต่างกัน |
| Data Imbalance | ข้อมูลภาษาอังกฤษมากกว่า ทำให้โมเดล Bias |
| Feature Entanglement | MFCC ผสม "ลักษณะอารมณ์" กับ "ลักษณะภาษา" แยกไม่ออก |
| Cultural Difference | การแสดงอารมณ์แบบวัฒนธรรมตะวันออก ≠ ตะวันตก |

---

# บทที่ 5 สรุปผลและทิศทางในอนาคต (Future of Work)

## 5.1 สรุปผลการดำเนินงาน

โครงงานนี้ได้พัฒนาระบบ SER แบบ Multilingual สำเร็จในระดับ Prototype และค้นพบข้อจำกัดสำคัญ:

1. ✅ พัฒนา Pipeline สกัด Feature และ Train โมเดล CNN + Bi-LSTM ได้สำเร็จ
2. ✅ ระบบ Data Augmentation (Noise, Pitch, Time Stretch) ทำงานถูกต้อง
3. ✅ ป้องกัน Data Leakage อย่างเคร่งครัดด้วยการ Split ไฟล์ก่อนทุกกระบวนการ
4. ✅ ระบบ Interactive Testing พร้อม Confidence Score และ Safety Logic
5. ❌ Accuracy บน Multilingual Test Set ต่ำกว่าเป้าหมาย เนื่องจาก Prosody Mismatch

## 5.2 Future of Work — แนวทางการพัฒนา: แยกโมเดลตามภาษา

### 5.2.1 แนวคิดหลัก

แทนที่จะฝึกโมเดลเดียวสำหรับทุกภาษา เปลี่ยนเป็น **ฝึกโมเดลแยกสำหรับแต่ละภาษา** แล้วใช้ Language Identifier คัดเลือกโมเดลที่เหมาะสมก่อนทำนาย

### 5.2.2 Pipeline ของแนวทางแยกภาษา

```
[Audio Input]
      │
      ▼
[Language Identifier]
  (ตรวจสอบภาษาอัตโนมัติ เช่น WhisperX, LangDetect)
      │
      ├──── ภาษาอังกฤษ ──► [EN Emotion Model] ──► Emotion
      │
      ├──── ภาษาเกาหลี  ──► [KO Emotion Model] ──► Emotion
      │
      ├──── ภาษาไทย     ──► [TH Emotion Model] ──► Emotion
      │
      └──── ภาษาอื่นๆ   ──► [Universal Fallback] ──► Emotion
```

### 5.2.3 ข้อดีของแนวทางนี้

- โมเดลแต่ละตัวเรียนรู้ Prosody เฉพาะของภาษานั้น → Accuracy สูงขึ้น
- สามารถ Fine-tune โมเดลแต่ละภาษาแยกอิสระกัน
- ขยายรองรับภาษาใหม่ได้โดยไม่กระทบโมเดลเดิม

### 5.2.4 ความท้าทายที่ต้องแก้ไขต่อ

- ต้องการข้อมูลอารมณ์ที่มี Label ถูกต้องสำหรับแต่ละภาษาแยกกัน
- Language Identifier ต้องมีความแม่นยำสูงเพื่อไม่ให้เลือกโมเดลผิด
- ข้อมูลภาษาไทยที่มี Label อารมณ์ยังมีน้อยมากในปัจจุบัน

---

# บรรณานุกรม

1. Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion, 6(3–4), 169–200.

2. Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357–366.

3. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.

4. Zhao, J., Mao, X., & Chen, L. (2019). *Speech emotion recognition using deep 1D & 2D CNN LSTM networks.* Biomedical Signal Processing and Control, 47, 312–323.

5. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLOS ONE, 13(5).

6. Schuller, B., et al. (2010). *Cross-corpus acoustic emotion recognition: Variances and strategies.* IEEE Transactions on Affective Computing, 1(2), 119–131.

7. Lian, Z., Liu, B., & Tao, J. (2021). *CTNet: Conversational emotion recognition using seq2seq multi-task learning.* IEEE/ACM Transactions on Audio, Speech, and Language Processing.

8. McFee, B., et al. (2015). *librosa: Audio and music signal analysis in python.* Proceedings of the 14th Python in Science Conference.

9. Abadi, M., et al. (2016). *TensorFlow: A system for large-scale machine learning.* 12th USENIX Symposium on OSDI.

10. Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv:1412.6980.

---

## ภาคผนวก ก: โครงสร้างไฟล์โปรแกรม

```
SeniorP1/รวมภาษา Ai/
│
├── [เครื่องมือตรวจสอบ]
│   ├── Check Gpu.py           ← ตรวจสอบ GPU / TensorFlow
│   └── Check_Data_Reader.py   ← นับไฟล์เสียงในแต่ละ Class
│
├── [เตรียมข้อมูล]
│   ├── Label.py               ← ดาวน์โหลด Korean Dataset
│   └── Repair_Scaler.py       ← สร้าง StandardScaler ย้อนหลัง
│
├── [Training]
│   ├── Train_Universal_Super.py   ← โมเดลหลัก (MFCC 40)
│   ├── Train_Test.py              ← Anti-Leakage Strict Mode
│   ├── Train_Model_RTX3060.py     ← High-Res (MFCC 128 + Mel)
│   └── Train_model_res.py         ← High-Res, Low VRAM
│
├── [Inference]
│   └── Test_Real_World.py     ← ทดสอบไฟล์เดียว
│
└── Test modle/
    ├── Test_Final.py          ← Interactive Test (Universal)
    ├── Test_Super_Model.py    ← Interactive Test (Safety Logic)
    ├── Evaluate_Model.py      ← Batch Evaluation (Instance Norm)
    └── Evaluate_Fix_Final.py  ← Batch Evaluation (Rebuilt Scaler)
```

## ภาคผนวก ข: Model Summary

```
Model: CNN + Bidirectional LSTM
Total Parameters: 653,061
─────────────────────────────────────────────────
Layer                    Output Shape     Params
─────────────────────────────────────────────────
Conv1D (256, k=5)        (None,130,256)   51,456
BatchNormalization        (None,130,256)    1,024
MaxPooling1D (2)          (None, 65,256)        0
Dropout (0.3)             (None, 65,256)        0
─────────────────────────────────────────────────
Conv1D (128, k=5)         (None, 65,128)  163,968
BatchNormalization         (None, 65,128)      512
MaxPooling1D (2)           (None, 32,128)        0
Dropout (0.3)              (None, 32,128)        0
─────────────────────────────────────────────────
Bidirectional LSTM (128)   (None, 32,256)  263,168
Dropout (0.3)              (None, 32,256)        0
─────────────────────────────────────────────────
Bidirectional LSTM (64)    (None,     128)  164,352
Dropout (0.3)              (None,     128)        0
─────────────────────────────────────────────────
Dense (64, ReLU, L2)       (None,      64)    8,256
Dropout (0.3)              (None,      64)        0
Dense (5, Softmax)         (None,       5)      325
─────────────────────────────────────────────────
```

## ภาคผนวก ค: ลำดับการรัน

```bash
# 1. ตรวจสอบ GPU
python "Check Gpu.py"

# 2. ตรวจสอบ Dataset
python "Check_Data_Reader.py"

# 3. เทรนโมเดล (เลือกอย่างใดอย่างหนึ่ง)
python "Train_Universal_Super.py"      # แนะนำ: เร็ว + ประหยัด
python "Train_Model_RTX3060.py"        # ถ้าต้องการ Accuracy สูงสุด

# 4. (ถ้าจำเป็น) สร้าง Scaler ย้อนหลัง
python "Repair_Scaler.py"

# 5. ทดสอบแบบ Interactive
python "Test modle/Test_Final.py"

# 6. ประเมินผลจริง (Batch)
python "Test modle/Evaluate_Fix_Final.py"
```
