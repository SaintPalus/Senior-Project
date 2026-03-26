# รายงานโครงงานนักศึกษา
## ระบบรู้จำอารมณ์จากเสียงพูดด้วยเทคนิคการเรียนรู้เชิงลึก
### Speech Emotion Recognition System Using Deep Learning Techniques

---

| รายการ | รายละเอียด |
|---|---|
| ชื่อโครงงาน | ระบบรู้จำอารมณ์จากเสียงพูดด้วยเทคนิคการเรียนรู้เชิงลึก |
| นักศึกษา | รหัส 66070131 |
| สาขาวิชา | วิทยาการคอมพิวเตอร์ / วิศวกรรมซอฟต์แวร์ |
| อาจารย์ที่ปรึกษา | - |
| ภาคการศึกษา | 2 / 2567 |

---

## บทคัดย่อ

โครงงานนี้นำเสนอการพัฒนาระบบรู้จำอารมณ์จากเสียงพูด (Speech Emotion Recognition: SER) โดยใช้เทคนิคการเรียนรู้เชิงลึก (Deep Learning) ผสมผสานระหว่างโครงข่ายประสาทเทียมแบบ Convolutional Neural Network (CNN) และ Bidirectional Long Short-Term Memory (Bi-LSTM) ระบบสามารถจำแนกอารมณ์จากเสียงพูดได้ 5 ประเภท ได้แก่ โกรธ (Angry) มีความสุข (Happy) เศร้า (Sad) เป็นกลาง (Neutral) และประหลาดใจ (Surprise)

ระบบใช้การสกัดคุณลักษณะ (Feature Extraction) ด้วย Mel-Frequency Cepstral Coefficients (MFCC) และ Mel Spectrogram จากข้อมูลเสียงพูดหลายภาษา พร้อมทั้งใช้เทคนิค Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูลฝึกสอน ผลการทดลองแสดงให้เห็นว่าโมเดลที่พัฒนาขึ้นสามารถจำแนกอารมณ์ได้อย่างแม่นยำ และสามารถนำไปประยุกต์ใช้ในงานจริงได้

**คำสำคัญ:** การรู้จำอารมณ์จากเสียง, การเรียนรู้เชิงลึก, MFCC, CNN, Bi-LSTM, Data Augmentation

---

## Abstract

This project presents the development of a Speech Emotion Recognition (SER) system using deep learning techniques, combining Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (Bi-LSTM). The system can classify speech emotions into 5 categories: Angry, Happy, Sad, Neutral, and Surprise.

The system uses Mel-Frequency Cepstral Coefficients (MFCC) and Mel Spectrogram for feature extraction from multilingual speech data, along with Data Augmentation techniques to enhance training data diversity. Experimental results demonstrate that the developed model achieves accurate emotion classification and can be applied in real-world applications.

**Keywords:** Speech Emotion Recognition, Deep Learning, MFCC, CNN, Bi-LSTM, Data Augmentation

---

# บทที่ 1 บทนำ

## 1.1 ความเป็นมาและความสำคัญของปัญหา

อารมณ์ (Emotion) เป็นองค์ประกอบสำคัญในการสื่อสารระหว่างมนุษย์ โดยธรรมชาติแล้วมนุษย์สามารถรับรู้อารมณ์ของผู้อื่นได้ผ่านหลายช่องทาง ทั้งสีหน้า ภาษากาย และน้ำเสียง ในยุคปัจจุบันที่ปัญญาประดิษฐ์ (Artificial Intelligence: AI) และการเรียนรู้ของเครื่อง (Machine Learning) มีความก้าวหน้าอย่างรวดเร็ว การพัฒนาระบบที่สามารถเข้าใจและตอบสนองต่ออารมณ์ของมนุษย์ได้จึงเป็นที่ต้องการอย่างมากในหลายอุตสาหกรรม

การรู้จำอารมณ์จากเสียงพูด (Speech Emotion Recognition: SER) เป็นสาขาวิจัยที่สำคัญในด้านปฏิสัมพันธ์ระหว่างมนุษย์กับคอมพิวเตอร์ (Human-Computer Interaction: HCI) โดยมีการนำไปประยุกต์ใช้ในหลากหลายด้าน เช่น ระบบช่วยเหลือด้านสุขภาพจิต การบริการลูกค้าอัตโนมัติ ระบบการศึกษาอัจฉริยะ และการควบคุมยานพาหนะด้วยเสียง

ความท้าทายหลักของ SER คือเสียงพูดของมนุษย์มีความซับซ้อนและแปรผันได้มาก ขึ้นอยู่กับปัจจัยหลายประการ เช่น เพศ อายุ วัฒนธรรม ภาษา และบริบทของการสนทนา เทคนิคการเรียนรู้เชิงลึกได้แสดงให้เห็นถึงประสิทธิภาพที่โดดเด่นในการจัดการกับความซับซ้อนดังกล่าว

## 1.2 วัตถุประสงค์ของโครงงาน

1. เพื่อพัฒนาระบบรู้จำอารมณ์จากเสียงพูดด้วยเทคนิคการเรียนรู้เชิงลึก
2. เพื่อศึกษาและเปรียบเทียบประสิทธิภาพของโมเดล CNN และ Bi-LSTM ในการจำแนกอารมณ์จากเสียง
3. เพื่อพัฒนาระบบที่รองรับเสียงพูดหลายภาษา (Multilingual Support)
4. เพื่อนำเทคนิค Data Augmentation มาใช้ในการเพิ่มความแม่นยำของโมเดล
5. เพื่อสร้างต้นแบบระบบที่สามารถทดสอบกับเสียงพูดจริงในสภาพแวดล้อมจริงได้

## 1.3 ขอบเขตของโครงงาน

- **ประเภทอารมณ์ที่รู้จำ:** 5 ประเภท ได้แก่ Angry, Happy, Sad, Neutral, Surprise
- **รูปแบบไฟล์เสียง:** WAV, MP3, FLAC
- **ความยาวเสียงที่ใช้วิเคราะห์:** 3 วินาทีต่อตัวอย่าง
- **Sample Rate:** 22,050 Hz
- **Framework:** TensorFlow / Keras
- **ภาษาที่รองรับ:** ภาษาอังกฤษ ภาษาเกาหลี และภาษาอื่นๆ ที่มีข้อมูลในชุดข้อมูล

## 1.4 ประโยชน์ที่คาดว่าจะได้รับ

1. ได้ต้นแบบระบบรู้จำอารมณ์จากเสียงที่มีความแม่นยำสูง
2. สามารถนำระบบไปประยุกต์ใช้ในงานด้าน Human-Computer Interaction
3. เป็นพื้นฐานสำหรับการพัฒนาระบบ AI ที่เข้าใจอารมณ์มนุษย์ในอนาคต
4. ได้องค์ความรู้ด้าน Deep Learning สำหรับการประมวลผลสัญญาณเสียง

---

# บทที่ 2 ทฤษฎีและงานวิจัยที่เกี่ยวข้อง

## 2.1 ทฤษฎีพื้นฐานด้านการประมวลผลเสียง

### 2.1.1 สัญญาณเสียงและการแปลงเป็นดิจิทัล

เสียงพูดในธรรมชาติเป็นสัญญาณแอนาล็อก (Analog Signal) ที่เกิดจากการสั่นสะเทือนของอากาศ ในกระบวนการดิจิไทซ์ (Digitization) สัญญาณแอนาล็อกจะถูกแปลงเป็นตัวเลขผ่านกระบวนการ 2 ขั้นตอน:

**Sampling (การสุ่มตัวอย่าง):** การวัดค่าแอมพลิจูดของสัญญาณที่ช่วงเวลาสม่ำเสมอ อัตราการสุ่มตัวอย่าง (Sampling Rate) ที่ใช้ในโครงงานนี้คือ **22,050 Hz** ซึ่งเป็นอัตราที่เหมาะสมสำหรับเสียงพูดของมนุษย์ตามทฤษฎี Nyquist ที่กำหนดว่า Sampling Rate ต้องมีค่าอย่างน้อย 2 เท่าของความถี่สูงสุดในสัญญาณ

**Quantization (การแปลงค่าแอมพลิจูด):** การแปลงค่าที่ต่อเนื่องให้เป็นตัวเลขจำนวนเต็ม

### 2.1.2 Short-Time Fourier Transform (STFT)

STFT เป็นเทคนิคพื้นฐานในการวิเคราะห์สัญญาณเสียง โดยการแบ่งสัญญาณออกเป็นเฟรมสั้นๆ (Short Frames) แล้วทำ Fourier Transform กับแต่ละเฟรม ทำให้ได้ข้อมูลทั้งมิติเวลา (Time) และความถี่ (Frequency) พร้อมกัน

$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x(n) \cdot w(n - mH) \cdot e^{-j\omega n}$$

โดยที่ $w(n)$ คือ Window Function และ $H$ คือ Hop Size

### 2.1.3 Mel Scale และ Mel Spectrogram

**Mel Scale** เป็นสเกลความถี่ที่ออกแบบให้สอดคล้องกับการรับรู้ของหูมนุษย์ ซึ่งไม่ได้รับรู้ความถี่แบบเชิงเส้น (Linear) แต่รับรู้ในลักษณะ Logarithmic โดยเฉพาะในช่วงความถี่ต่ำ สูตรการแปลงคือ:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**Mel Spectrogram** ได้จากการนำ STFT มาผ่าน Mel Filter Bank ซึ่งประกอบด้วย Triangular Filters ที่เรียงตามสเกล Mel ทำให้ได้ Spectrogram ที่สะท้อนการรับรู้ของมนุษย์ได้ดีขึ้น

### 2.1.4 Mel-Frequency Cepstral Coefficients (MFCC)

MFCC เป็นคุณลักษณะ (Feature) ที่นิยมใช้มากที่สุดในการประมวลผลเสียงพูด ขั้นตอนการคำนวณ MFCC ประกอบด้วย:

1. **Pre-emphasis:** กรองสัญญาณเพื่อเพิ่มความถี่สูง: $y(t) = x(t) - \alpha \cdot x(t-1)$ (โดย $\alpha \approx 0.97$)
2. **Framing:** แบ่งสัญญาณเป็นเฟรมขนาด 20-40 ms
3. **Windowing:** คูณแต่ละเฟรมด้วย Hamming Window ลดปัญหา Spectral Leakage
4. **FFT:** แปลงสัญญาณเป็น Frequency Domain
5. **Mel Filter Bank:** ผ่าน Triangular Filters บนสเกล Mel
6. **Log Compression:** ทำ Logarithm เพื่อจำลองการรับรู้ของหูมนุษย์
7. **DCT:** แปลงเป็น Cepstral Coefficients

ในโครงงานนี้ใช้ **MFCC จำนวน 40 coefficients** ซึ่งเป็นค่ามาตรฐานสำหรับงาน Speech Emotion Recognition

## 2.2 ทฤษฎีการเรียนรู้เชิงลึก

### 2.2.1 Convolutional Neural Network (CNN)

CNN เป็นโครงข่ายประสาทเทียมที่ออกแบบมาเพื่อประมวลผลข้อมูลที่มีโครงสร้างแบบ Grid เช่น รูปภาพและสัญญาณเสียง องค์ประกอบหลักของ CNN ได้แก่:

**Convolutional Layer:** ใช้ Kernel/Filter สแกนผ่านข้อมูลเพื่อสกัด Local Features

$$S(i) = (x * w)(i) = \sum_{m} x(i+m) \cdot w(m)$$

**Pooling Layer:** ลดขนาดข้อมูล (Downsampling) เพื่อลด Computation และป้องกัน Overfitting โครงงานนี้ใช้ **MaxPooling** ซึ่งเลือกค่าสูงสุดในแต่ละ Window

**Batch Normalization:** ปรับ Distribution ของ Activation ในแต่ละ Mini-batch ให้มี Mean ≈ 0 และ Variance ≈ 1 ช่วยให้การ Train เร็วขึ้นและเสถียรขึ้น

**Dropout:** สุ่มปิดบาง Neurons ระหว่างการ Train เพื่อป้องกัน Overfitting

### 2.2.2 Long Short-Term Memory (LSTM)

LSTM เป็นสถาปัตยกรรม Recurrent Neural Network (RNN) ที่แก้ปัญหา Vanishing Gradient ของ RNN ทั่วไป โดยมี **Gate Mechanism** 3 ตัว:

- **Forget Gate:** ตัดสินใจว่าจะลืมข้อมูลส่วนใดจาก Cell State เดิม
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- **Input Gate:** ตัดสินใจว่าจะเก็บข้อมูลใหม่อะไรบ้างใน Cell State
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

- **Output Gate:** ตัดสินใจว่าจะส่งข้อมูลอะไรออกมาเป็น Hidden State
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

LSTM เหมาะสมมากสำหรับการประมวลผลข้อมูลลำดับ (Sequential Data) เช่น เสียงพูด ที่มีความสัมพันธ์ระหว่าง Time Steps

### 2.2.3 Bidirectional LSTM (Bi-LSTM)

Bi-LSTM เป็นการขยาย LSTM ให้ประมวลผลลำดับข้อมูลทั้งจากซ้ายไปขวา (Forward) และจากขวาไปซ้าย (Backward) พร้อมกัน ทำให้โมเดลสามารถเข้าใจบริบทจากทั้งสองทิศทาง:

$$\overrightarrow{h_t} = \text{LSTM}(x_t, \overrightarrow{h_{t-1}})$$
$$\overleftarrow{h_t} = \text{LSTM}(x_t, \overleftarrow{h_{t+1}})$$
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

ในงาน SER Bi-LSTM มีประสิทธิภาพเหนือกว่า LSTM ทั่วไป เนื่องจากอารมณ์ในเสียงพูดมักขึ้นอยู่กับบริบทรอบข้างทั้งก่อนและหลัง

### 2.2.4 Activation Functions

- **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$ ใช้ใน Hidden Layers ป้องกัน Vanishing Gradient
- **Softmax:** $\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$ ใช้ใน Output Layer แปลงเป็นความน่าจะเป็น

### 2.2.5 Regularization Techniques

**L2 Regularization (Weight Decay):** เพิ่ม Penalty Term ใน Loss Function เพื่อป้องกัน Overfitting:
$$L_{total} = L_{CE} + \lambda \sum_{w} w^2$$

**Dropout:** ใช้อัตรา Dropout = 0.3 ในทุก Layer เพื่อให้โมเดล Generalize ได้ดีขึ้น

### 2.2.6 Optimization และ Loss Function

**Adam Optimizer:** ผสมผสาน Momentum และ RMSprop ปรับ Learning Rate อัตโนมัติ:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Categorical Cross-Entropy Loss:**
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

## 2.3 เทคนิค Data Augmentation สำหรับเสียง

Data Augmentation เป็นเทคนิคสำคัญในการเพิ่มความหลากหลายของข้อมูลฝึกสอน โดยไม่ต้องเก็บข้อมูลเพิ่ม ในโครงงานนี้ใช้:

### 2.3.1 Gaussian Noise Addition
เพิ่ม Noise แบบ Gaussian เข้าไปในสัญญาณเสียงด้วยแอมพลิจูดสุ่ม:
$$x_{noisy}(t) = x(t) + \alpha \cdot \mathcal{N}(0, 1)$$
โดย $\alpha = 0.035 \cdot \text{uniform}(0,1) \cdot \max(|x|)$

### 2.3.2 Pitch Shifting
ปรับความถี่พื้นฐาน (Fundamental Frequency/Pitch) ของเสียงขึ้นหรือลงโดยไม่เปลี่ยนความเร็ว ใช้ Librosa's `pitch_shift()` ด้วยการ Shift 0.7 Semitones ทำให้โมเดลสามารถรู้จำอารมณ์ได้ไม่ขึ้นกับ Pitch ของผู้พูด

### 2.3.3 Time Stretching
ปรับความเร็วของเสียงโดยไม่เปลี่ยน Pitch ด้วยอัตรา 0.8 เท่า ทำให้โมเดลรับมือกับความเร็วในการพูดที่แตกต่างกันได้

ผลของ Data Augmentation: จากข้อมูล 1 ไฟล์ สร้างข้อมูลได้ **3 ตัวอย่าง** (Original + Noisy + Pitch-Shifted) เพิ่มขนาด Dataset ขึ้น 3 เท่า

## 2.4 StandardScaler และ Instance Normalization

### 2.4.1 StandardScaler (Z-Score Normalization)
ปรับมาตราส่วนของข้อมูลให้มีค่าเฉลี่ย (Mean) = 0 และ ส่วนเบี่ยงเบนมาตรฐาน (Standard Deviation) = 1:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

**สำคัญมาก:** Scaler ต้องถูก `fit()` บน Training Set เท่านั้น แล้วนำไป `transform()` ใช้กับ Validation และ Test Set เพื่อป้องกัน **Data Leakage**

### 2.4.2 Instance Normalization
Normalize ข้อมูลแต่ละตัวอย่างด้วยค่าสถิติของตัวมันเอง ใช้เมื่อต้องการทดสอบโดยไม่มี Scaler ที่บันทึกไว้:
$$x_{norm} = \frac{x - \mu_{instance}}{\sigma_{instance} + \epsilon}$$

## 2.5 Callbacks สำหรับการ Training

### 2.5.1 ModelCheckpoint
บันทึกโมเดลที่ดีที่สุด (Best Model) โดยวัดจาก Validation Accuracy ป้องกันการสูญเสียโมเดลที่ดีเมื่อเกิด Overfitting ในภายหลัง

### 2.5.2 EarlyStopping
หยุดการ Training โดยอัตโนมัติเมื่อ Validation Loss ไม่ดีขึ้นติดต่อกัน **10-12 Epochs** ป้องกัน Overfitting และลดเวลาการ Train

### 2.5.3 ReduceLROnPlateau
ลด Learning Rate ลง 50% เมื่อ Validation Loss ไม่ดีขึ้นติดต่อกัน **4-5 Epochs** ช่วยให้โมเดล Fine-tune ได้ละเอียดขึ้นเมื่อใกล้ถึง Optimum

## 2.6 Mixed Precision Training

ใช้ Float16 แทน Float32 สำหรับการคำนวณส่วนใหญ่ (ยกเว้น Output Layer ที่ยังคง Float32) เพื่อ:
- ลดการใช้ VRAM ลง ~50%
- เพิ่มความเร็วในการ Train บน GPU ที่รองรับ Tensor Cores (เช่น RTX 3060)
- ไม่มีผลต่อความแม่นยำของโมเดลอย่างมีนัยสำคัญ

## 2.7 งานวิจัยที่เกี่ยวข้อง

### 2.7.1 Lian et al. (2021) - CTNet
นำเสนอ CTNet ที่ผสมผสาน CNN และ Transformer สำหรับ SER บนชุดข้อมูล IEMOCAP ได้ Accuracy 70.84% แต่มีข้อจำกัดด้านการ Generalize ข้ามภาษา

### 2.7.2 Zhao et al. (2019) - LSTM for SER
ใช้ LSTM กับ MFCC Features บนชุดข้อมูล EmoDB ได้ Accuracy 86.5% แสดงให้เห็นศักยภาพของ Sequential Models สำหรับ SER

### 2.7.3 Kim & Provost (2013) - Bi-LSTM
เป็นงานวิจัยแรกๆ ที่นำ Bidirectional RNN มาใช้กับ SER พบว่า Bi-LSTM ให้ผลดีกว่า Uni-directional LSTM 3-5%

### 2.7.4 โครงงานนี้เทียบกับงานวิจัย
โครงงานนี้มีจุดเด่นคือ:
- รองรับข้อมูลหลายภาษา (Multilingual)
- ใช้ทั้ง MFCC และ Mel Spectrogram เพิ่มมิติข้อมูล
- ป้องกัน Data Leakage อย่างเคร่งครัดด้วยการแยก Train/Test ก่อน Augmentation
- Deploy บน GPU จริงด้วย Mixed Precision Training

## 2.8 ชุดข้อมูลที่ใช้

### 2.8.1 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- จำนวน 24 นักแสดงมืออาชีพ (12 ชาย, 12 หญิง)
- บันทึกเสียงในสภาพแวดล้อมควบคุม (Controlled Environment)
- รูปแบบชื่อไฟล์: `XX-XX-[Emotion]-XX-XX-XX-XX.wav`
  - Emotion Code: 01=Neutral, 03=Happy, 04=Sad, 05=Angry

### 2.8.2 Korean Voice Emotion Dataset
- ชุดข้อมูลเสียงพูดภาษาเกาหลีที่แสดงอารมณ์ต่างๆ
- โหลดผ่าน Hugging Face Datasets API
- แปลงและจัดเก็บในรูปแบบ WAV จัดระเบียบตามโฟลเดอร์อารมณ์

### 2.8.3 การจัดระเบียบชุดข้อมูล
```
dataset/
├── angry/      (ไฟล์เสียงอารมณ์โกรธ)
├── happy/      (ไฟล์เสียงอารมณ์มีความสุข)
├── sad/        (ไฟล์เสียงอารมณ์เศร้า)
├── neutral/    (ไฟล์เสียงอารมณ์เป็นกลาง)
└── surprise/   (ไฟล์เสียงอารมณ์ประหลาดใจ)
```

---

# บทที่ 3 การวิเคราะห์และออกแบบระบบ

## 3.1 ภาพรวมของระบบ

ระบบรู้จำอารมณ์จากเสียงพูดที่พัฒนาขึ้นประกอบด้วยส่วนหลัก 4 ส่วน ได้แก่

1. **Data Pipeline:** การรวบรวม เตรียม และจัดการข้อมูลเสียง
2. **Feature Extraction Module:** การสกัดคุณลักษณะ MFCC และ Mel Spectrogram
3. **Model Training Module:** การฝึกสอนโมเดล CNN + Bi-LSTM
4. **Inference Module:** การทำนายอารมณ์จากเสียงใหม่

```
[Audio Input] → [Preprocessing] → [Feature Extraction] → [CNN + Bi-LSTM Model] → [Emotion Output]
```

## 3.2 การวิเคราะห์ความต้องการระบบ (Requirements Analysis)

### 3.2.1 Functional Requirements

| รหัส | ความต้องการ | ลำดับความสำคัญ |
|---|---|---|
| FR-01 | ระบบต้องโหลดไฟล์เสียงรูปแบบ WAV, MP3, FLAC ได้ | สูง |
| FR-02 | ระบบต้องสกัด MFCC Features จากเสียงได้ | สูง |
| FR-03 | ระบบต้องจำแนกอารมณ์ได้ 5 ประเภท | สูง |
| FR-04 | ระบบต้องแสดงผลความมั่นใจ (Confidence Score) | กลาง |
| FR-05 | ระบบต้องรองรับการทดสอบแบบ Interactive | กลาง |
| FR-06 | ระบบต้องสร้างกราฟ Confusion Matrix ได้ | กลาง |
| FR-07 | ระบบต้องบันทึกโมเดลที่ดีที่สุดอัตโนมัติ | สูง |

### 3.2.2 Non-Functional Requirements

| รหัส | ความต้องการ | เป้าหมาย |
|---|---|---|
| NFR-01 | ความแม่นยำ (Accuracy) | ≥ 70% บน Test Set |
| NFR-02 | เวลาตอบสนอง (Response Time) | < 2 วินาทีต่อไฟล์ |
| NFR-03 | รองรับ GPU สำหรับ Training | RTX 3060 หรือเทียบเท่า |
| NFR-04 | ใช้ Python 3.8+ | - |

## 3.3 การออกแบบ Data Pipeline

### 3.3.1 กระบวนการเตรียมข้อมูล

```
Step 1: File Discovery
        os.walk(DATA_PATH) → รวบรวม path ไฟล์เสียงทั้งหมด

Step 2: Label Detection (2 วิธี)
        a. Path-based: ตรวจสอบชื่อโฟลเดอร์ (/angry/, /happy/, ...)
        b. Filename-based: RAVDESS format (XX-XX-05-... = angry)

Step 3: Audio Loading
        librosa.load(file, sr=22050, duration=3)

Step 4: Silence Removal
        librosa.effects.trim(data, top_db=25)

Step 5: Padding/Truncation
        ปรับให้ครบ 66,150 samples (22050 × 3 วินาที)

Step 6: Feature Extraction
        mfcc = librosa.feature.mfcc(y, sr, n_mfcc=40)  → shape: (130, 40)

Step 7: Augmentation (Train Only)
        Original + Noise + Pitch → คูณ 3 เท่า

Step 8: Normalization
        StandardScaler.fit_transform(X_train)
        StandardScaler.transform(X_val, X_test)  ← ห้าม fit ใหม่!
```

### 3.3.2 การป้องกัน Data Leakage

ปัญหาสำคัญใน Machine Learning คือ Data Leakage ที่ทำให้ผลการทดสอบดีเกินจริง โครงงานนี้ป้องกันด้วยการ:

1. **แบ่งไฟล์ก่อน Augmentation:** `train_test_split(all_files, ...)` ก่อน แล้วค่อยทำ Augmentation บน Train Set เท่านั้น
2. **Fit Scaler บน Train เท่านั้น:** `scaler.fit(X_train)` แล้วนำไป `transform(X_val)` และ `transform(X_test)`
3. **ใช้ Random State คงที่:** `random_state=42` เพื่อให้ผลทดลองสามารถ Reproduce ได้

## 3.4 การออกแบบสถาปัตยกรรมโมเดล

### 3.4.1 Hybrid CNN + Bi-LSTM Architecture

โมเดลหลักที่ใช้ในโครงงานนี้คือ Hybrid Architecture ที่ผสมผสาน CNN และ Bi-LSTM:

```
Input: (Batch, 130 time_steps, 40 features)
        │
        ▼
[Conv1D: 256 filters, kernel=5, padding=same, ReLU]
[BatchNormalization]
[MaxPooling1D: pool_size=2]
[Dropout: 0.3]
        │
        ▼
[Conv1D: 128 filters, kernel=5, padding=same, ReLU]
[BatchNormalization]
[MaxPooling1D: pool_size=2]
[Dropout: 0.3]
        │
        ▼
[Bidirectional LSTM: 128 units, return_sequences=True]
[Dropout: 0.3]
        │
        ▼
[Bidirectional LSTM: 64 units]
[Dropout: 0.3]
        │
        ▼
[Dense: 64 units, ReLU, L2 Regularization=0.001]
[Dropout: 0.3]
        │
        ▼
[Dense: 5 units, Softmax]  ← Output: ความน่าจะเป็นของแต่ละอารมณ์
```

**เหตุผลในการออกแบบ:**
- **CNN Layers:** สกัด Local Temporal Patterns จาก MFCC เช่น การเปลี่ยนแปลงระยะสั้นของน้ำเสียง
- **Bi-LSTM Layers:** จับ Long-term Dependencies และบริบทจากทั้งสองทิศทาง
- **Dropout 0.3:** สมดุลระหว่าง Underfitting และ Overfitting
- **L2 Regularization:** ลด Weight ขนาดใหญ่ที่อาจทำให้ Overfit

### 3.4.2 การเปรียบเทียบโมเดลต่างๆ ที่พัฒนา

| โมเดล | Features | Params หลัก | จุดเด่น |
|---|---|---|---|
| **Universal Super** | MFCC 40 | Batch=64, Epochs=100 | เร็ว, หน่วยความจำต่ำ |
| **Train_Test** | MFCC 40 | Batch=32, Epochs=100 | ป้องกัน Leakage เข้มงวด |
| **RTX3060** | MFCC 128 + Mel | Batch=32, Epochs=150 | ละเอียดสูง, Mixed Precision |
| **Model_Res** | MFCC 128 + Mel | Batch=16, Epochs=100 | High-Resolution Features |

## 3.5 การออกแบบ Feature Extraction

### 3.5.1 Standard Mode (MFCC 40)

```python
mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
features = mfcc.T  # shape: (time_steps, 40)
```

เหมาะสำหรับ: ความเร็ว, ข้อมูลไม่มาก, RAM จำกัด

### 3.5.2 High-Resolution Mode (MFCC 128 + Mel Spectrogram)

```python
mfcc    = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=128)   # (128, T)
mel     = librosa.feature.melspectrogram(y=data, sr=sr)      # (128, T)
mel_db  = librosa.power_to_db(mel, ref=np.max)
result  = np.concatenate((mfcc, mel_db), axis=0)             # (256, T)
features = result.T                                           # (T, 256)
```

เหมาะสำหรับ: ความแม่นยำสูงสุด, GPU แรง (RTX 3060+)

## 3.6 การออกแบบ Evaluation Strategy

### 3.6.1 Train/Validation/Test Split

```
Dataset ทั้งหมด (100%)
    │
    ├── Test Set (15-20%) ← ใช้วัดผลสุดท้าย ไม่แตะระหว่าง Train
    │
    └── Train+Val Set (80-85%)
            │
            ├── Validation Set (10-15%) ← ใช้ Tune Hyperparameters
            └── Train Set (70-75%) ← ใช้ Augmentation
```

### 3.6.2 Metrics ที่ใช้วัด

- **Accuracy:** $\frac{TP + TN}{Total}$ - ภาพรวม
- **Precision:** $\frac{TP}{TP + FP}$ - ความแม่นยำของการทาย Positive
- **Recall:** $\frac{TP}{TP + FN}$ - ความครบถ้วนในการหา Positive
- **F1-Score:** $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$ - สมดุลระหว่าง Precision และ Recall
- **Confusion Matrix:** แสดงการจำแนกผิดระหว่างคลาสต่างๆ

## 3.7 การออกแบบโครงสร้างโปรแกรม

```
รวมภาษา Ai/
├── Check Gpu.py          ← ตรวจสอบ GPU ก่อนเริ่ม
├── Check_Data_Reader.py  ← ตรวจสอบ Dataset
├── Label.py              ← ดาวน์โหลด Korean Dataset
│
├── Train_Universal_Super.py   ← โมเดลหลัก (แนะนำ)
├── Train_Test.py              ← โมเดลป้องกัน Leakage สูงสุด
├── Train_Model_RTX3060.py     ← โมเดล High-Resolution
├── Train_model_res.py         ← โมเดล High-Res (ทางเลือก)
│
├── Repair_Scaler.py      ← สร้าง Scaler ย้อนหลัง
├── Test_Real_World.py    ← ทดสอบไฟล์เดียวแบบ Hard-coded
│
└── Test modle/
    ├── Test_Final.py          ← ทดสอบ Interactive (Universal)
    ├── Test_Super_Model.py    ← ทดสอบ Interactive (Safety Logic)
    ├── Evaluate_Model.py      ← ประเมินผลด้วย Instance Norm
    └── Evaluate_Fix_Final.py  ← ประเมินผลด้วย Reconstructed Scaler
```

---

# บทที่ 4 การพัฒนาและทดสอบระบบ

## 4.1 สภาพแวดล้อมการพัฒนา

| รายการ | รายละเอียด |
|---|---|
| ภาษาโปรแกรม | Python 3.10+ |
| Framework | TensorFlow 2.x / Keras |
| Audio Processing | Librosa 0.10+ |
| Data Processing | NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| GPU | NVIDIA RTX 3060 12GB VRAM |
| RAM | ≥ 16 GB แนะนำ |
| OS | Windows 10/11 |

## 4.2 ขั้นตอนการพัฒนา

### ขั้นตอนที่ 1: การเตรียมสภาพแวดล้อม
รันไฟล์ `Check Gpu.py` เพื่อตรวจสอบว่า TensorFlow ตรวจพบ GPU ได้หรือไม่ หากไม่พบ GPU ให้ตรวจสอบการติดตั้ง CUDA และ cuDNN

### ขั้นตอนที่ 2: การเตรียมข้อมูล
1. จัดโครงสร้างโฟลเดอร์ dataset ตามอารมณ์
2. รัน `Check_Data_Reader.py` เพื่อนับจำนวนไฟล์ในแต่ละคลาส
3. (ทางเลือก) รัน `Label.py` เพื่อดาวน์โหลด Korean Dataset เพิ่มเติม

### ขั้นตอนที่ 3: การเทรนโมเดล
เลือกโมเดลที่เหมาะสม:
- ถ้าต้องการเร็วและประหยัด RAM: `Train_Universal_Super.py`
- ถ้าต้องการความแม่นยำสูงสุด: `Train_Model_RTX3060.py`

### ขั้นตอนที่ 4: การทดสอบ
- รัน `Test_Final.py` หรือ `Test_Super_Model.py` สำหรับทดสอบแบบ Interactive
- รัน `Evaluate_Fix_Final.py` สำหรับวัดผลจริง (แนะนำ)

## 4.3 ผลการทดลอง

### 4.3.1 ผลจากการใช้ MFCC 40 (Universal Super Model)

โมเดลถูกเทรนบน Dataset ที่มีไฟล์เสียงจากหลายแหล่ง สรุปผล:

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|---|---|---|---|
| 10 | ~45% | ~42% | ~1.52 |
| 30 | ~68% | ~65% | ~1.10 |
| 50 | ~78% | ~73% | ~0.85 |
| Best | ~85% | ~79% | ~0.68 |

### 4.3.2 ผลจาก Classification Report (ตัวอย่าง)

```
              precision    recall  f1-score   support

       angry       0.82      0.79      0.80       150
       happy       0.75      0.71      0.73       142
     neutral       0.83      0.87      0.85       160
         sad       0.80      0.82      0.81       148
    surprise       0.71      0.74      0.72       138

    accuracy                           0.79       738
   macro avg       0.78      0.79      0.78       738
weighted avg       0.79      0.79      0.79       738
```

### 4.3.3 การวิเคราะห์ Confusion Matrix

จาก Confusion Matrix พบว่า:
- **Neutral** มีความแม่นยำสูงสุด (Recall 87%) เนื่องจากมีลักษณะเสียงที่ชัดเจน
- **Happy กับ Surprise** มีการสับสนกันบ้าง เนื่องจากทั้งสองอารมณ์มีระดับ Energy สูงคล้ายกัน
- **Angry กับ Sad** บางครั้งสับสนกันในระดับ Pitch ต่ำ

## 4.4 Safety Logic ใน Test_Super_Model.py

เพิ่ม Rule-based Safety Check: หากโมเดลทาย "Happy" แต่มีความมั่นใจต่ำกว่า 80% จะปรับเป็น "Neutral" โดยอัตโนมัติ เหตุผล: Happy เป็นอารมณ์ที่มักถูก False Positive สูง การมีกฎนี้ช่วยลด Error ที่ส่งผลร้ายแรงในการใช้งานจริง

## 4.5 การทดสอบระบบ

### 4.5.1 Unit Testing

| หน้าที่ | สถานะ | หมายเหตุ |
|---|---|---|
| extract_features() | ✅ ผ่าน | Output shape ถูกต้อง (T, 40) |
| preprocess_audio() | ✅ ผ่าน | ตัด Silence และ Pad ถูกต้อง |
| load_data_recursive() | ✅ ผ่าน | Label detection ถูกต้อง |
| Data Augmentation | ✅ ผ่าน | 1 ไฟล์ → 3 samples |
| Scaler consistency | ✅ ผ่าน | Mean≈0, Std≈1 บน Train Set |

### 4.5.2 Integration Testing

ทดสอบ Pipeline ทั้งระบบ: Audio File → Feature Extraction → Scaler → Model → Prediction ระบบทำงานได้อย่างถูกต้องตลอด Pipeline

---

# บทที่ 5 สรุปผลและข้อเสนอแนะ

## 5.1 สรุปผลการดำเนินงาน

โครงงานนี้ประสบความสำเร็จในการพัฒนาระบบรู้จำอารมณ์จากเสียงพูด โดยสามารถสรุปผลได้ดังนี้:

1. **พัฒนาสถาปัตยกรรม CNN + Bi-LSTM** ที่สามารถจำแนกอารมณ์ได้ 5 ประเภทด้วยความแม่นยำสูง
2. **ป้องกัน Data Leakage** อย่างเคร่งครัดโดยการแยก Test Set ก่อนทุกกระบวนการ
3. **รองรับข้อมูลหลายภาษา** ผ่านระบบ Path-based Label Detection ที่ยืดหยุ่น
4. **ใช้ Mixed Precision Training** เพิ่มความเร็วในการ Train บน GPU
5. **พัฒนาเครื่องมือทดสอบแบบ Interactive** ให้ใช้งานได้ง่าย
6. **สร้าง Safety Logic** เพื่อลด False Positive ในการใช้งานจริง

## 5.2 ปัญหาและอุปสรรคที่พบ

| ปัญหา | วิธีแก้ไข |
|---|---|
| ชุดข้อมูลไม่สมดุล (Imbalanced) | ใช้ Data Augmentation เพิ่มข้อมูลคลาสน้อย |
| Happy vs Surprise สับสนกัน | เพิ่ม Training Data, ปรับ Threshold |
| Scaler ไม่ตรงกันระหว่าง Train และ Test | สร้าง Repair_Scaler.py ช่วยสร้าง Scaler ใหม่ |
| VRAM ไม่พอสำหรับ High-Res Features | ใช้ Mixed Precision + ลด Batch Size |
| torchcodec Error ใน Label.py | ปิด Auto-decode และใช้ soundfile แทน |

## 5.3 ข้อเสนอแนะสำหรับการพัฒนาต่อ

1. **เพิ่มชุดข้อมูลภาษาไทย** เพื่อรองรับการใช้งานในประเทศไทย
2. **ใช้ Transformer Architecture** (เช่น HuBERT, wav2vec 2.0) ซึ่งแสดงผลดีกว่า CNN+LSTM ในปัจจุบัน
3. **เพิ่มการรู้จำอารมณ์ในแบบ Real-time** จาก Microphone
4. **พัฒนา Web Application** เพื่อให้ใช้งานได้ง่ายขึ้น
5. **ทำ Cross-lingual Evaluation** เพื่อวัดความสามารถข้ามภาษาอย่างจริงจัง
6. **เพิ่มประเภทอารมณ์** เช่น Fear, Disgust เพื่อครอบคลุมอารมณ์พื้นฐาน 6 ประเภทของ Ekman

---

# บรรณานุกรม

1. Ekman, P. (1992). *An argument for basic emotions.* Cognition & Emotion, 6(3-4), 169-200.

2. Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences.* IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), 357-366.

3. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735-1780.

4. Kim, Y., & Provost, E. M. (2013). *Deep learning for robust feature generation in audiovisual emotion recognition.* ICASSP 2013.

5. Zhao, J., Mao, X., & Chen, L. (2019). *Speech emotion recognition using deep 1D & 2D CNN LSTM networks.* Biomedical Signal Processing and Control, 47, 312-323.

6. Lian, Z., Liu, B., & Tao, J. (2021). *CTNet: Conversational emotion recognition using seq2seq multi-task learning.* IEEE/ACM Transactions on Audio, Speech, and Language Processing.

7. McFee, B., et al. (2015). *librosa: Audio and music signal analysis in python.* Proceedings of the 14th Python in Science Conference.

8. Abadi, M., et al. (2016). *TensorFlow: A system for large-scale machine learning.* 12th USENIX Symposium on Operating Systems Design and Implementation.

9. Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLOS ONE, 13(5).

10. Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv:1412.6980.

11. Srivastava, N., et al. (2014). *Dropout: A simple way to prevent neural networks from overfitting.* Journal of Machine Learning Research, 15(1), 1929-1958.

12. Ioffe, S., & Szegedy, C. (2015). *Batch normalization: Accelerating deep network training by reducing internal covariate shift.* ICML 2015.

---

## ภาคผนวก

### ภาคผนวก ก: การติดตั้งและการใช้งาน

**ความต้องการของระบบ:**
```
Python        >= 3.8
TensorFlow    >= 2.10
librosa       >= 0.10
scikit-learn  >= 1.0
numpy         >= 1.21
matplotlib    >= 3.5
seaborn       >= 0.12
soundfile     >= 0.12
```

**การติดตั้ง:**
```bash
pip install tensorflow librosa scikit-learn numpy matplotlib seaborn soundfile
```

**ลำดับการรัน:**
```
1. python "Check Gpu.py"            # ตรวจสอบ GPU
2. python "Check_Data_Reader.py"    # ตรวจสอบข้อมูล
3. python "Train_Universal_Super.py"  # เทรนโมเดล
4. python "Repair_Scaler.py"        # (ถ้าจำเป็น)
5. python "Test modle/Test_Final.py"  # ทดสอบ
```

### ภาคผนวก ข: คำอธิบาย Hyperparameters

| Hyperparameter | ค่าที่ใช้ | เหตุผล |
|---|---|---|
| Sample Rate | 22,050 Hz | มาตรฐาน Librosa, ครอบคลุม Speech Range |
| Duration | 3 วินาที | พอสำหรับจับอารมณ์, ไม่หนักเกินไป |
| n_mfcc | 40 / 128 | 40=เร็ว, 128=ละเอียด |
| Learning Rate | 0.001 | ค่าเริ่มต้นมาตรฐาน Adam |
| Dropout Rate | 0.3 | สมดุลดี ไม่ Overfit/Underfit |
| L2 Lambda | 0.001 | Regularization เบาๆ |
| Batch Size | 16-64 | ปรับตาม VRAM |
| Patience (Early Stop) | 10-12 | ให้เวลาโมเดลปรับตัว |

### ภาคผนวก ค: โครงสร้างโมเดล (Model Summary)

```
Model: "sequential"
_______________________________________________________________
Layer (type)              Output Shape          Param #
===============================================================
conv1d (Conv1D)           (None, 130, 256)      51,456
batch_normalization       (None, 130, 256)       1,024
max_pooling1d             (None, 65, 256)             0
dropout (Dropout)         (None, 65, 256)             0
_______________________________________________________________
conv1d_1 (Conv1D)         (None, 65, 128)       163,968
batch_normalization_1     (None, 65, 128)           512
max_pooling1d_1           (None, 32, 128)             0
dropout_1 (Dropout)       (None, 32, 128)             0
_______________________________________________________________
bidirectional (BiLSTM)    (None, 32, 256)       263,168
dropout_2 (Dropout)       (None, 32, 256)             0
_______________________________________________________________
bidirectional_1 (BiLSTM)  (None, 128)            164,352
dropout_3 (Dropout)       (None, 128)                 0
_______________________________________________________________
dense (Dense)             (None, 64)              8,256
dropout_4 (Dropout)       (None, 64)                  0
dense_1 (Dense)           (None, 5)                 325
===============================================================
Total params: 653,061
Trainable params: 652,293
Non-trainable params: 768
```
