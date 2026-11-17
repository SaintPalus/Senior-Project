"""
Speech Emotion Recognition — Hugging Face Spaces
รองรับ 5 ภาษา: จีน ญี่ปุ่น เกาหลี ไทย อังกฤษ
"""

import os
import sys
import pickle
import warnings
import tempfile
import numpy as np

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LANGUAGES  = ['Chinese', 'Japan', 'Korean', 'Thai', 'English']
LANG_LABEL = {'Chinese':'Chinese (จีน)', 'Japan':'Japanese (ญี่ปุ่น)',
               'Korean':'Korean (เกาหลี)', 'Thai':'Thai (ไทย)', 'English':'English (อังกฤษ)'}
LANG_AUTO  = 'Auto-detect (ตรวจจับอัตโนมัติ)'

EMO_EMOJI  = {'Angry':'😡','Happy':'😄','Sad':'😢','Neutral':'😐',
               'Fear':'😨','Surprise':'😲','Disgust':'🤢'}
EMO_COLOR  = {'Angry':'#E74C3C','Happy':'#F39C12','Sad':'#3498DB',
               'Neutral':'#95A5A6','Fear':'#8E44AD','Surprise':'#1ABC9C','Disgust':'#27AE60'}

N_MELS = 128
MAX_T  = 130
SR     = 22050


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.relu(x + self.block(x))

class EmotionResNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.layer1 = nn.Sequential(ResBlock(32),  nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),  nn.BatchNorm2d(64),  nn.ReLU(True))
        self.layer2 = nn.Sequential(ResBlock(64),  nn.Conv2d(64,128,3,stride=2,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.layer3 = nn.Sequential(ResBlock(128), nn.Conv2d(128,256,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(256), nn.ReLU(True))
        self.pool   = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4,256), nn.ReLU(True), nn.Dropout(0.4), nn.Linear(256,n))
    def forward(self, x): return self.classifier(self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))))


_cache = {}

def load_model(lang):
    if lang in _cache:
        return _cache[lang]
    pt  = os.path.join(MODEL_DIR, f"{lang}_model.pt")
    enc = os.path.join(MODEL_DIR, f"{lang}_label_encoder.pkl")
    if not os.path.exists(pt):
        return None, None, None
    with open(enc, 'rb') as f:
        le = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(pt, map_location=device, weights_only=False)
    model  = EmotionResNet(ckpt['n_classes']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    _cache[lang] = (model, le, device)
    return model, le, device


def load_lang_detector():
    path = os.path.join(MODEL_DIR, "lang_detector.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def audio_to_mel(y, sr):
    import librosa
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] >= MAX_T:
        mel_db = mel_db[:, :MAX_T]
    else:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_T-mel_db.shape[1])),
                        mode='constant', constant_values=mel_db.min())
    return mel_db.astype(np.float32)


def predict_emotion(audio_path, lang_choice):
    import librosa
    try:
        y, sr = librosa.load(audio_path, sr=SR)
    except Exception as e:
        return None, None, None, str(e)

    if np.abs(y).max() < 0.005:
        return None, None, None, "เสียงเบาเกินไป — ลองพูดดังขึ้น"

    mel = audio_to_mel(y, sr)

    if lang_choice == LANG_AUTO:
        det = load_lang_detector()
        if det is None:
            return None, None, None, "ไม่พบ lang_detector.pkl"
        lang = det.predict(mel.flatten().reshape(1, -1))[0]
    else:
        lang = next((k for k, v in LANG_LABEL.items() if v == lang_choice), lang_choice)

    model, le, device = load_model(lang)
    if model is None:
        return None, None, None, f"ไม่พบ model สำหรับ {lang}"

    x = torch.tensor(mel[np.newaxis, np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

    idx      = int(np.argmax(probs))
    emotion  = le.classes_[idx]
    conf     = probs[idx] * 100
    all_prob = {le.classes_[i]: float(probs[i]*100) for i in range(len(probs))}
    return emotion, conf, all_prob, lang


# ── Gradio UI ──────────────────────────────────
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

history_data  = []
lang_choices  = [LANG_AUTO] + [LANG_LABEL[l] for l in LANGUAGES]

print("โหลด models ...")
for lang in LANGUAGES:
    m, _, _ = load_model(lang)
    if m: print(f"  ✓ {lang}")


def process_audio(audio_input, lang_choice):
    if audio_input is None:
        return None, "กรุณาอัดเสียงหรืออัปโหลดไฟล์", None

    if isinstance(audio_input, tuple):
        sr_in, arr = audio_input
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if np.abs(arr).max() > 1:
            arr = arr / 32768.0
        peak = np.abs(arr).max()
        if peak > 0:
            arr = arr / peak * 0.9
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, arr, sr_in)
        audio_path = tmp.name
    else:
        audio_path = audio_input

    emotion, conf, all_prob, lang_or_err = predict_emotion(audio_path, lang_choice)

    if emotion is None:
        return None, f"❌ {lang_or_err}", None

    lang = lang_or_err

    import librosa, librosa.display
    y, sr = librosa.load(audio_path, sr=SR)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    color = EMO_COLOR.get(emotion, '#2C3E50')

    ax1 = fig.add_subplot(gs[0, 0])
    t   = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(t, y, color=color, linewidth=0.6, alpha=0.85)
    ax1.set_title('Waveform', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, t[-1])
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=512)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                              sr=sr, hop_length=512, x_axis='time', y_axis='mel',
                              ax=ax2, cmap='magma')
    ax2.set_title('Mel-Spectrogram', fontsize=11, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :])
    emotions_sorted = sorted(all_prob, key=lambda e: -all_prob[e])
    bars = ax3.barh(emotions_sorted,
                    [all_prob[e] for e in emotions_sorted],
                    color=[EMO_COLOR.get(e, '#AAB7B8') for e in emotions_sorted],
                    edgecolor='white', linewidth=0.8, alpha=0.9)
    for bar, emo in zip(bars, emotions_sorted):
        v = all_prob[emo]
        ax3.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=10,
                 fontweight='bold' if emo == emotion else 'normal')
    ax3.set_xlim(0, 115)
    ax3.set_xlabel('Confidence (%)', fontsize=10)
    ax3.set_title('Emotion Confidence', fontsize=11, fontweight='bold')
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    emoji = EMO_EMOJI.get(emotion, '')
    fig.suptitle(f'{LANG_LABEL.get(lang, lang)}  →  {emoji} {emotion}  ({conf:.1f}%)',
                 fontsize=15, fontweight='bold', color=color, y=1.01)
    plt.tight_layout()

    history_data.append({'lang': LANG_LABEL.get(lang, lang), 'emotion': emotion, 'conf': f'{conf:.1f}%'})
    result_text = (
        f"🌐 ภาษา: **{LANG_LABEL.get(lang, lang)}**\n\n"
        f"{emoji} อารมณ์: **{emotion}**\n\n"
        f"📊 Confidence: **{conf:.1f}%**\n\n"
        f"🤖 Model: Mel-Spectrogram ResNet"
    )
    hist_md = "| # | ภาษา | อารมณ์ | Conf |\n|---|------|--------|------|\n"
    for i, h in enumerate(reversed(history_data[-10:]), 1):
        hist_md += f"| {i} | {h['lang']} | {h['emotion']} | {h['conf']} |\n"

    return fig, result_text, hist_md


title_html = """
<div style="text-align:center; padding:20px; background:linear-gradient(135deg,#667eea,#764ba2);
            border-radius:12px; color:white; margin-bottom:20px">
  <h1 style="margin:0; font-size:2em">🎙️ Speech Emotion Recognition</h1>
  <p style="margin:8px 0 0; opacity:0.9; font-size:1.1em">
    รองรับ 5 ภาษา: จีน • ญี่ปุ่น • เกาหลี • ไทย • อังกฤษ
  </p>
  <p style="margin:4px 0 0; opacity:0.8; font-size:0.9em">Model: Mel-Spectrogram ResNet | Auto Language Detection</p>
</div>"""

with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
    gr.HTML(title_html)
    with gr.Row():
        with gr.Column(scale=1):
            audio_in  = gr.Audio(sources=["microphone","upload"], type="numpy", label="🎙️ อัดเสียง หรือ อัปโหลดไฟล์")
            lang_sel  = gr.Dropdown(choices=lang_choices, value=LANG_AUTO, label="🌐 เลือกภาษา")
            btn       = gr.Button("🔍 วิเคราะห์อารมณ์", variant="primary", size="lg")
            result_md = gr.Markdown(label="ผลลัพธ์")
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Feature Visualization")
    with gr.Accordion("📋 ประวัติการวิเคราะห์ (10 ล่าสุด)", open=False):
        hist_out = gr.Markdown()
    btn.click(fn=process_audio, inputs=[audio_in, lang_sel], outputs=[plot_out, result_md, hist_out])

demo.launch()
