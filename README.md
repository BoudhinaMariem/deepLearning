# 🎙️ Audio Emotion Classification – CREMA-D Dataset

## 📌 About the Project
This project aims to classify human emotions from audio recordings, focusing on two categories:
- 😄 Happy
- 😢 Sad

We convert each audio file into a **Mel-Spectrogram** and use Deep Learning models (CNN, ResNet50, and an Encoder–Decoder Autoencoder) to learn emotion-specific acoustic patterns.

---

## 📂 About the Dataset (CREMA-D)

**Dataset link:**  
https://www.kaggle.com/datasets/ejlok1/cremad

### 📌 Context  
The CREMA-D dataset is widely used for speech emotion recognition because it contains **a large variety of speakers**, reducing information leakage and improving generalization.  
Unlike many audio datasets with few speakers, CREMA-D includes many actors, making it less prone to overfitting.

### 📌 Content  
- **7,442** original audio clips  
- **91 actors** (48 males, 43 females)  
- Age range: **20–74 years**  
- Ethnic diversity: African American, Asian, Caucasian, Hispanic, Unspecified  
- Actors pronounce **12 different sentences**  
- Each sentence is performed with one of **6 emotions**:  
  **Anger, Disgust, Fear, Happy, Neutral, Sad**  
- 4 emotion intensity levels: Low, Medium, High, Unspecified  
- Audio format: `.wav`

---

## 🛠️ Preprocessing
- Load audio with **Librosa**
- Convert waveform → **Mel-Spectrogram**
- Apply normalization and padding to obtain a fixed-size spectrogram
- Split dataset into Train / Validation / Test

---

## 🤖 Models Used

### 1️⃣ **Autoencoder (Encoder–Decoder)**
- Convolutional Encoder to learn compressed audio representations  
- Decoder to reconstruct spectrograms  
- Helps to improve feature extraction before classification

### 2️⃣ **CNN From Scratch**
- Custom Convolutional Neural Network  
- Baseline model trained directly on Mel-Spectrograms

### 3️⃣ **ResNet50 (Transfer Learning)**
- Pretrained on ImageNet  
- Used as a feature extractor on spectrogram images  
- Custom classification head added for emotion recognition

### 4️⃣ **Fine-Tuned ResNet50**
- Unfreeze deeper layers  
- Lower learning rate for fine adjustment  
- Best performing model

---

## 📊 Results
| Model | Accuracy |
|-------|----------|
| Autoencoder features + classifier | ~85% |
| CNN from scratch | ~85–88% |
| ResNet50 | ~90–92% |
| **Fine-tuned ResNet50** | **~94–96%** |

---

## ▶️ Run the Notebook
```bash
pip install librosa tensorflow numpy matplotlib seaborn scikit-learn
jupyter notebook audio_deep.ipynb
