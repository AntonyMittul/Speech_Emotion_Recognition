# 🎙 Speech Emotion Recognition (SER)

This project implements a **Speech Emotion Recognition** system that detects human emotions from speech using **LSTM neural networks**.  
It uses the **Toronto Emotional Speech Set (TESS)** dataset and advanced audio feature extraction to classify speech into seven emotion categories.

---

## 📌 Objective
The goal is to make machines understand and classify emotions from voice, enabling use cases in:
- **Customer Support** – Detect stress or anger during calls.
- **Healthcare** – Monitor patient emotional well-being.
- **Virtual Assistants** – Enhance human-computer interaction.

---

## 🗂 Dataset
- **Source**: [TESS Toronto Emotional Speech Set (Kaggle)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)  
- **Total Samples**: ~2800 audio files.  
- **Emotions**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Pleasant Surprise
  - Sad

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Loaded `.wav` audio files and extracted labels from filenames.
- Standardized sampling rates and normalized signals.
- Visualized **waveforms** and **spectrograms** for different emotions.

### 2️⃣ Feature Extraction
- Computed **Mel Frequency Cepstral Coefficients (MFCCs)** as the main audio features.
- Used 40 MFCCs per sample with statistical aggregation.

### 3️⃣ Model Development
- **Architecture**:
  - **LSTM** layer (256 units) for sequential audio learning.
  - Dense layers (128, 64 units) with ReLU activation.
  - Dropout layers (0.5) for regularization.
  - Softmax output layer (7 neurons for 7 emotions).
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy

### 4️⃣ Training & Evaluation
- Data split into training, validation, and testing sets.
- **Epochs**: 30  
- **Batch Size**: 64  
- **Accuracy**:
  - **Training Accuracy**: ~99%
  - **Validation Accuracy**: ~96%
- Plotted accuracy and loss curves for performance analysis.

---

## 📊 Results
- High validation accuracy (~96%) across all emotion classes.
- Stable training with no significant overfitting.
- Confusion matrix analysis shows strong classification performance.

---

## 🚀 Technologies Used
- **Python**
- **Librosa** – Audio processing
- **NumPy & Pandas** – Data handling
- **Matplotlib & Seaborn** – Visualization
- **TensorFlow/Keras** – Model building & training

---

## 📌 Future Enhancements
- Real-time emotion detection from live audio streams.
- Experiment with **CNN-based spectrogram classification**.
- Extend to multilingual datasets for broader coverage.

---
