# SPEECH EMOTION RECOGNITION

##🎙 Speech Emotion Recognition (SER)
This project implements a Speech Emotion Recognition system capable of detecting human emotions from speech using LSTM neural networks. It leverages the Toronto Emotional Speech Set (TESS) dataset and advanced audio feature extraction to classify speech into seven emotion categories.

##📌 Objective
To enable machines to recognize and classify emotions from audio, enabling use cases in:

Customer Support – Detect stress or anger during calls.

Healthcare – Monitor patient emotional well-being.

Virtual Assistants – Enhance human-computer interaction.

##🗂 Dataset
Source: TESS Toronto Emotional Speech Set (Kaggle)

Total Samples: ~2800 audio files.

Emotions:

Angry

Disgust

Fear

Happy

Neutral

Pleasant Surprise

Sad

##⚙️ Methodology
###1️⃣ Data Preprocessing
Loaded .wav audio files and extracted labels from filenames.

Standardized sampling rates and normalized signals.

Visualized waveforms and spectrograms for emotion analysis.

###2️⃣ Feature Extraction
Computed Mel Frequency Cepstral Coefficients (MFCCs) as key audio features.

Used 40 MFCCs per sample with statistical aggregation.

###3️⃣ Model Development
Architecture:

LSTM layer (256 units) for sequential audio pattern learning.

Dense layers (128, 64 units) with ReLU activation.

Dropout layers (0.5) for regularization.

Softmax output layer (7 neurons for 7 emotions).

Optimizer: Adam

Loss: Categorical Crossentropy

###4️⃣ Training & Evaluation
Training set, validation set split for performance measurement.

Epochs: 30

Batch Size: 64

Accuracy:

Training Accuracy: ~99%

Validation Accuracy: ~96%

Plotted accuracy and loss curves for model performance analysis.

###📊 Results
High validation accuracy (~96%) across all emotion classes.

Stable training without significant overfitting.

Confusion matrix shows strong classification performance across all categories.

##🚀 Technologies Used
Python

Librosa – Audio processing

NumPy & Pandas – Data handling

Matplotlib & Seaborn – Visualization

TensorFlow/Keras – Model building and training

##📌 Future Enhancements
Deploy model for real-time emotion detection from live audio.

Experiment with CNN-based spectrogram classification.

Extend to multilingual datasets for broader coverage.
