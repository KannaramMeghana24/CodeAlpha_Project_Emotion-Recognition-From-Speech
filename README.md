# Emotion Recognition from Speech

## Overview

This project focuses on recognizing emotions from speech using machine learning. The model processes audio signals and classifies them into different emotional categories, such as happy, sad, angry, neutral, fearful, and surprised. This can be useful in applications like virtual assistants, customer service analysis, mental health monitoring, and human-computer interaction.

## Features

- Extracts features from speech signals (MFCC, Chroma, Mel Spectrogram, Zero Crossing Rate, Spectral Contrast, etc.).
- Uses a deep learning model for emotion classification.
- Supports various audio formats for input (.wav, .mp3, etc.).
- Can be integrated into real-time applications.
- Provides visualization tools for audio features.
- Supports model evaluation and fine-tuning options.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/emotion-recognition-speech.git
   cd emotion-recognition-speech
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv env
   source env/bin/activate   # On Windows use: env\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter notebook:
   ```sh
   jupyter notebook
   ```
2. Open `Emotion Recognition From Speech.ipynb` and execute the cells.
3. To predict emotions from an audio file, use:
   ```sh
   python predict.py --file path/to/audio.wav
   ```
4. To train the model with a custom dataset:
   ```sh
   python train.py --dataset path/to/dataset
   ```

## Dataset

This project uses the **RAVDESS** dataset, which contains emotional speech recordings. Other datasets like **TESS**, **CREMA-D**, and **SAVEE** can also be used. The dataset includes recordings with different emotional tones to help train the model.

## Model Details

- Extracts audio features using **Librosa**.
- Uses **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory (LSTMs)**, or **Transformers** for classification.
- Implemented with **TensorFlow/Keras** and **PyTorch**.
- Supports transfer learning for improving performance with pre-trained models.

## Results

- Achieved an accuracy of \~85% on the test set.
- Model generalizes well to different voices and emotions.
- Confusion matrix and classification reports are provided for performance evaluation.

## Visualization

- Includes tools for visualizing Mel Spectrograms, MFCCs, and waveform analysis.
- Displays confusion matrices to evaluate model performance.

