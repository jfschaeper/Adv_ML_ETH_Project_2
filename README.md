# ECG Signal Classification

This project classifies ECG (electrocardiogram) signals into different heartbeat types using deep learning, classical statistical models, and signal processing techniques.

## Project Structure

- `combine_normal_beats_CNN.ipynb` – Final working notebook. Uses CNNs on extracted heartbeats (QRST complexes).
- `LSTM.ipynb` – Alternative approach using raw time series and LSTM models (less effective).
- `just_run_ols.py` – Classical baseline using engineered features and regression models.
- `functions.py` – Helper functions for signal filtering, feature engineering, and model preprocessing.

---

## Approach

### 1. Signal Preprocessing

- ECG signals are filtered using [`biosppy`](https://github.com/PIA-Group/BioSPPy) and `heartpy`.
- R-peaks are detected using several ECG segmenters (`engzee`, `ssf`, `hamilton`, etc.).
- From these R-peaks, heartbeats (QRST segments) are extracted.

### 2. Feature Extraction

We engineer features in three domains:

#### a. Time-Domain Features
- Number of R/Q/S/T peaks
- Ratios between peak amplitudes (clinically relevant markers)
- Variance, skewness, and kurtosis of the raw signal

#### b. Frequency-Domain Features
- Spectral energy in specific bands using Welch's method

#### c. AR Model Features
- Coefficients and residual variance from an AR(15) time series model

### 3. Modeling

#### CNN (Final model)
- Trains a convolutional neural network on extracted heartbeats
- Predictions are made at the beat level, then aggregated to the patient level using majority vote

#### Tried but not used:
- LSTM model on padded raw ECG sequences
- Classical models using OLS regression with hand-crafted features

---

## Results

- The CNN model trained on individual heartbeats performed best, offering accurate classification and robustness to noisy signals.
- Aggregating beat-level predictions improves overall performance.
