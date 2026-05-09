# ECG Arrhythmia Classification Under Exam Stress

## Overview

This project analyzes electrocardiogram (ECG) signals from the **MIT-BIH Noise Stress Test Database** to classify different types of cardiac arrhythmias. Using machine learning and deep learning approaches, we develop models to automatically detect and categorize abnormal heart rhythms, with a focus on stress-induced cardiac patterns.

## Features

- **ECG Signal Processing**: Load, visualize, and preprocess raw ECG data
- **Beat Detection & Segmentation**: Automatic extraction of individual heartbeats with context windows
- **Feature Engineering**: Statistical features (mean, std, max, min) extraction from beat segments
- **Multi-Model Approach**:
  - **Random Forest Classifier**: Baseline machine learning model
  - **1D CNN (Deep Learning)**: Advanced neural network for time-series signal classification
- **Overfitting Prevention**: Dropout regularization and early stopping mechanisms
- **Model Evaluation**: Comprehensive metrics including confusion matrices and classification reports

## Dataset

**MIT-BIH Noise Stress Test Database (nstdb)**

- **Description**: High-quality ECG signals recorded during stress conditions
- **Records**: Multiple 10-minute ECG recordings with various noise levels
- **Sampling Rate**: 360 Hz
- **Lead**: Single-lead recordings
- **Annotations**: Expert-annotated beat labels

### Beat Classifications

The model classifies heartbeats into five categories:

| Label | Beat Type |
|-------|-----------|
| **N** | Normal beat |
| **V** | Premature ventricular contraction (PVC) |
| **A** | Atrial premature beat (APB) |
| **L** | Left bundle branch block beat |
| **R** | Right bundle branch block beat |

## Installation

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow wfdb
```

### Environment Setup

1. Clone or download the project
2. Place the MIT-BIH database in the project directory
3. Update file paths in the notebook to match your local setup

## Project Structure

```
project-root/
│
├── Exam Stress Project.ipynb          # Main analysis notebook
├── mit-bih-noise-stress-test-database # ECG dataset directory
│   ├── 118e_6.*                       # ECG record files
│   ├── 119e_6.*                       # (Multiple records available)
│   └── ...
└── README.md                          # This file
```

## Usage

### 1. Loading ECG Records

```python
import wfdb

path = r"path/to/mit-bih-noise-stress-test-database/118e_6"
record = wfdb.rdrecord(path)
annotation = wfdb.rdann(path, "atr")
```

### 2. Feature Extraction

```python
# Extract beat segments with ±150 sample window
window = 150
beats_data = []

for i, s in enumerate(annotation.sample):
    if s - window > 0 and s + window < len(record.p_signal):
        segment = record.p_signal[s-window:s+window, 0]
        beats_data.append((segment, annotation.symbol[i]))
```

### 3. Training Models

**Random Forest Approach:**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
score = rf_model.score(X_test, y_test)
```

**Deep Learning (CNN) Approach:**
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv1D(32, 5, activation='relu', input_shape=(300, 1)),
    layers.MaxPooling1D(2),
    layers.Dropout(0.2),
    # ... additional layers
    layers.Dense(len(classes), activation='softmax')
])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)
```

## Model Architecture

### 1D CNN Model

```
Input: (batch_size, 300, 1) - 300 samples per beat segment

Conv1D (32 filters, 5x1)  → ReLU
MaxPooling1D (2x1)        → 50% reduction
Dropout (0.2)

Conv1D (64 filters, 5x1)  → ReLU
MaxPooling1D (2x1)        → 50% reduction
Dropout (0.2)

Conv1D (128 filters, 3x1) → ReLU
MaxPooling1D (2x1)        → 50% reduction

Flatten

Dense (64)                → ReLU
Dropout (0.5)             → Overfitting prevention

Output: Dense (5, softmax) → Classification probabilities
```

## Results & Performance

The model provides:

- **Confusion Matrix Visualization**: Shows classification accuracy per beat type
- **Classification Report**: Precision, Recall, and F1-Score metrics
- **Training History**: Loss and accuracy curves over epochs
- **Test Accuracy**: Overall performance on held-out test set

### Key Features:

- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Stratified Split**: Maintains class distribution in train/test sets
- **Label Encoding**: Automatic conversion of categorical labels to numeric format

## Dependencies

```
numpy              - Numerical computing
pandas             - Data manipulation
scikit-learn       - Machine learning algorithms
matplotlib         - Visualization
seaborn            - Statistical plots
tensorflow/keras   - Deep learning framework
wfdb               - ECG database utilities
```

## Visualization Outputs

- ECG signal waveforms with beat annotations
- Confusion matrices for model evaluation
- Training/validation accuracy and loss curves
- Feature distribution analysis

## Technical Highlights

✓ **Data Augmentation via Stress Conditions**: Dataset includes noise-injected variants
✓ **Robust Preprocessing**: Proper windowing and normalization
✓ **Multiple Validation Approaches**: Train/validation/test split with stratification
✓ **Regularization Techniques**: Dropout layers and early stopping
✓ **Professional Evaluation**: Multi-metric assessment of model performance

## Future Enhancements

- Multi-lead ECG analysis
- Ensemble methods combining RF and CNN
- Real-time ECG monitoring application
- Extended beat classification categories
- Temporal sequence modeling (LSTM/GRU)

## References

- MIT-BIH Arrhythmia Database
- WFDB Toolbox Documentation
- Cardiac Arrhythmia Classification Literature

## License

This project uses the MIT-BIH Noise Stress Test Database. Please refer to the original database documentation for usage terms.
Author/Sana Elbakry
## Contact & Support

For questions or improvements, please refer to the project notebook for implementation details.

---

**Project Status**: Active  
**Last Updated**: 2026  
**Python Version**: 3.8+
