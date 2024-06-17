# ECG Classification Using Transformer Networks

This repository contains code for classifying ECG signals using Transformer networks. The project involves data preprocessing, model creation, and training to accurately classify ECG signals into different categories.

## Table of Contents
- [About the Project](#about-the-project)
- [Notebooks](#notebooks)
- [Dataset Links](#dataset-links)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## About the Project

This project aims to use the power of Transformer networks to classify ECG signals. The repository includes Jupyter notebooks for data preprocessing, model creation, and training. The ECG signals are processed into windows and R-peak masks to enhance the model's performance.

## Notebooks

### 1. incart_ecg_splitter.ipynb
- **Description**: Processes the St Petersburg INCART 12-lead Arrhythmia Database, splitting ECG records into segments.
- **Dataset Source**: [St. Petersburg INCART Database](https://physionet.org/content/incartdb/1.0.0/)

### 2. ptb_ecg_splitter.ipynb
- **Description**: Processes the PTB Diagnostic ECG Database, splitting records based on various cardiac conditions.
- **Dataset Source**: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)

### 3. ptbxl_ecg_filter.ipynb
- **Description**: Processes the PTB-XL ECG dataset, handling preprocessing and segmentation for model training.
- **Dataset Source**: [PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)

### 4. ecg_transformer_window_signals.ipynb
- **Description**: Creates and trains a Transformer model for ECG signal classification. It preprocesses signals by generating windows and R-peak masks, and uses an enhanced positional encoding layer to amplify R-peak points, helping the model focus on important features.
- **Key Steps**:
  1. **Data Preprocessing**:
     - Download datasets from the previous notebooks.
     - Organize signals into folders based on their classes.
     - Resample signals to 100 Hz if necessary.
     - Detect R-peak points using the biosppy library.
     - Generate windows and R-peak masks from the processed signals.
  2. **Model Creation**:
     - **Enhanced Positional Encoding**: Amplifies R-peak points to help the model focus on significant features.
     - **Model Architecture**: Uses Transformer blocks, 1D CNN layers for embedding, and bidirectional LSTM layers for sequence learning.
     - **Model Compilation**: Compiled with the Adam optimizer (learning rate: 0.001) and Sparse Categorical Crossentropy loss function.
  3. **Training and Evaluation**: Trains and evaluates the model on preprocessed ECG data.
- **Output**: A trained Transformer model for ECG classification.

## Dataset Links

- **PTB-XL 1.0.3**: [PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)
- **PTB**: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)
- **St Petersburg INCART 12-lead Arrhythmia Database**: [St. Petersburg INCART Database](https://physionet.org/content/incartdb/1.0.0/)

## Requirements

Ensure you have the following dependencies installed:

- tensorflow==2.15.0
- wfdb==4.1.2
- numpy==1.26.4
- scipy==1.13.1
- biosppy==2.2.2
- matplotlib==3.9.0

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Installation

Clone this repository:

```bash
git clone https://github.com/eemreguven/ECG-Classification-Using-Transformer-Networks.git
cd ECG-Classification-Using-Transformer-Networks
```
## Usage
Run the Jupyter notebooks in the following order for data preprocessing and model training:

incart_ecg_splitter.ipynb,
ptb_ecg_splitter.ipynb,
ptbxl_ecg_process.ipynb,
ecg_transformer_window_signals.ipynb

## Contact

For any inquiries or support, please contact [e.emreguven@outlook.com].
