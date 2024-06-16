# ECG Transformer Model Creation Notebook

This directory contains a Jupyter notebook used for creating a Transformer model for ECG signal classification. The notebook processes ECG signals, generates windows, and applies a Transformer-based neural network to classify the signals.

## Notebook

### ecg_transformer_window_signals.ipynb
- **Description**: This notebook focuses on creating and training a Transformer model for ECG signal classification. It preprocesses the ECG signals by generating windows and R-peak masks, then applies a Transformer-based neural network to classify the signals into different categories.
- **Key Steps**:
  1. **Data Preprocessing**:
     - Download the datasets prepared by the previous three notebooks (`incart_ecg_splitter.ipynb`, `ptb_ecg_splitter.ipynb`, `ptbxl_ecg_process.ipynb`).
     - Organize the signals into folders based on their classes.
     - Resample the signals to 100 Hz if necessary.
     - Detect R-peak points using the biosppy library.
     - Generate windows and R-peak masks from the processed signals.
  2. **Model Creation**:
     - **Enhanced Positional Encoding**:
       - The model uses an enhanced positional encoding layer that amplifies R-peak points to help the model focus on important features in the ECG signal.
     - **Model Architecture**:
       - The model uses a series of Transformer blocks, each consisting of multi-head self-attention and feed-forward neural network layers.
       - The input to the model includes windows of ECG signals and corresponding R-peak masks.
       - The model uses 1D CNN layers to create embeddings for each window.
       - Positional encoding is applied to these embeddings.
       - Multiple Transformer blocks process the sequence data.
       - The output of the Transformer blocks is averaged and flattened.
       - Dense layers are used for final classification.
     - **Model Compilation**:
       - The model is compiled using the Adam optimizer with a learning rate of 0.001.
       - The loss function used is Sparse Categorical Crossentropy.
       - The model is evaluated based on accuracy.
  3. **Training and Evaluation**:
     - Train the model on the preprocessed ECG data.
     - Evaluate the model performance using appropriate metrics.
- **Output**: The trained Transformer model capable of classifying ECG signals into predefined categories.

## Usage

The notebook is designed to be run in a Jupyter Notebook environment. Before running the notebook, ensure you have all the required dependencies installed. The dataset should be prepared and placed in the appropriate directory as specified in the notebook.

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

## Contact

For any inquiries or support, please contact [e.emreguven@outlook.com].
