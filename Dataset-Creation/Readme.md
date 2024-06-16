# ECG Data Preparation Notebooks

This directory contains three Jupyter notebooks used for preparing ECG datasets from various sources. The sources include PTB, INCART, and PTB-XL datasets. Below is a summary of each notebook and the datasets they use.

## Notebooks

### 1. incart_ecg_splitter.ipynb
- **Description**: This notebook processes the St Petersburg INCART 12-lead Arrhythmia Database. It splits the ECG records into individual segments suitable for training machine learning models.
- **Dataset Source**: [St. Petersburg INCART Database](https://physionet.org/content/incartdb/1.0.0/)

### 2. ptb_ecg_splitter.ipynb
- **Description**: This notebook processes the PTB Diagnostic ECG Database. It splits the ECG records into individual segments, focusing on various cardiac conditions.
- **Dataset Source**: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)

### 3. ptbxl_ecg_process.ipynb
- **Description**: This notebook processes the PTB-XL ECG dataset. It handles the preprocessing and segmentation of ECG signals to prepare them for model training and evaluation.
- **Dataset Source**: [PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)

## Dataset Links

- **PTB-XL 1.0.3**: [PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)
- **PTB**: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)
- **St Petersburg INCART 12-lead Arrhythmia Database**: [St. Petersburg INCART Database](https://physionet.org/content/incartdb/1.0.0/)

## Requirements

Ensure you have the following dependencies installed:

- jupyter
- numpy
- pandas
- scipy
- matplotlib
- wfdb
- biosppy

You can install the required packages using the following command:

````bash
pip install -r requirements.txt
````

## Usage

Each notebook is designed to be run in a Jupyter Notebook environment. Before running the notebooks, ensure you have all the required dependencies installed. The datasets should be downloaded from their respective sources and placed in the appropriate directories as specified in the notebooks.

## Contact

For any inquiries or support, please contact [e.emreguven@outlook.com].
