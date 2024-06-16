import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wfdb
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import resample
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

@tf.keras.utils.register_keras_serializable()
class EnhancedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, r_peak_factor=10, **kwargs):
        super(EnhancedPositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.r_peak_factor = r_peak_factor

    def get_angles(self, pos, i, d_model):
        angles = pos / tf.math.pow(10000.0, (2 * (tf.cast(i, tf.float32) // 2)) / tf.cast(d_model, tf.float32))
        return angles

    def call(self, inputs, r_peaks):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]

        angle_rads = self.get_angles(
            tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32),
            tf.range(feature_dim)[tf.newaxis, :],
            feature_dim
        )

        angle_rads = tf.where(tf.range(feature_dim) % 2 == 0,
                              tf.math.sin(angle_rads),
                              tf.math.cos(angle_rads))
        pos_encoding = angle_rads[tf.newaxis, :, :]
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])

        r_peaks_amplified = r_peaks * self.r_peak_factor

        return inputs + tf.cast(pos_encoding, tf.float32) + r_peaks_amplified

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "r_peak_factor": self.r_peak_factor
        })
        return config

def load_model(model_path):
    custom_objects = {
        'EnhancedPositionalEncoding': EnhancedPositionalEncoding
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Class names for the model's predictions
class_names = ['NORM', 'MI', 'HYP']
class_names_full = ['Healthy (NORM)', 'Myocardial Infarction (MI)', 'Hypertrophy (HYP)']

# Function to generate windows and R-peak masks from the signal
def signal_generator(signal, r_peaks):
    signal_length = signal.shape[0]
    window_size = 150  # 10 seconds at 100 Hz
    stride = 75  # 50% overlap
    num_channels = signal.shape[1]
    num_windows = (signal_length - window_size) // stride + 1
    windows = []
    r_peak_windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window = signal[start:end].copy()
        windows.append(window)

        window_r_peaks = np.zeros(window_size)
        r_peak_positions = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
        window_r_peaks[r_peak_positions] = 1
        r_peak_windows.append(np.tile(window_r_peaks, (num_channels, 1)).T)

    arr_windows = np.array(windows)
    arr_r_peak_windows = np.array(r_peak_windows)

    return arr_windows, arr_r_peak_windows

def process_signal(signal, sampling_rate):
    # Her kanalı sırayla yeniden örnekleme
    if sampling_rate != 100:
        signal_resampled = []
        for channel in range(signal.shape[1]):
            resampled_channel = resample(signal[:, channel], int(100 * (signal.shape[0] / sampling_rate)))
            signal_resampled.append(resampled_channel)
        signal = np.array(signal_resampled).T
        sampling_rate = 100
    else:
        signal_resampled = signal
    
    # İlk kanalı kullanarak R-peak'leri tespit etme
    analysis = ecg.ecg(signal=signal_resampled[:, 0], sampling_rate=sampling_rate, show=False)
    r_peaks = analysis['rpeaks']

    return signal_resampled, r_peaks

# Function to predict using the model
def predict(signal, r_peaks):
    windows, r_peak_windows = signal_generator(signal, r_peaks)
    reshaped_windows = np.expand_dims(windows, axis=0)
    reshaped_r_peak_windows = np.expand_dims(r_peak_windows, axis=0)
    predictions = model.predict([reshaped_windows, reshaped_r_peak_windows])
    return predictions

# Function to update status messages in the GUI
def update_status(message):
    status_label.config(text=message)
    root.update_idletasks()

# Function to handle model file selection
def on_model_select():
    global model
    model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5")])
    if model_path:
        try:
            update_status("Loading model...")
            model = load_model(model_path)
            update_status("Model loaded successfully. Now select the ECG file.")
            select_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Function to plot the signal in the main window
def plot_signals(signal, sampling_rate):
    time = np.arange(len(signal)) / sampling_rate

    # Create a matplotlib figure with adjusted size
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, signal)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('ECG Signal')
    ax.grid(True)
    
    return fig

# Function to handle ECG file selection and processing
def on_file_select():
    file_path = filedialog.askopenfilename(filetypes=[("WFDB header files", "*.hea")])
    if file_path:
        try:
            # Display the file name
            file_name = os.path.basename(file_path)
            file_label.config(text=f"Selected file: {file_name}")
            
            record_path = file_path[:-4]
            signals, fields = wfdb.rdsamp(record_path)
            sampling_rate = fields['fs']
            if signals.shape[0] < sampling_rate*10:
                messagebox.showerror("Error", "Signal is too short. Please select a signal of at least 10 seconds.")
                return
            if signals.shape[0] > sampling_rate*10:
                signals = signals[:sampling_rate*10]

            # Process the signal
            signal, r_peaks = process_signal(signals, sampling_rate)

            # Plot the signal in the main window
            for widget in plot_frame.winfo_children():
                widget.destroy()  # Önceki grafiği temizleme

            fig = plot_signals(signal, sampling_rate)
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update status and make predictions
            update_status("Processing...")
            predictions = predict(signal, r_peaks)
            update_status("Processing complete")

            # Display predictions
            result_text = ""
            for i, class_name in enumerate(class_names):
                result_text += "{}: {:.2f}%\n".format(class_name, predictions[0][i] * 100)
            predicted_class = np.argmax(predictions, axis=-1)[0]
            diagnosis_text = "Diagnosis: {}".format(class_names_full[predicted_class])
            result_label.config(text=result_text)
            diagnosis_label.config(text=diagnosis_text)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# GUI setup
root = tk.Tk()
root.title("ECG Signal Classifier")
root.state("zoomed")  # Fullscreen mode

# Main frame for controls
frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=20)

# Buttons to select the model file and the ECG file, placed side by side
model_button = tk.Button(frame, text="Select Model File", command=on_model_select, width=20, height=2)
model_button.grid(row=0, column=0, padx=10, pady=10)

select_button = tk.Button(frame, text="Select ECG File", command=on_file_select, width=20, height=2, state="disabled")
select_button.grid(row=0, column=1, padx=10, pady=10)

# Label to display the selected file name
file_label = tk.Label(frame, text="", justify="left", font=("Arial", 12), fg="blue")
file_label.grid(row=1, column=0, columnspan=2, pady=10)

# Status label for displaying messages
status_label = tk.Label(frame, text="", justify="left", font=("Arial", 13), fg="green")
status_label.grid(row=2, column=0, columnspan=2, pady=10)

# Result label for displaying predictions
result_label = tk.Label(frame, text="", justify="left", font=("Arial", 13), fg="black")
result_label.grid(row=3, column=0, columnspan=2, pady=10)

# Diagnosis label for displaying the main prediction
diagnosis_label = tk.Label(frame, text="", justify="left", font=("Arial", 16, "bold"), fg="blue")
diagnosis_label.grid(row=4, column=0, columnspan=2, pady=10)

# Frame for plotting the signal
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Run the Tkinter event loop
root.mainloop()
