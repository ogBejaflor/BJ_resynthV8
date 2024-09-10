import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import librosa
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from sklearn.preprocessing import MinMaxScaler
import random
import threading
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from audio_analysis import extract_features


class BJSampleResynthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BJ Sample Resynth")
        self.samples = []
        self.target_audio = None
        self.window_size = 2048
        self.hop_length = self.window_size // 2  # Default to 50% overlap
        self.dataset_features = None
        self.target_features = None
        self.resynthesized_audio = None
        self.sample_rate = None  # Set sample rate based on the first loaded sample

        # Create UI elements

        # Reset buttons
        reset_frame = tk.Frame(root)
        reset_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.randomize_weights_button = tk.Button(reset_frame, text="Randomize Weights", command=self.randomize_weights)
        self.randomize_weights_button.pack(side=tk.LEFT, padx=5)

        self.reset_defaults_button = tk.Button(reset_frame, text="Reset to Default", command=self.reset_to_default)
        self.reset_defaults_button.pack(side=tk.LEFT, padx=5)

        self.reset_analysis_button = tk.Button(reset_frame, text="Reset Analysis", command=self.reset_analysis)
        self.reset_analysis_button.pack(side=tk.LEFT, padx=5)

        self.tutorial_button = tk.Button(reset_frame, text="Tutorial", command=self.show_full_tutorial)
        self.tutorial_button.pack(side=tk.LEFT, padx=5)

        # First column
        column1 = tk.Frame(root)
        column1.pack(side=tk.LEFT, padx=10, pady=10)

        self.load_button = tk.Button(column1, text="Load Samples for Dataset", command=self.load_samples)
        self.load_button.pack(pady=5)

        self.window_size_label = tk.Label(column1, text="Window Size:")
        self.window_size_label.pack(pady=5)

        self.window_size_scale = tk.Scale(column1, from_=256, to_=15000, orient=tk.HORIZONTAL, command=self.update_window_size)
        self.window_size_scale.set(self.window_size)
        self.window_size_scale.pack(pady=5)

        self.window_size_entry = tk.Entry(column1)
        self.window_size_entry.insert(0, str(self.window_size))
        self.window_size_entry.pack(pady=5)

        self.overlap_label = tk.Label(column1, text="Overlap (%):")
        self.overlap_label.pack(pady=5)

        self.overlap_scale = tk.Scale(column1, from_=0, to_=99, orient=tk.HORIZONTAL, command=self.update_overlap)
        self.overlap_scale.set(50)  # Default to 50% overlap
        self.overlap_scale.pack(pady=5)

        self.overlap_entry = tk.Entry(column1)
        self.overlap_entry.insert(0, str(50))  # Default to 50% overlap
        self.overlap_entry.pack(pady=5)

        self.energy_chunk_var = tk.BooleanVar(value=True)  # Energy-Based Chunking ON by default
        self.energy_chunk_check = tk.Checkbutton(column1, text="Energy-Based Chunking", variable=self.energy_chunk_var)
        self.energy_chunk_check.pack(pady=5)

        self.larger_frames_var = tk.BooleanVar(value=True)  # Larger Frames ON by default
        self.larger_frames_check = tk.Checkbutton(column1, text="Larger Frames for Low-Frequency Content", variable=self.larger_frames_var)
        self.larger_frames_check.pack(pady=5)

        self.analyze_button = tk.Button(column1, text="Analyze Samples", command=self.analyze_samples)
        self.analyze_button.pack(pady=5)
        self.analyze_button.config(state=tk.DISABLED)

        self.sort_store_button = tk.Button(column1, text="Sort and Store Dataset", command=self.sort_and_store_dataset)
        self.sort_store_button.pack(pady=5)
        self.sort_store_button.config(state=tk.DISABLED)

        # Second column
        column2 = tk.Frame(root)
        column2.pack(side=tk.LEFT, padx=10, pady=10)

        self.load_target_button = tk.Button(column2, text="Load Target Audio", command=self.load_target_audio)
        self.load_target_button.pack(pady=5)

        self.analyze_target_button = tk.Button(column2, text="Analyze Target File", command=self.analyze_target_file)
        self.analyze_target_button.pack(pady=5)
        self.analyze_target_button.config(state=tk.DISABLED)

        self.sort_store_target_button = tk.Button(column2, text="Sort and Store Target Audio", command=self.sort_and_store_target_audio)
        self.sort_store_target_button.pack(pady=5)
        self.sort_store_target_button.config(state=tk.DISABLED)

        self.crossfade_label = tk.Label(column2, text="Crossfade Length:")
        self.crossfade_label.pack(pady=5)

        self.crossfade_scale = tk.Scale(column2, from_=0, to_=1000, orient=tk.HORIZONTAL, command=self.update_crossfade)
        self.crossfade_scale.set(self.hop_length // 2)  # Default crossfade length
        self.crossfade_scale.pack(pady=5)

        self.crossfade_entry = tk.Entry(column2)
        self.crossfade_entry.insert(0, str(self.hop_length // 2))
        self.crossfade_entry.pack(pady=5)

        self.dynamic_crossfade_var = tk.BooleanVar(value=True)  # Dynamic Crossfade ON by default
        self.dynamic_crossfade_check = tk.Checkbutton(column2, text="Dynamic Crossfade", variable=self.dynamic_crossfade_var)
        self.dynamic_crossfade_check.pack(pady=5)

        self.low_energy_thresh_label = tk.Label(column2, text="Low Energy Threshold:")
        self.low_energy_thresh_label.pack(pady=5)

        self.low_energy_thresh_scale = tk.Scale(column2, from_=0.0, to_=1.0, resolution=0.01, orient=tk.HORIZONTAL)
        self.low_energy_thresh_scale.set(0.1)  # Default low energy threshold
        self.low_energy_thresh_scale.pack(pady=5)

        self.normalize_var = tk.BooleanVar(value=True)  # Normalize ON by default
        self.normalize_check = tk.Checkbutton(column2, text="Normalize Amplitude", variable=self.normalize_var)
        self.normalize_check.pack(pady=5)

        self.pitch_shift_var = tk.BooleanVar()
        self.pitch_shift_check = tk.Checkbutton(column2, text="Pitch Shift", variable=self.pitch_shift_var)
        self.pitch_shift_check.pack(pady=5)

        # Toggle option for DTW
        self.dtw_var = tk.BooleanVar()  # DTW is off by default
        self.dtw_check = tk.Checkbutton(column2, text="DTW for Matching", variable=self.dtw_var)
        self.dtw_check.pack(pady=5)

        self.resynthesize_button = tk.Button(column2, text="Resynthesize Audio", command=self.resynthesize_audio)
        self.resynthesize_button.pack(pady=5)
        self.resynthesize_button.config(state=tk.DISABLED)

        self.save_button = tk.Button(column2, text="Save Resynthesized Audio", command=self.save_resynthesized_audio)
        self.save_button.pack(pady=5)
        self.save_button.config(state=tk.DISABLED)

        self.progress_label = tk.Label(column2, text="")
        self.progress_label.pack(pady=5)

        self.progress = ttk.Progressbar(column2, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=5)

        self.audio_player = tk.Button(column2, text="Play Resynthesized Audio", command=self.play_audio)
        self.audio_player.pack(pady=5)
        self.audio_player.config(state=tk.DISABLED)

        # Weights Section
        weight_column1 = tk.Frame(root)
        weight_column1.pack(side=tk.LEFT, padx=10, pady=10)

        weight_column2 = tk.Frame(root)
        weight_column2.pack(side=tk.LEFT, padx=10, pady=10)

        self.weights = {
            'chroma': tk.DoubleVar(value=1.0),
            'mfcc': tk.DoubleVar(value=1.0),
            'spectral_centroid': tk.DoubleVar(value=1.0),
            'spectral_rolloff': tk.DoubleVar(value=1.0),
            'zero_crossing_rate': tk.DoubleVar(value=1.0),
            'rms': tk.DoubleVar(value=1.0),
            'pitch': tk.DoubleVar(value=1.0),
            'amplitude': tk.DoubleVar(value=1.0),
            'mel_spectrogram': tk.DoubleVar(value=1.0),
            'tempo': tk.DoubleVar(value=1.0),
            'transient': tk.DoubleVar(value=1.0),
            'spectral_flatness': tk.DoubleVar(value=1.0),
        }

        self.weight_labels = {}
        self.weight_sliders = {}

        weight_keys = list(self.weights.keys())
        split_index = len(weight_keys) // 2

        for i, key in enumerate(weight_keys[:split_index]):
            self.weight_labels[key] = tk.Label(weight_column1, text=f"{key.capitalize()} Weight:")
            self.weight_labels[key].pack(pady=5)

            self.weight_sliders[key] = tk.Scale(weight_column1, from_=0.0, to_=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.weights[key])
            self.weight_sliders[key].pack(pady=5)

        for i, key in enumerate(weight_keys[split_index:]):
            self.weight_labels[key] = tk.Label(weight_column2, text=f"{key.capitalize()} Weight:")
            self.weight_labels[key].pack(pady=5)

            self.weight_sliders[key] = tk.Scale(weight_column2, from_=0.0, to_=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.weights[key])
            self.weight_sliders[key].pack(pady=5)

        # Show the brief tutorial on startup
        self.show_brief_tutorial()

    def show_brief_tutorial(self):
        """Display a small window with a brief tutorial when the app starts."""
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("Welcome to BJ Sample Resynth App")
        tutorial_text = """
🎉 Welcome to BJ Sample Resynth App! 🎉

1. 🚀 Load and Analyze Dataset
   - Click "Load Samples for Dataset" to select a folder with audio files.
   - Adjust Window Size and Overlap for perfect analysis.
   - Optional: Enable "Energy-Based Chunking" or "Use Larger Frames for Low-Frequency Content" for advanced tweaks.
   - Click "Analyze Samples" and then "Sort and Store Dataset".

2. 🔍 Load and Analyze Target Audio
   - Click "Load Target Audio" to select your target file.
   - Click "Analyze Target File" and then "Sort and Store Target Audio".

3. 🎚️ Adjust Feature Weights
   - Use sliders to adjust feature weights like Spectral Flatness, MFCC, and more!
   - Optional: Try "DTW for Matching" to improve feature alignment.
   - Click "Randomize Weights" to explore different configurations with just a click.

4. 🎶 Resynthesize and Save
   - Click "Resynthesize Audio" to generate your masterpiece.
   - Optional: Enable "Dynamic Crossfade" and adjust the Low Energy Threshold for smoother sound transitions.
   - Use "Play" to preview and "Save" to keep your creation!

💡 Enjoy experimenting with the magical world of audio resynthesis! ✨
"""
        tutorial_label = tk.Label(tutorial_window, text=tutorial_text, justify=tk.LEFT)
        tutorial_label.pack(padx=10, pady=10)

        close_button = tk.Button(tutorial_window, text="Close", command=tutorial_window.destroy)
        close_button.pack(pady=5)

    def show_full_tutorial(self):
        """Display a window with the full detailed tutorial."""
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("BJ Sample Resynth App - Tutorial")
        tutorial_text = """
🌟 **Detailed Tutorial for BJ Sample Resynth** 🌟

Welcome to BJ Sample Resynth! This guide will help you navigate the app's features and understand how to make the most of them for your audio processing needs. Let’s dive in!

### 1. 🎛️ **Load and Analyze Dataset**

**Step 1: Load Samples**
- Click **"Load Samples for Dataset"** to select a folder containing audio files you want to analyze. The app will load these files for further processing.

**Step 2: Adjust Window Size and Overlap**
- **Window Size**: This controls how much of the audio is analyzed at once. A larger window size improves frequency resolution, which is crucial for accurately analyzing lower frequencies such as bass. Use the slider to set the window size according to your needs.
- **Overlap**: This determines how much each window overlaps with the next one. More overlap provides a smoother analysis but increases processing time. Set this value to balance detail with performance.

**Optional Adjustments:**
- **Energy-Based Chunking**: When enabled, this feature analyzes sustained notes (like bass) more effectively by chunking the audio based on energy levels. This can improve the handling of sounds with long durations.
- **Larger Frames for Low-Frequency Content**: Activating this option will use larger frames specifically for low-frequency content, enhancing the analysis of bass and low-pitched sounds.

**Next Steps:**
- Click **"Analyze Samples"** to start analyzing the loaded audio files.
- After analysis, click **"Sort and Store Dataset"** to organize and save your processed dataset, preparing it for comparison with the target audio.

### 2. 🎯 **Load and Analyze Target Audio**

**Step 1: Load Target Audio**
- Click **"Load Target Audio"** to select the audio file you want to resynthesize.

**Step 2: Analyze Target File**
- Click **"Analyze Target File"** to process the target audio using the settings you’ve configured.
- Once the analysis is complete, click **"Sort and Store Target Audio"** to prepare the target audio for the resynthesis process.

### 3. 🎚️ **Adjust Feature Weights**

**Feature Weights:**
- **Spectral Flatness**: This weight helps distinguish tonal sounds (like bass) from noise. Increasing it makes the app focus more on flatness, which is useful for identifying bass-heavy elements.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the timbral characteristics of the audio, essential for understanding the texture and quality of the sound.
- **Pitch**: Adjusts the influence of the pitch on the analysis. A higher weight means the pitch will play a more significant role in matching features.
- **RMS (Root Mean Square)**: Represents the energy or loudness of the audio signal. Increasing this weight helps in focusing on the amplitude of the sound.
- **Chroma**: This feature relates to the pitch class profile, important for understanding harmonic content.
- **Mel Spectrogram**: Represents the audio in the mel scale, useful for capturing detailed frequency content.
- **Tempo**: Influences the timing and rhythm aspects of the audio. Adjust this weight to emphasize the temporal aspects of the sound.
- **Transient**: This feature captures sharp changes in the audio signal, useful for detecting percussive elements.
- **Spectral Centroid**: Indicates the center of mass of the spectrum, useful for identifying the brightness of the sound.
- **Spectral Rolloff**: Helps to distinguish between harmonic and non-harmonic content by focusing on the frequency below which a certain percentage of the total spectral energy is contained.

**Optional Adjustments:**
- **DTW (Dynamic Time Warping) for Matching**: When enabled, DTW aligns time-stretched features more accurately, improving the temporal alignment between the dataset and target audio. This is particularly useful for matching audio features that vary in speed or length.

**Experiment with Weights:**
- You can fine-tune the resynthesis by adjusting these weights. To explore different configurations, click **"Randomize Weights"** to shuffle the feature weights and see how it affects the audio output.

### 4. 🎵 **Resynthesize and Save**

**Step 1: Resynthesize Audio**
- After analyzing and configuring the weights, click **"Resynthesize Audio"** to generate your output. The app will use the settings and weights to create a new audio file based on the target audio and dataset.

**Optional Adjustments:**
- **Dynamic Crossfade**: Enable this feature for smooth transitions between different audio segments. Adjust the **Low Energy Threshold** to control how sensitive the crossfade is to changes in energy levels.

**Step 2: Preview and Save**
- Use the **"Play"** button to listen to the resynthesized audio. If you're satisfied with the result, click **"Save"** to export the audio file.

🎉 **That’s it!** You’ve now learned how to use BJ Sample Resynth to analyze, adjust, and resynthesize audio. Explore the different settings and weights to create unique and customized audio outputs.

**Pro Tip:** Experiment with various settings and weights to uncover new and exciting audio possibilities. Enjoy the creative process! 🎶

"""
        # Using a Text widget with a scrollbar for the tutorial
        text_widget = tk.Text(tutorial_window, wrap=tk.WORD)
        text_widget.insert(tk.END, tutorial_text)
        text_widget.configure(state='disabled')  # Make the text read-only
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(tutorial_window, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget['yscrollcommand'] = scrollbar.set

        close_button = tk.Button(tutorial_window, text="Close", command=tutorial_window.destroy)
        close_button.pack(pady=5)
        
    def randomize_weights(self):
        """Randomize weights for each feature between 0 and 2."""
        for key in self.weights:
            random_weight = random.uniform(0, 2)
            self.weights[key].set(random_weight)
        self.progress_label.config(text="Weights randomized")

    def resynthesize_audio(self):
        """Resynthesize audio using the dataset and target features."""
        if self.dataset_features is None or self.target_features is None:
            messagebox.showerror("Error", "Dataset or target audio not available")
            return

        self.progress_label.config(text="Resynthesizing audio...")
        self.progress['maximum'] = len(self.target_features)
        self.progress['value'] = 0
        self.root.update_idletasks()

        try:
            dataset_chops = [np.array(row['chop']) for _, row in self.dataset_features.iterrows()]

            resynthesized_audio = []
            previous_chop = None
            crossfade_length = int(self.crossfade_entry.get())
            for i, target_row in self.target_features.iterrows():
                target_features = np.concatenate([
                    target_row['chroma'],
                    target_row['mfcc'],
                    [target_row['spectral_centroid']],
                    [target_row['spectral_rolloff']],
                    [target_row['zero_crossing_rate']],
                    [target_row['rms']],
                    [target_row['pitch']],
                    [target_row['amplitude']],
                    target_row['mel_spectrogram'],
                    [target_row['tempo']],
                    [target_row['transient']]
                ])

                dataset_features = np.array([
                    np.concatenate([
                        row['chroma'],
                        row['mfcc'],
                        [row['spectral_centroid']],
                        [row['spectral_rolloff']],
                        [row['zero_crossing_rate']],
                        [row['rms']],
                        [row['pitch']],
                        [row['amplitude']],
                        row['mel_spectrogram'],
                        [row['tempo']],
                        [row['transient']]
                    ])
                    for _, row in self.dataset_features.iterrows()
                ])

                # Apply weights to features
                weights = np.concatenate([
                    np.full(len(target_row['chroma']), self.weights['chroma'].get()),
                    np.full(len(target_row['mfcc']), self.weights['mfcc'].get()),
                    np.full(1, self.weights['spectral_centroid'].get()),
                    np.full(1, self.weights['spectral_rolloff'].get()),
                    np.full(1, self.weights['zero_crossing_rate'].get()),
                    np.full(1, self.weights['rms'].get()),
                    np.full(1, self.weights['pitch'].get()),
                    np.full(1, self.weights['amplitude'].get()),
                    np.full(len(target_row['mel_spectrogram']), self.weights['mel_spectrogram'].get()),
                    np.full(1, self.weights['tempo'].get()),
                    np.full(1, self.weights['transient'].get())
                ])

                target_features = target_features * weights
                dataset_features = dataset_features * weights

                # Apply DTW if enabled
                if self.dtw_var.get():
                    distances = [fastdtw(target_features, dataset_feature, dist=euclidean)[0] for dataset_feature in dataset_features]
                else:
                    distances = np.linalg.norm(dataset_features - target_features, axis=1)

                closest_index = np.argmin(distances)
                closest_chop = dataset_chops[closest_index]

                # Apply normalization and pitch shifting if enabled
                if self.normalize_var.get():
                    target_rms = target_row['rms']
                    closest_chop_rms = librosa.feature.rms(y=closest_chop).mean()
                    closest_chop = closest_chop * (target_rms / closest_chop_rms)

                if self.pitch_shift_var.get():
                    target_pitch = target_row['pitch']
                    closest_chop_pitch = librosa.yin(closest_chop, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')).mean()
                    n_steps = librosa.hz_to_midi(target_pitch) - librosa.hz_to_midi(closest_chop_pitch)
                    closest_chop = librosa.effects.pitch_shift(closest_chop, sr=self.sample_rate, n_steps=n_steps)

                # Apply cross-fade between adjacent chops
                if previous_chop is not None:
                    fade_length = crossfade_length  # Cross-fade length
                    fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_length)))
                    fade_out = 0.5 * (1 - np.cos(np.pi * np.linspace(1, 0, fade_length)))
                    previous_chop[-fade_length:] = previous_chop[-fade_length:] * fade_out + closest_chop[:fade_length] * fade_in
                    resynthesized_audio.extend(previous_chop[:-fade_length])
                    closest_chop = closest_chop[fade_length:]

                previous_chop = closest_chop

                self.progress['value'] = i + 1
                self.root.update_idletasks()

            resynthesized_audio.extend(previous_chop)
            self.resynthesized_audio = np.array(resynthesized_audio, dtype=np.float32)
            self.progress_label.config(text="Resynthesis complete")
            self.audio_player.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resynthesize audio: {e}")
            self.progress_label.config(text="Failed to resynthesize audio")

    def reset_to_default(self):
        self.window_size = 2048
        self.window_size_scale.set(self.window_size)
        self.window_size_entry.delete(0, tk.END)
        self.window_size_entry.insert(0, str(self.window_size))

        self.overlap_scale.set(50)
        self.overlap_entry.delete(0, tk.END)
        self.overlap_entry.insert(0, str(50))

        self.crossfade_scale.set(self.hop_length // 2)
        self.crossfade_entry.delete(0, tk.END)
        self.crossfade_entry.insert(0, str(self.hop_length // 2))

        self.normalize_var.set(True)
        self.pitch_shift_var.set(False)
        self.dynamic_crossfade_var.set(True)

        for key in self.weights:
            self.weights[key].set(1.0)

        self.progress_label.config(text="Values reset to default")

    def reset_analysis(self):
        analysed_dir = 'analysed_files'
        try:
            if os.path.exists(analysed_dir):
                shutil.rmtree(analysed_dir)
            self.dataset_features = None
            self.target_features = None
            self.analyze_button.config(state=tk.DISABLED)
            self.sort_store_button.config(state=tk.DISABLED)
            self.analyze_target_button.config(state=tk.DISABLED)
            self.sort_store_target_button.config(state=tk.DISABLED)
            self.resynthesize_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.audio_player.config(state=tk.DISABLED)
            self.progress_label.config(text="Analysis reset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset analysis: {e}")

    def update_window_size(self, value):
        self.window_size = int(value)
        self.window_size_entry.delete(0, tk.END)
        self.window_size_entry.insert(0, str(self.window_size))
        self.update_hop_length()

    def update_overlap(self, value):
        overlap_percent = int(value)
        self.overlap_entry.delete(0, tk.END)
        self.overlap_entry.insert(0, str(overlap_percent))
        self.update_hop_length()

    def update_hop_length(self):
        overlap_percent = int(self.overlap_entry.get())
        self.hop_length = self.window_size * (100 - overlap_percent) // 100

    def update_crossfade(self, value):
        self.crossfade_entry.delete(0, tk.END)
        self.crossfade_entry.insert(0, str(value))

    def load_samples(self):
        """Allow loading samples from a folder or multi-selecting individual files, and resample to 44100 Hz if needed."""
        self.samples = []
        target_sample_rate = 44100  # Ensure everything is resampled to 44100 Hz
        
        # Ask the user if they want to load a folder or individual files
        file_option = messagebox.askyesno("Load Audio", "Do you want to load a folder of samples? (Select 'No' for individual files)")
        
        if file_option:
            # Folder selection
            folder_path = filedialog.askdirectory()  # Ask user to select a directory
            if folder_path:
                audio_extensions = ['.wav', '.mp3']  # Supported file extensions
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in audio_extensions):
                            file_path = os.path.join(root, file)
                            try:
                                y, sr = librosa.load(file_path, sr=None)  # Load audio without resampling
                                if sr != target_sample_rate:
                                    # Resample to 44100 Hz if the sample rate is different
                                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
                                    sr = target_sample_rate  # Update the sample rate to 44100 Hz
                                self.samples.append((y, sr))
                            except Exception as e:
                                messagebox.showerror("Sample Loading Error", f"Error loading {file_path}: {e}")
                                self.progress_label.config(text="Sample loading error")
                                return
            else:
                self.progress_label.config(text="No folder selected.")
        else:
            # Multi-select file option
            file_paths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.mp3")])
            if file_paths:
                for file_path in file_paths:
                    try:
                        y, sr = librosa.load(file_path, sr=None)  # Load audio without resampling
                        if sr != target_sample_rate:
                            # Resample to 44100 Hz if the sample rate is different
                            y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
                            sr = target_sample_rate  # Update the sample rate to 44100 Hz
                        self.samples.append((y, sr))
                    except Exception as e:
                        messagebox.showerror("Sample Loading Error", f"Error loading {file_path}: {e}")
                        self.progress_label.config(text="Sample loading error")
                        return
            else:
                self.progress_label.config(text="No files selected.")

        if self.samples:
            self.progress_label.config(text="Samples loaded successfully")
            self.analyze_button.config(state=tk.NORMAL)
        else:
            self.progress_label.config(text="No valid audio files found.")


    def analyze_samples(self):
        if not self.samples:
            messagebox.showerror("Error", "No samples loaded")
            return

        self.progress_label.config(text="Analyzing samples...")
        self.progress['maximum'] = len(self.samples)
        self.progress['value'] = 0
        self.root.update_idletasks()

        dataset = []
        try:
            for i, (y, sr) in enumerate(self.samples):
                self.sample_rate = sr  # Ensure the sample rate is consistent
                hops = librosa.util.frame(y, frame_length=self.window_size, hop_length=self.hop_length)
                if hops.shape[1] == 0:
                    continue
                for chop in hops.T:
                    features = extract_features(chop, sr)
                    dataset.append(features)

                self.progress['value'] = i + 1
                self.root.update_idletasks()

            if not dataset:
                raise ValueError("No valid frames found in the samples.")

            self.dataset_features = pd.DataFrame(dataset)

            self.progress_label.config(text="Samples analyzed successfully")
            self.sort_store_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze samples: {e}")
            self.progress_label.config(text="Failed to analyze samples")

    def load_target_audio(self):
        """Load the target audio file for resynthesis."""
        self.target_audio = None
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            try:
                y, sr = librosa.load(file_path, sr=None)  # Load audio without resampling
                if self.sample_rate is None:
                    self.sample_rate = sr  # Set the sample rate based on the first loaded target
                elif sr != self.sample_rate:
                    messagebox.showerror("Sample Rate Mismatch", f"Target sample rate ({sr} Hz) does not match dataset sample rate ({self.sample_rate} Hz).")
                    return
                self.target_audio = (y, sr)
                self.progress_label.config(text="Target audio loaded successfully")
                self.analyze_target_button.config(state=tk.NORMAL)  # Enable the 'Analyze Target' button
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load target audio: {e}")
                self.progress_label.config(text="Failed to load target audio")
        else:
            self.progress_label.config(text="Failed to load target audio")

    def analyze_target_file(self):
        """Analyze the target audio file and extract features."""
        if self.target_audio is None:
            messagebox.showerror("Error", "No target audio loaded")
            return

        y, sr = self.target_audio
        self.progress_label.config(text="Analyzing target file...")
        self.progress['maximum'] = len(y) // self.window_size
        self.progress['value'] = 0
        self.root.update_idletasks()

        target_features = []
        try:
            hops = librosa.util.frame(y, frame_length=self.window_size, hop_length=self.hop_length)
            if hops.shape[1] == 0:
                raise ValueError("No valid frames found in the target audio.")

            for i, chop in enumerate(hops.T):
                features = extract_features(chop, sr)
                target_features.append(features)

                self.progress['value'] = i + 1
                self.root.update_idletasks()

            self.target_features = pd.DataFrame(target_features)
            self.progress_label.config(text="Target file analyzed successfully")
            self.sort_store_target_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze target file: {e}")
            self.progress_label.config(text="Failed to analyze target file")

    def sort_and_store_dataset(self):
        if self.dataset_features is None:
            messagebox.showerror("Error", "No dataset features to sort and store")
            return

        self.progress_label.config(text="Sorting and storing dataset...")
        self.progress['maximum'] = 100  # Fake maximum for progress bar
        self.progress['value'] = 0
        self.root.update_idletasks()

        try:
            base_dir = 'analysed_files/analysed_dataset/'
            features_dir = os.path.join(base_dir, 'features')
            chops_dir = os.path.join(base_dir, 'dataset_chops')

            os.makedirs(features_dir, exist_ok=True)
            os.makedirs(chops_dir, exist_ok=True)

            feature_summaries = self.dataset_features.copy()
            feature_summaries['chroma_mean'] = self.dataset_features['chroma'].apply(lambda x: np.mean(x))
            feature_summaries['mfcc_mean'] = self.dataset_features['mfcc'].apply(lambda x: np.mean(x))
            feature_summaries['mel_spectrogram_mean'] = self.dataset_features['mel_spectrogram'].apply(lambda x: np.mean(x))

            features = feature_summaries.drop(columns=['chop', 'chroma', 'mfcc', 'mel_spectrogram']).values
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            sorted_indices = np.lexsort(scaled_features.T)
            sorted_features = feature_summaries.iloc[sorted_indices]

            sorted_features.to_pickle(os.path.join(features_dir, 'dataset_features.pkl'))

            for i, row in sorted_features.iterrows():
                chop = np.array(row['chop'])
                file_path = os.path.join(chops_dir, f'chop_{i}.wav')
                write(file_path, self.sample_rate, chop.astype(np.float32))  # Ensure original sample rate is preserved

            self.progress_label.config(text="Dataset sorted and stored successfully")
            self.load_target_button.config(state=tk.NORMAL)
            self.progress['value'] = 100
            self.root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to sort and store dataset: {e}")
            self.progress_label.config(text="Failed to sort and store dataset")

    def sort_and_store_target_audio(self):
        if self.target_features is None:
            messagebox.showerror("Error", "No target features to sort and store")
            return

        self.progress_label.config(text="Sorting and storing target audio...")
        self.progress['maximum'] = 100  # Fake maximum for progress bar
        self.progress['value'] = 0
        self.root.update_idletasks()

        try:
            base_dir = 'analysed_files/analysed_target_file/'
            features_dir = os.path.join(base_dir, 'features')
            chops_dir = os.path.join(base_dir, 'target_chops')

            os.makedirs(features_dir, exist_ok=True)
            os.makedirs(chops_dir, exist_ok=True)

            feature_summaries = self.target_features.copy()
            feature_summaries['chroma_mean'] = self.target_features['chroma'].apply(lambda x: np.mean(x))
            feature_summaries['mfcc_mean'] = self.target_features['mfcc'].apply(lambda x: np.mean(x))
            feature_summaries['mel_spectrogram_mean'] = self.target_features['mel_spectrogram'].apply(lambda x: np.mean(x))

            features = feature_summaries.drop(columns=['chop', 'chroma', 'mfcc', 'mel_spectrogram']).values
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            sorted_indices = np.lexsort(scaled_features.T)
            sorted_features = feature_summaries.iloc[sorted_indices]

            sorted_features.to_pickle(os.path.join(features_dir, 'target_features.pkl'))

            for i, row in sorted_features.iterrows():
                chop = np.array(row['chop'])
                file_path = os.path.join(chops_dir, f'chop_{i}.wav')
                write(file_path, self.sample_rate, chop.astype(np.float32))  # Ensure original sample rate is preserved

            self.progress_label.config(text="Target audio sorted and stored successfully")
            self.resynthesize_button.config(state=tk.NORMAL)
            self.progress['value'] = 100
            self.root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to sort and store target audio: {e}")
            self.progress_label.config(text="Failed to sort and store target audio")

    def play_audio(self):
        if self.resynthesized_audio is not None:
            import sounddevice as sd
            sd.play(self.resynthesized_audio, self.sample_rate)
            sd.wait()
        else:
            messagebox.showerror("Error", "No resynthesized audio available to play")

    def save_resynthesized_audio(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".wav",
                                                 filetypes=[("WAV Files", "*.wav")])
        if file_path:
            try:
                write(file_path, self.sample_rate, self.resynthesized_audio)
                messagebox.showinfo("Success", "Resynthesized audio saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save resynthesized audio: {e}")