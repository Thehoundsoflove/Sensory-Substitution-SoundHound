import os
import cv2
import numpy as np
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def image_to_sine_wave(image_path, output_audio_path, duration=3, sample_rate=44100):
    """
    Convert an image to an enhanced sine wave representation for audio.
    Enhanced mapping includes frequency range, edge detection, and spatial effects.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Resize image to a fixed size (50x50) for simplicity
    img = cv2.resize(img, (350, 350))

    # Edge detection using Sobel filter
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges_normalized = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

    # Normalize pixel values to range 0-1
    img_normalized = img / 255.0

    # Create a time array for the audio signal
    total_samples = duration * sample_rate
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # Initialize audio signals for stereo (left and right channels)
    audio_signal_left = np.zeros(total_samples)
    audio_signal_right = np.zeros(total_samples)

    # Map pixel values to frequencies and amplitudes
    pixel_values = img_normalized.flatten()
    edge_values = edges_normalized.flatten()
    num_pixels = len(pixel_values)
    samples_per_pixel = total_samples // num_pixels

    for i, (pixel_value, edge_value) in enumerate(zip(pixel_values, edge_values)):
        start_idx = i * samples_per_pixel
        end_idx = (i + 1) * samples_per_pixel

        # Frequency mapping: pixel intensity affects frequency range
        frequency = 100 + pixel_value * 4900  # Frequency in Hz (100 Hz to 5000 Hz)

        # Edge-weighted amplitude
        amplitude = edge_value  # Edge strength determines loudness

        # Generate sine wave
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t[start_idx:end_idx])

        # Spatial effects: left/right channel based on horizontal position
        col = i % 50  # Column index in the 50x50 image
        panning_factor = col / 50.0  # Left-right balance
        audio_signal_left[start_idx:end_idx] += (1 - panning_factor) * sine_wave
        audio_signal_right[start_idx:end_idx] += panning_factor * sine_wave

    # Normalize the audio signals
    max_amplitude = max(np.max(np.abs(audio_signal_left)), np.max(np.abs(audio_signal_right)))
    if max_amplitude > 0:
        audio_signal_left = np.int16(audio_signal_left / max_amplitude * 32767)
        audio_signal_right = np.int16(audio_signal_right / max_amplitude * 32767)
    else:
        raise ValueError("Generated audio contains only silence.")

    # Combine left and right channels into stereo
    stereo_signal = np.column_stack((audio_signal_left, audio_signal_right))

    # Write the stereo audio signal to a .wav file
    write(output_audio_path, sample_rate, stereo_signal)
    print(f"Audio saved to {output_audio_path}")

def select_image():
    """
    Open a file dialog to select an image and generate the sine wave audio.
    """
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if file_path:
        input_label.config(text=f"Selected Image: {os.path.basename(file_path)}")
        output_audio_path = filedialog.asksaveasfilename(
            title="Save Audio As",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav")]
        )
        if output_audio_path:
            try:
                image_to_sine_wave(file_path, output_audio_path)
                messagebox.showinfo("Success", f"Audio file saved to: {output_audio_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

# Create the GUI
root = tk.Tk()
root.title("Image to Enhanced Sine Wave Converter")

# Frame for instructions
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, padx=10, pady=10, sticky="EW")

# Instruction label
instruction_label = ttk.Label(
    frame, 
    text="Select an image to generate a sine wave audio file. "
         "The image will be resized to 50x50 resolution."
)
instruction_label.grid(row=0, column=0, pady=(0, 10), sticky="W")

# Input image label
input_label = ttk.Label(frame, text="No image selected.")
input_label.grid(row=1, column=0, pady=(0, 10), sticky="W")

# Select image button
select_button = ttk.Button(frame, text="Select Image", command=select_image)
select_button.grid(row=2, column=0, pady=(10, 0), sticky="W")

# Run the main loop
root.mainloop()
