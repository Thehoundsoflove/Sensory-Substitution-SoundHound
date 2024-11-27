import os
import cv2
import numpy as np
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk


def image_to_sine_wave(image_path, output_audio_path, duration=3, sample_rate=44100):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    img = cv2.resize(img, (350, 350))

    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edges_normalized = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)

    img_normalized = img / 255.0
    total_samples = duration * sample_rate
    t = np.linspace(0, duration, total_samples, endpoint=False)

    audio_signal_left = np.zeros(total_samples)
    audio_signal_right = np.zeros(total_samples)

    pixel_values = img_normalized.flatten()
    edge_values = edges_normalized.flatten()
    num_pixels = len(pixel_values)
    samples_per_pixel = total_samples // num_pixels

    for i, (pixel_value, edge_value) in enumerate(zip(pixel_values, edge_values)):
        start_idx = i * samples_per_pixel
        end_idx = (i + 1) * samples_per_pixel
        frequency = 100 + pixel_value * 4900
        amplitude = edge_value
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t[start_idx:end_idx])
        col = i % 50
        panning_factor = col / 50.0
        audio_signal_left[start_idx:end_idx] += (1 - panning_factor) * sine_wave
        audio_signal_right[start_idx:end_idx] += panning_factor * sine_wave

    max_amplitude = max(np.max(np.abs(audio_signal_left)), np.max(np.abs(audio_signal_right)))
    if max_amplitude > 0:
        audio_signal_left = np.int16(audio_signal_left / max_amplitude * 32767)
        audio_signal_right = np.int16(audio_signal_right / max_amplitude * 32767)
    else:
        raise ValueError("Generated audio contains only silence.")

    stereo_signal = np.column_stack((audio_signal_left, audio_signal_right))
    write(output_audio_path, sample_rate, stereo_signal)
    print(f"Audio saved to {output_audio_path}")


def select_image():
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


def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the camera.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image from camera.")
        return

    # Save the captured image temporarily
    temp_image_path = "captured_image.png"
    cv2.imwrite(temp_image_path, frame)

    # Display the captured image in the GUI
    captured_img = Image.open(temp_image_path)
    captured_img.thumbnail((300, 300))
    captured_img_tk = ImageTk.PhotoImage(captured_img)
    image_label.config(image=captured_img_tk)
    image_label.image = captured_img_tk
    input_label.config(text="Captured Image")

    # Prompt user to save the audio file
    output_audio_path = filedialog.asksaveasfilename(
        title="Save Audio As",
        defaultextension=".wav",
        filetypes=[("WAV Files", "*.wav")]
    )
    if output_audio_path:
        try:
            image_to_sine_wave(temp_image_path, output_audio_path)
            messagebox.showinfo("Success", f"Audio file saved to: {output_audio_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


# Create the GUI
root = tk.Tk()
root.title("Image to Enhanced Sine Wave Converter")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, padx=10, pady=10, sticky="EW")

instruction_label = ttk.Label(
    frame,
    text="Select an image or capture one with the camera to generate a sine wave audio file."
)
instruction_label.grid(row=0, column=0, pady=(0, 10), sticky="W")

input_label = ttk.Label(frame, text="No image selected.")
input_label.grid(row=1, column=0, pady=(0, 10), sticky="W")

select_button = ttk.Button(frame, text="Select Image", command=select_image)
select_button.grid(row=2, column=0, pady=(10, 5), sticky="W")

capture_button = ttk.Button(frame, text="Capture Image", command=capture_image)
capture_button.grid(row=3, column=0, pady=(5, 10), sticky="W")

image_label = ttk.Label(frame)
image_label.grid(row=4, column=0, pady=(10, 0))

root.mainloop()
