import numpy as np
import cv2
from scipy.io.wavfile import read
import tkinter as tk
from tkinter import filedialog

def decode_audio_to_image(audio_path, output_image_path):
    """
    Decode an audio file back into an image representation.
    """
    # Read the audio file
    sample_rate, audio_signal = read(audio_path)

    # Check if the audio signal is empty or silent
    max_val = np.max(np.abs(audio_signal))
    if max_val == 0:
        raise ValueError("Audio signal is silent or contains no valid data.")

    # Normalize the audio signal to the range -1 to 1
    audio_signal = audio_signal / max_val

    # Determine image size (assumed to be 50x50 based on first program)
    image_size = (350, 350)

    # Create an empty image to store the decoded information
    decoded_image = np.zeros(image_size, dtype=np.uint8)

    # Length of the signal (in terms of audio samples)
    total_samples = len(audio_signal)

    # Calculate how many samples correspond to each pixel
    pixels_per_sample = total_samples // (image_size[0] * image_size[1])

    # Decode the audio into pixel values
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            # For each pixel, accumulate the signal's corresponding section
            start_idx = (i * image_size[1] + j) * pixels_per_sample
            end_idx = start_idx + pixels_per_sample
            pixel_value = np.sum(audio_signal[start_idx:end_idx])

            # Ensure that pixel_value is a valid number
            if np.isnan(pixel_value):
                pixel_value = 0  # Replace NaN with 0

            # Map the sum of the segment to a valid pixel value (0-255)
            decoded_image[i, j] = np.clip(int((pixel_value + 1) * 127), 0, 255)

    # Save the decoded image
    cv2.imwrite(output_image_path, decoded_image)
    print(f"Decoded image saved to {output_image_path}")

def open_audio():
    """
    Open the audio file dialog and pass the selected audio path to decode to an image.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter root window
    audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if audio_path:
        # Decode audio to image and save it
        output_image_path = "decoded_image.png"
        decode_audio_to_image(audio_path, output_image_path)

# Run the program by calling the open_audio function
open_audio()
