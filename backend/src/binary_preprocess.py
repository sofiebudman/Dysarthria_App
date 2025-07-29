
import numpy as np
import librosa
import io

#TODO: Invalid input shape for input Tensor("data:0", shape=(32,), dtype=float32). Expected shape (None, 16, 8, 1), but input has incompatible shape (32,)
def binary_preprocess_audio(file):

    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio with librosa
    y, sr = librosa.load(audio_buffer, sr=None)

    # Extract MFCCs
    mean_mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128), axis=1)

    return mean_mfcc  # Shape: (128,)

