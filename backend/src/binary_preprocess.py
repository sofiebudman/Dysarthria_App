import numpy as np
import librosa
import io

def preprocess_audio(file):

    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio with librosa
    y, sr = librosa.load(audio_buffer, sr=None)

    # Extract MFCCs
    mean_mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128), axis=1)

    return mean_mfcc  # Shape: (128,)
