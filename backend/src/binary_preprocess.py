
import numpy as np
import librosa
import io
def binary_preprocess_audio(file):
    import librosa
    import numpy as np
    import io

    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  # shape: (128, T)
    mean_mfcc = np.mean(mfcc, axis=1)  # shape: (128,)

    reshaped = mean_mfcc.reshape(16, 8, 1)  # shape: (16, 8, 1)
    return np.expand_dims(reshaped, axis=0)  # shape: (1, 16, 8, 1)