<<<<<<< Updated upstream
import io
import cv2 
import librosa
import numpy as np
def russain_preprocess_audio(file):
=======
import numpy as np
import librosa
import io
import cv2  # OpenCV for color mapping and resizing


def multiclass_preprocess_audio(file):
>>>>>>> Stashed changes
    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes) #read audio


    y, sr = librosa.load(audio_buffer, sr=None)

    S = librosa.stft(y, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    S_normalized = 255 * (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_normalized = S_normalized.astype(np.uint8)


    S_color = cv2.applyColorMap(S_normalized, cv2.COLORMAP_VIRIDIS)  # Shape: (freq, time, 3), dtype: uint8


    S_resized = cv2.resize(S_color, (128, 128), interpolation=cv2.INTER_LINEAR)


    S_resized = S_resized.astype(np.float32) / 255.0

    S_final = np.expand_dims(  S_resized, axis=0)

<<<<<<< Updated upstream
    return S_final
=======
    return S_final
>>>>>>> Stashed changes
