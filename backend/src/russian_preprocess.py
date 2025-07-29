import cv2
from IPython.display import Audio
import soundfile as sf

import librosa

import scipy.ndimage
import librosa
import numpy as np
import io
def preprocess_audio(file):
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

    return S_final 




def mel_to_audio(mel_img, sr=16000, n_fft=1024, target_duration=None, denoise=True):
    mel_img_db = mel_img * 80 - 80  

    if denoise:
        
        threshold = np.percentile(mel_img_db, 5)
        mel_img_db = np.clip(mel_img_db, threshold, 0)

    
        mel_img_db = scipy.ndimage.gaussian_filter(mel_img_db, sigma=1.0)

    mel_power = librosa.db_to_power(mel_img_db)
    stft_est = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft)
    wav = librosa.griffinlim(stft_est, n_iter=64, momentum=0.99)


    wav = wav / np.max(np.abs(wav)) * 0.9

   
    if target_duration is not None:
        current_duration = len(wav) / sr
        stretch_factor = target_duration / current_duration
        wav = librosa.resample(wav, orig_sr=sr, target_sr=int(sr * stretch_factor))

    return wav

