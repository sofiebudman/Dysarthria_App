import cv2
from IPython.display import Audio
import soundfile as sf
import librosa
import scipy.ndimage
import librosa
import numpy as np
import io

# AttributeError: 'collections.OrderedDict' object has no attribute 'predict'
def english_preprocess_audio(file):
  

    # Read the uploaded audio file
    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load with librosa
    y, sr = librosa.load(audio_buffer, sr=None)

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    S_norm = np.clip((S_dB + 80) / 80, 0, 1)  # shape: (128, time)

    # Resize or pad to fixed size (128x128)
    if S_norm.shape[1] < 128:
        pad_width = 128 - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_norm = S_norm[:, :128]

    # Add channel and batch dimension: (1, 1, 128, 128)
    S_final = np.expand_dims(S_norm, axis=0)  # (1, 128, 128)
    S_final = np.expand_dims(S_final, axis=0)  # (1, 1, 128, 128)

    return S_final





def mel_to_audio_eng(mel_img, sr=16000, n_fft=1024, target_duration=None, denoise=True):
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
