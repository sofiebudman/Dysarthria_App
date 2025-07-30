from IPython.display import Audio
import soundfile as sf
import librosa
import scipy.ndimage
import librosa
import numpy as np
import io
import torch.nn.functional as F
import torch 


def russian_preprocess_audio(file):
    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=None)

    # Full mel spectrogram
    full_S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    full_S_dB = librosa.power_to_db(full_S, ref=np.max)
    full_S_norm = np.clip((full_S_dB + 80) / 80, 0, 1)  # shape: (128, T)

    # Resize to (128, 128) with bilinear interpolation (like training)
    tensor = torch.tensor(full_S_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, T)
    resized = F.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False)
    cropped = resized.squeeze().numpy()  # shape (128, 128)

    return cropped[np.newaxis, np.newaxis, :, :], full_S_norm


def mel_to_audio_rus(mel_img, sr=16000, n_fft=1024, target_duration=None, denoise=True):
    # De-normalize [0,1] → [-80, 0] dB
    mel_img_db = mel_img * 80 - 80

    # Optional denoising: clip low-energy and smooth
    if denoise:
        threshold = np.percentile(mel_img_db, 5)
        mel_img_db = np.clip(mel_img_db, threshold, 0)
        mel_img_db = scipy.ndimage.gaussian_filter(mel_img_db, sigma=1.0)

    # Convert back to waveform using Griffin-Lim
    mel_power = librosa.db_to_power(mel_img_db)
    stft_est = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft)
    wav = librosa.griffinlim(stft_est, n_iter=64, momentum=0.99)

    # Normalize to avoid clipping
    wav = wav / np.max(np.abs(wav)) * 0.9

    # Optional duration matching
    if target_duration is not None:
        current_duration = len(wav) / sr
        stretch_factor = target_duration / current_duration
        wav = librosa.resample(wav, orig_sr=sr, target_sr=int(sr * stretch_factor))

    return wav
