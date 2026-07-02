import numpy as np
import librosa

def load_and_align_waveform(file_path, target_sr=16000, duration=10):
    """
    Loads raw biomedical audio samples and forces them into an exact 
    10-second uniform length to ensure input tensor alignment .
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
        target_length = target_sr * duration
        
        # Strictly pad or truncate to keep the sample sizes uniform
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        else:
            audio = audio[:target_length]
        return audio
    except Exception as e:
        print(f"[ERROR] Failed loading audio asset {file_path}: {e} - utils.py:20")
        return None

def extract_acoustic_features(audio, sr=16000, n_mels=128, n_mfcc=13):
    """
    Extracts log-Mel Spectrogram matrices or MFCCs from clinical acoustic signals 
    to feed into the TinyML classifier model layers .
    """
    if audio is None:
        return None
    
    # 1. Compute Mel Spectrogram Base 
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Optional alternative feature array: MFCC extraction 
    # mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Standardize output array shape for tensor graph compatibility: (Height, Width, Channels)
    normalized_features = log_mel_spec[..., np.newaxis]
    return normalized_features