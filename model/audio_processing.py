import numpy as np
import librosa

def parse_and_normalize_waveform(file_path, sample_rate=16000, max_duration=10):
    """Loads raw target audio assets and clips/pads to exact uniform length profiles."""
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=max_duration)
        fixed_size = sample_rate * max_duration
        
        if len(audio) < fixed_size:
            audio = np.pad(audio, (0, fixed_size - len(audio)), 'constant')
        else:
            audio = audio[:fixed_size]
        return audio
    except Exception as e:
        print(f"[ERROR] Failed to extract from {file_path}: {e} - audio_processing.py:16")
        return None

def generate_mel_spectrogram_matrix(audio_array, sample_rate=16000, n_mels=128):
    """Transforms a raw 1D audio waveform vector into a 2D Log-Mel structural array."""
    if audio_array is None:
        return None
    mel_power_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_power_spec, ref=np.max)
    return log_mel_spec[..., np.newaxis] # Appends channels layer