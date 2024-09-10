import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def extract_features(chop, sr):
    """
    Extracts various audio features from a given audio segment (chop).
    :param chop: Audio segment (numpy array).
    :param sr: Sample rate of the audio segment.
    :return: Dictionary containing extracted features.
    """
    chroma = librosa.feature.chroma_stft(y=chop, sr=sr).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=chop, sr=sr, n_mfcc=13).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=chop, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=chop, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=chop).mean()
    rms = librosa.feature.rms(y=chop).mean()
    pitch = librosa.yin(chop, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')).mean()
    amplitude = librosa.amplitude_to_db(np.abs(chop)).mean()
    mel_spectrogram = librosa.feature.melspectrogram(y=chop, sr=sr).mean(axis=1)
    tempo, _ = librosa.beat.beat_track(y=chop, sr=sr)

    # Transient (onset strength)
    onset_strength = librosa.onset.onset_strength(y=chop, sr=sr).mean()

    return {
        'chroma': chroma,
        'mfcc': mfcc,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate,
        'rms': rms,
        'pitch': pitch,
        'amplitude': amplitude,
        'mel_spectrogram': mel_spectrogram,
        'tempo': tempo,
        'transient': onset_strength,  # Adding transient feature
        'chop': chop
    }


def parallel_extract_features(chops, sr, max_workers=4):
    """
    Extract features from multiple audio chops in parallel.
    :param chops: List of audio chops (numpy arrays).
    :param sr: Sample rate for all audio chops.
    :param max_workers: Maximum number of threads for parallel processing.
    :return: List of feature dictionaries for each chop.
    """
    def process_chop(chop):
        return extract_features(chop, sr)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        feature_list = list(executor.map(process_chop, chops))

    return feature_list


def frame_audio_and_extract_features(audio, window_size, hop_length, sr, max_workers=4):
    """
    Frame the audio into overlapping windows and extract features from each frame in parallel.
    :param audio: Audio signal (numpy array).
    :param window_size: Size of each window (number of samples).
    :param hop_length: Number of samples between the starts of consecutive frames.
    :param sr: Sample rate of the audio.
    :param max_workers: Maximum number of threads for parallel processing.
    :return: List of feature dictionaries.
    """
    # Frame the audio into overlapping windows
    hops = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_length)

    # Extract features in parallel
    features = parallel_extract_features(hops.T, sr, max_workers=max_workers)

    return features