import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from concurrent.futures import ThreadPoolExecutor


def apply_weights_to_features(target_row, weights):
    """Apply weights to the features of the target audio."""
    return np.concatenate([
        target_row['chroma'] * weights['chroma'],
        target_row['mfcc'] * weights['mfcc'],
        [target_row['spectral_centroid'] * weights['spectral_centroid']],
        [target_row['spectral_rolloff'] * weights['spectral_rolloff']],
        [target_row['zero_crossing_rate'] * weights['zero_crossing_rate']],
        [target_row['rms'] * weights['rms']],
        [target_row['pitch'] * weights['pitch']],
        [target_row['amplitude'] * weights['amplitude']],
        target_row['mel_spectrogram'] * weights['mel_spectrogram'],
        [target_row['tempo'] * weights['tempo']],
        [target_row['transient'] * weights['transient']]
    ])


def calculate_distances(target_features, dataset_features, dtw_enabled=False):
    """Calculate distances between the target and dataset features."""
    if dtw_enabled:
        return [fastdtw(target_features, dataset_feature, dist=euclidean)[0] for dataset_feature in dataset_features]
    else:
        return np.linalg.norm(dataset_features - target_features, axis=1)


def apply_normalization(chop, target_rms):
    """Normalize amplitude of chop to match the target's RMS."""
    chop_rms = librosa.feature.rms(y=chop).mean()
    return chop * (target_rms / chop_rms)


def apply_pitch_shift(chop, target_pitch, sample_rate):
    """Shift the pitch of the chop to match the target's pitch."""
    chop_pitch = librosa.yin(chop, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')).mean()
    n_steps = librosa.hz_to_midi(target_pitch) - librosa.hz_to_midi(chop_pitch)
    return librosa.effects.pitch_shift(chop, sr=sample_rate, n_steps=n_steps)


def apply_crossfade(previous_chop, closest_chop, crossfade_length):
    """Apply a crossfade between two consecutive chops."""
    fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, crossfade_length)))
    fade_out = 0.5 * (1 - np.cos(np.pi * np.linspace(1, 0, crossfade_length)))
    previous_chop[-crossfade_length:] = previous_chop[-crossfade_length:] * fade_out + closest_chop[:crossfade_length] * fade_in
    return previous_chop[:-crossfade_length], closest_chop[crossfade_length:]


def calculate_distance_for_chop(i, target_row, dataset_chops, dataset_features, weights, dtw_enabled):
    """Calculate the distance for a specific chop in parallel."""
    target_feats = apply_weights_to_features(target_row, weights)
    dataset_feats = np.array([apply_weights_to_features(row, weights) for row in dataset_features])
    distances = calculate_distances(target_feats, dataset_feats, dtw_enabled)
    closest_index = np.argmin(distances)
    return closest_index, dataset_chops[closest_index]


def resynthesize_audio(dataset_features, target_features, dataset_chops, weights, crossfade_length, sample_rate, normalize=True, pitch_shift=False, dtw_enabled=False):
    """Resynthesize audio based on dataset and target features."""
    resynthesized_audio = []
    previous_chop = None
    num_target_features = len(target_features)

    with ThreadPoolExecutor() as executor:
        future_to_chop = {
            executor.submit(calculate_distance_for_chop, i, target_row, dataset_chops, dataset_features, weights, dtw_enabled): i
            for i, target_row in target_features.iterrows()
        }

        for future in future_to_chop:
            try:
                closest_index, closest_chop = future.result()

                # Apply normalization and pitch shifting if enabled
                if normalize:
                    target_rms = target_features.iloc[closest_index]['rms']
                    closest_chop = apply_normalization(closest_chop, target_rms)

                if pitch_shift:
                    target_pitch = target_features.iloc[closest_index]['pitch']
                    closest_chop = apply_pitch_shift(closest_chop, target_pitch, sample_rate)

                # Apply crossfade if applicable
                if previous_chop is not None:
                    previous_chop, closest_chop = apply_crossfade(previous_chop, closest_chop, crossfade_length)
                    resynthesized_audio.extend(previous_chop)

                previous_chop = closest_chop

            except Exception as e:
                print(f"Error in resynthesizing chop: {e}")

    if previous_chop is not None:
        resynthesized_audio.extend(previous_chop)

    return np.array(resynthesized_audio, dtype=np.float32)