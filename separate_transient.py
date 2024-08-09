# Separate audio into transient and residual components

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import pywt
import utils

# TODO: Separate harmonic component first

def separate_transient(audio_file, output_path, window_size=1024, overlap=0.5, window_type='hamming',
                       wavelet='db1', s=0, p=2, threshold=1.5, plot=False):
    """
    Separate audio into transient and residual components
        Input: audio_file (str) - path to audio file
               output_path (str) - path to output folder
               window_size (int) - length of window in samples
               overlap (float) - percent overlap between windows
               window_type (str) - type of window to use
               wavelet (str) - wavelet type
               s (int) - weighting of scale/level of DWT coefficients
               p (int) - power of the modulus to focus on type of transients
               threshold (float) - threshold value for pruning
               plot (bool) - whether to plot the audio signal, transient audio, and residual audio
        Output: audio (np.array) - original audio
                transient_audio (np.array) - transient part of audio
                residual_audio (np.array) - residual part of audio
                sr (int) - sample rate of audio
    """
    assert np.log2(window_size).is_integer(), "Window size must be a power of 2"

    # Load audio file
    audio, sr = librosa.load(audio_file)

    hop_length = int(window_size * (1 - overlap))

    # Window audio
    window = librosa.filters.get_window(window_type, window_size)
    audio_frames = librosa.util.frame(audio, frame_length=window_size, hop_length=hop_length, axis=0) * window

    # Initialize list to store coefficient trees and regularity measure at each frame
    coeffs = []
    k = np.zeros((len(audio_frames), window_size//2))

    # Loop through audio frames
    for i, frame in enumerate(audio_frames):
        # Take multilevel DWT of audio frame
        coeffs.append(pywt.wavedec(frame, wavelet=wavelet))
        # Calculate modulus of regularity at each leaf of coefficient tree
        k[i] = utils.calculate_regularity_measure(coeffs[i], s=s, p=p)

    # Prune coefficient tree based on thresholded regularity measure
    pruned_coeffs = utils.prune_coefficient_tree(coeffs, k, threshold=threshold)

    # Reconstruction of the transient part of the audio for each frame
    transient_audio_frames = np.zeros_like(audio_frames)
    # Loop through audio frames
    for frame_number, frame in enumerate(pruned_coeffs):
        # Reconstruct audio frame using inverse DWT
        transient_audio_frames[frame_number] = pywt.waverec(frame, wavelet=wavelet)

    assert transient_audio_frames.shape == audio_frames.shape, "Transient audio shape does not match input audio shape"

    # Implement overlap-add to reverse the windowing process done earlier
    output_length = len(audio)
    transient_audio = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i, frame in enumerate(transient_audio_frames):
        start = i * hop_length
        end = start + window_size
        transient_audio[start:end] += frame * window
        window_sum[start:end] += window

    # Normalize transient_audio by window_sum, avoiding division by zero
    window_sum[window_sum == 0] = 1
    transient_audio /= window_sum

    # Write the transient audio file to output folder
    sf.write(output_path + 'transient_' + audio_file.split('/')[-1], transient_audio, sr)

    # Calculate residual audio
    residual_audio = audio - transient_audio

    # Write the residual audio file to output folder
    sf.write(output_path + 'residual_' + audio_file.split('/')[-1], residual_audio, sr)

    if plot:
        # Plot the audio signal, the transient part of the audio, and the residual part of the audio
        utils.plot_transient_residual_audio(audio, transient_audio, residual_audio, sr)

    return audio, transient_audio, residual_audio, sr