# Run onset detection using the transient audio signal output from separate.py

import numpy as np
from separate_transient import separate_transient
import librosa
import matplotlib.pyplot as plt

input_path = 'audio/'
output_path = 'output/'
audio_file = input_path + '463715__carloscarty__quena-bamboo-flute-latin-loop-06-100bpm-a.wav'

# TODO: Play with ODF and peak-picking parameters

if __name__ == '__main__':
    audio, transient_audio, residual_audio, sr = separate_transient(audio_file, output_path, plot=False)

    odf_original_spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets_from_original = librosa.onset.onset_detect(y=audio, sr=sr, 
                                                      onset_envelope=odf_original_spectral_flux, units='time')

    odf_transient_spectral_flux = librosa.onset.onset_strength(y=transient_audio, sr=sr)
    onsets_from_transient = librosa.onset.onset_detect(y=transient_audio, sr=sr,
                                                       onset_envelope=odf_transient_spectral_flux, units='time')

    print(onsets_from_original)
    print(onsets_from_transient)
    
    # Plot the input audio signal and transient with onsets overlayed
    plt.figure(figsize=(10, 6))
    t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    plt.subplot(2, 1, 1)
    plt.plot(t, audio, label='Audio Signal')
    plt.vlines(onsets_from_original, -1, 1, color='r', linestyle='--', label='Onsets')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal with Onsets')
    plt.subplot(2, 1, 2)
    plt.plot(t, transient_audio, label='Transient Audio')
    plt.vlines(onsets_from_transient, -1, 1, color='r', linestyle='--', label='Onsets')
    plt.show()