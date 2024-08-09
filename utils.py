# Utility functions

import numpy as np
import copy
import matplotlib.pyplot as plt

def trim_audio(audio):
    """
    Trim audio to the nearest power of 2
        Input: audio (np.array) - audio signal
        Output: audio (np.array) - trimmed audio signal
    """
    return audio[:2**int(np.log2(len(audio)))]

def calculate_regularity_measure(coeffs, s=0, p=2):
    """
    Calculate modulus of regularity at each leaf of coefficient tree
        Input: coeffs (list) - list of multilevel DWT coefficients
                s (int) - weighting of scale/level of DWT coefficients
                p (int) - power of the modulus to focus on different types of transients
        Output: k (np.array) - regularity measure at each leaf of coefficient tree
    """
    maxlevel = len(coeffs)
    assert len(coeffs[0]) == 1, "The first level of the coefficients should have only one element"
    assert all([len(coeffs[i]) == 2*len(coeffs[i-1]) for i in range(2, maxlevel)]), "The number of coefficients" + \
          "at each level should be double the previous level"

    # create a copy of the coeffs list
    k = copy.deepcopy(coeffs)

    # 0th level
    k[1][0] += (np.abs(k[0][0]) ** p)

    # 1st level to maxlevel-1
    for i in range(1, len(coeffs)-1):
        for j in range(len(coeffs[i])):
            add = ( (np.abs(k[i][j]) ** p) * (2**(i*s)) )
            k[i+1][2*j] += add
            k[i+1][2*j+1] += add

    # Return the last level (leaves of the coefficient tree) as a numpy array
    k = np.array(k[maxlevel-1])
    return k

def prune_coefficient_tree(coeffs, k, threshold=1.5):
    """
    Prune the coefficient tree based on the thresholded regularity measure
        Input: coeffs (list) - list of multilevel DWT coefficients
                k (np.array) - regularity measure at each leaf of coefficient tree
                threshold (float) - threshold value for pruning
        Output: coeffs (list) - pruned list of multilevel DWT coefficients
    """
    assert len(coeffs) == len(k), "Number of frames in coeffs and k should be the same"
    assert len(coeffs[0][-1]) == len(k[0]), "Number of leaves in the coefficient tree and k should be the same"
    # Calculate the mean of the regularity measure for each frame
    mean_k = np.mean(k, axis=1)
    # Max pool the means across every 3 frames
    mean_k = np.concatenate(([np.max(mean_k[:2])],
                             [np.max(mean_k[i-1:i+2]) for i in range(1,len(mean_k)-1)],
                             [np.max(mean_k[-2:])]))
    # Loop across the frames
    for frame_number in range(len(k)):
        # Loop across the leaves
        for leaf_number, leaf_value in enumerate(k[frame_number]):
            # Set leaves of pruned_coeffs to 0 if value of k is less than threshold * mean_k for the frame
            if leaf_value < threshold * mean_k[frame_number]:
                coeffs[frame_number][-1][leaf_number] = 0
        # Loop across the levels of the coefficient tree, stopping at the 2nd level
        for level in range(len(coeffs[frame_number])-2, 1, -1):
            # Loop across the coefficients at each level
            for coeff_number in range(len(coeffs[frame_number][level])):
                # If the two parent values are 0, set the coefficient to 0
                if coeffs[frame_number][level+1][2*coeff_number] == 0 and \
                   coeffs[frame_number][level+1][2*coeff_number+1] == 0:
                    coeffs[frame_number][level][coeff_number] = 0
        # If 2nd level coeff is 0, set 1st level coeff to 0
        if coeffs[frame_number][1] == 0:
            coeffs[frame_number][0] = 0
    return coeffs

def plot_transient_residual_audio(audio, transient_audio, residual_audio, sr):
    """
    Plot the audio signal, the transient part of the audio, and the residual part of the audio
        Input: audio (np.array) - original audio signal
               transient_audio (np.array) - estimated transient part of the audio signal
               residual_audio (np.array) - residual part of the audio signal
               sr (int) - sampling rate of audio signal
    """
    t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(t, audio)
    plt.title('Original Signal')
    plt.subplot(3, 1, 2)
    plt.plot(t, transient_audio)
    plt.title('Transient Audio')
    plt.subplot(3, 1, 3)
    plt.plot(t, residual_audio)
    plt.title('Residual Audio')
    plt.tight_layout()
    plt.show()