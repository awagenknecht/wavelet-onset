# wavelet-onset
Work-in-progress repository for wavelet-based methods of transient estimation and onset detection in audio signals.

One such method for transient estimation comes from the following paper:
Laurent Daudet, Stéphane Molla, Bruno Torrésani. Transient detection and encoding using wavelet
coefficient trees. Colloques sur le Traitement du Signal et des Images, Sep 2001, Toulouse, France. hal-01306501

This approach is summarized by the following steps.
1. Estimate and remove the harmonic component from an input audio signal.
2. Take a Discrete Wavelet Transform respresentation across frames of the remaining part of the audio.
3. Prune the dyadic tree of the DWT representation for each frame.
    a. Consider the leaves of the tree, which are coefficients at the highest level of decomposition and finest time localization.
    b. For each leaf, calculate a regularity measure by summing powers of all coefficients in the branch.
    c. Remove branches for which the regularity measure at the leaf is less than a defined threshold.
4. Reconstruct an audio signal from the pruned DWT trees, which serves as an estimate of the transient portion of the original audio signal.