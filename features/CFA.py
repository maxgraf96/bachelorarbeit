import glob
import time

import math

import numpy as np
import scipy
from numba import jit
from scipy.signal import stft

import Processing

"""
Continuous frequency activation feature via http://www.cp.jku.at/research/papers/Seyerlehner_etal_DAFx_2007.pdf
"""

@jit(nopython=True, cache=True)
def calculate_peakiness(peaks, minima, act_func):
    """
    Calculates the "peakiness" of an activation function. The higher the peakiness, the more frequencies were held out
    over a period of time.
    :param peaks: A numpy array containing the indices of the relative maxima of the activation function
    :param minima: A numpy array containing the indices of the relative minima of the activation function
    :param act_func: The activation function is a block of the spectrogram summed up along the y-axis
    :return: The "peakiness" of the activation function
    """

    pvvalues = []
    for peakidx in peaks:
        # find nearest local minimum to the left
        smaller_peak_indices = minima[minima < peakidx]
        search = np.abs(smaller_peak_indices - peakidx)
        if len(search) == 0:
            nearest_idx_l = 0
        else:
            nearest_idx_l = smaller_peak_indices[search.argmin()]
        # find nearest local minimum to the right
        greater_peak_indices = minima[minima > peakidx]
        search = np.abs(greater_peak_indices - peakidx)
        if len(search) == 0:
            nearest_idx_r = 0
        else:
            nearest_idx_r = greater_peak_indices[search.argmin()]

        xl = act_func[peakidx] - act_func[nearest_idx_l]
        xr = act_func[peakidx] - act_func[nearest_idx_r]
        height = min(xl, xr)
        width = peakidx - nearest_idx_l if xl < xr else nearest_idx_r - peakidx
        pv = height / width
        pvvalues.append(pv)

    pvvalues = np.array(pvvalues)
    pvvalues[::-1].sort()  # sort descending
    finals = pvvalues[0:5] # Take the 5 highest values for the final result
    peakiness = np.sum(finals)
    return peakiness


def calculate_from_spectrogram(spectrogram):
    """
    Calculates the CFA value from a spectrogram
    :param spectrogram: A numpy array containing the spectrogram data
    :return: The CFA value
    """

    # Binarize
    spectrogram = np.where(spectrogram > 10, 1, 0)

    # Create blocks consisting of 3 frames each
    no_blocks = math.ceil(spectrogram.shape[1] / 3)
    peakis = []  # in the end this list contains the peakiness values for all blocks
    for step in range(no_blocks):
        start = step * 3
        end = start + 3
        if end > spectrogram.shape[1]:
            end = spectrogram.shape[1]
        block = spectrogram[:, start:end]

        # Compute the frequency activation function
        act_func = np.sum(block, axis=1) / block.shape[1]

        # Detect strong peaks
        peaks = scipy.signal.argrelextrema(act_func, np.greater)[0]  # [0] because we only want the indices
        minima = scipy.signal.argrelextrema(act_func, np.less)[0]
        peakis.append(calculate_peakiness(peaks, minima, act_func))


    result = np.sum(peakis) / len(peakis)
    return result

def calculate_cfa(file=None, spec=None):

    tstart = time.time()

    # Get the spectrogram
    if file is None and spec is None:
        raise ValueError("Either a file for conversion or a spectrogram must be passed to this function.")
    elif file is not None and spec is None:
        spec = Processing.cfa_preprocessing(file)

    result = calculate_from_spectrogram(spec)

    tend = time.time()
    #print("CFA extrema calculation: ", str(tend - tstart))

    return result


def calculate_cfas(path_speech, path_music, max_cfas):
    # -------------------------- Speech --------------------------
    speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
    sp_cfas = []
    for file in range(max_cfas):
        print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
        result = calculate_cfa(speech_files[file])
        sp_cfas.append(result)

    print("Processed " + str(max_cfas) + " speech files.")

    sp_cfas = np.array(sp_cfas)
    sp_lbls = np.zeros(len(sp_cfas))
    print("Average CFA speech: " + str(np.average(sp_cfas)))

    # -------------------------- Music --------------------------
    if path_music is None:
        return

    music_files = glob.glob(path_music + "/**/*.wav", recursive=True)  # list of files in given path
    mu_cfas = []

    # In this case iterate as long as the number of cfas of both classes are the same as the cfa result
    # is not linked to the duration of the file -> we need the same number of cfas so that the classifier is
    # not biased
    i = 0
    for file in range(max_cfas):
        print(str(file) + " of " + str(len(music_files)) + " - processing " + str(music_files[file]))
        result = calculate_cfa(music_files[file])
        mu_cfas.append(result)
        i += 1

    print("Processed " + str(max_cfas) + " music files.")

    # Convert CFA list into array for fitting
    mu_cfas = np.array(mu_cfas)
    # Create the labels for the music files
    mu_lbls = np.ones(len(mu_cfas))

    # Concat arrays
    trn = np.concatenate((sp_cfas, mu_cfas))
    lbls = np.concatenate((sp_lbls, mu_lbls))

    print("Average CFA speech: " + str(np.average(sp_cfas)) + ", Average CFA music: " + str(np.average(mu_cfas)))

    return trn, lbls

