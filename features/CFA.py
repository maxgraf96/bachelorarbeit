# Copyright (c) 2019 Max Graf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import scipy
from numba import jit

import Processing

"""
Continuous frequency activation feature as described in http://www.cp.jku.at/research/papers/Seyerlehner_etal_DAFx_2007.pdf
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

    # List of the calculated peaks
    pvvalues = []
    for peakidx in peaks:
        # Find nearest local minimum to the left
        smaller_peak_indices = minima[minima < peakidx]
        search = np.abs(smaller_peak_indices - peakidx)
        if len(search) == 0:
            nearest_idx_l = 0
        else:
            nearest_idx_l = smaller_peak_indices[search.argmin()]
        # Find nearest local minimum to the right
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


def calculate_from_spectrogram(spectrogram, threshold):
    """
    Calculates the CFA value from a spectrogram
    :param spectrogram: A numpy array containing the spectrogram data
    :param threshold: The threshold value for the distinction of the two classes
    :return: The CFA value
    """

    # Binarize
    spectrogram = np.where(spectrogram > 10, 1, 0)

    # Create blocks
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
        peaks = scipy.signal.argrelextrema(act_func, np.greater)[0]  # [0] because only the indices are necessary
        minima = scipy.signal.argrelextrema(act_func, np.less)[0]
        peakis.append(calculate_peakiness(peaks, minima, act_func))

    # Binarize the values of the summated peaks
    result = np.where(np.array(peakis) > threshold, 1, 0)
    # The final result is the percentage of detected "music"
    result = np.count_nonzero(result) / len(result)
    return result, np.array(peakis)

def calculate_cfa(file=None, spec=None, threshold=3.25):
    """
    Calculates the CFA value for either a file or a given spectrogram.
    :param file: The audio file for which the CFA value should be calculated. Can be None iff a spectrogram is specified.
    :param spec: The spectrogram from which to calculate the CFA value. Can be None iff a file is specified.
    :param threshold: The threshold against which the final value should be compared.
    :return: The classification result and the list of the peaks in the activation function
    """

    # Get the spectrogram
    if file is None and spec is None:
        raise ValueError("Either a file for conversion or a spectrogram must be passed to this function.")
    elif file is not None and spec is None:
        spec = Processing.cfa_preprocessing(file)

    result, peakis = calculate_from_spectrogram(spec, threshold)
    return result, peakis


# def calculate_cfas(path_speech, path_music, max_cfas):
#     """
#     CURRENTLY NOT IN USE. Could be used to train a classifier on CFA values, which has been omitted because the threshold
#     comparison has proven to be both faster and more accurate.
#     Calculates CFA feature values. There are two termination conditions for this process.
#     1. The processed data reaches the max_cfas.
#     2. All files in the "path_speech" folder or "path_music" folder have been processed.
#     :param path_speech: The (absolute or relative) path to the speech training data.
#     :param path_music: The (absolute or relative) path to the music training data.
#     :param max_cfas: The number of CFAs to calculate
#     :return: The CFA values and corresponding labels
#     """
#
#     # -------------------------- Speech --------------------------
#     speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
#     sp_cfas = []
#     for file in range(max_cfas):
#         print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
#         result = calculate_cfa(speech_files[file])
#         sp_cfas.append(result)
#
#     print("Processed " + str(max_cfas) + " speech files.")
#
#     sp_cfas = np.array(sp_cfas)
#     sp_lbls = np.zeros(len(sp_cfas))
#     print("Average CFA speech: " + str(np.average(sp_cfas)))
#
#     # -------------------------- Music --------------------------
#     if path_music is None:
#         return
#
#     music_files = glob.glob(path_music + "/**/*.wav", recursive=True)  # list of files in given path
#     mu_cfas = []
#
#     # In this case iterate as long as the number of cfas of both classes are the same as the cfa result
#     # is not linked to the duration of the file -> we need the same number of cfas so that the classifier is
#     # not biased
#     i = 0
#     for file in range(max_cfas):
#         print(str(file) + " of " + str(len(music_files)) + " - processing " + str(music_files[file]))
#         result = calculate_cfa(music_files[file])
#         mu_cfas.append(result)
#         i += 1
#
#     print("Processed " + str(max_cfas) + " music files.")
#
#     # Convert CFA list into array for fitting
#     mu_cfas = np.array(mu_cfas)
#     # Create the labels for the music files
#     mu_lbls = np.ones(len(mu_cfas))
#
#     # Concat arrays
#     trn = np.concatenate((sp_cfas, mu_cfas))
#     lbls = np.concatenate((sp_lbls, mu_lbls))
#
#     print("Average CFA speech: " + str(np.average(sp_cfas)) + ", Average CFA music: " + str(np.average(mu_cfas)))
#
#     return trn, lbls

