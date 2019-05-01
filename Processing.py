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

import glob
import time

import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
from numba import jit

import AudioConverter as ac
import util

@jit(cache=True)
def preprocessing(wav_path):
    """
    Preprocessing for both MFCC and CFA calculation.
    :param wav_path: The (absolute or relative) path to the file to preprocess
    :return: The signal converted to mono,
        the sample rate of the signal,
        the frequencies of the spectrogram,
        the times of the spectrogram,
        the spectrogram itself
    """
    (rate, signal) = scipy.io.wavfile.read(wav_path)
    sig = np.array(signal)

    # Convert signal to mono
    sig = util.stereo2mono(sig)

    # Estimate the spectrogram using a Hanning window
    window = np.hanning(256)  # 256 samples correspond to ~ 25ms

    # Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spectrogram = scipy.signal.stft(sig, window=window, nperseg=256)

    if rate < 11025:
        raise ValueError(
            "The sampling rate of the incoming signal is too low for Continuous Frequency Activation processing.")

    # Cut the spectrogram to 11khz for cfa processing
    # NOTE: Assuming that the frequencies are distributed linearly along the spectrogram
    upper_limit_idx = np.argmin(np.abs(frequencies - (11025 / 2)))
    spectrogram = spectrogram[:upper_limit_idx, :]

    return sig, rate, frequencies, times, spectrogram


@jit(cache=True)
def cfa_filerate_preprocessing(file):
    """
    Filter rate preprocessing for CFA
    :param file: The file to preprocess. The function checks if an 11kHz file exists and if not, converts the given file
    to an 11kHz MONO file.
    :return: the processed file or the original file (if it already exists and was 11kHz)
    """
    # Read the audio file and cut off all frequencies > 11kHz
    # Check if a converted wav is present before converting to 11kHz wav
    if ".mp3" in file:
        file = ac.mp3_to_11_khz_wav(file)
        return file

    converted_path = file[:-4] + "_11_kHz.wav"
    if "11_kHz" in file:
        pass
    elif "11_kHz" not in file and not glob.glob(converted_path):
        file = ac.wav_to_11_khz(file)
    else:
        file = converted_path

    return file


@jit(cache=True)
def cfa_preprocessing(wav_path):
    """
    Processing specific for the CFA feature.
    :param wav_path: The (absolute or relative) path to the wav file to preprocess
    :return: The spectrogram of the mono signal
    """
    wav_path = cfa_filerate_preprocessing(wav_path)
    (rate, signal) = wav.read(wav_path)
    sig = np.array(signal)

    # Convert signal to mono
    sig = util.stereo2mono(sig)

    # Estimate the spectrogram using a Hanning window
    window = np.hanning(256)  # 256 samples correspond to ~ 25ms

    # Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spectrogram = scipy.signal.stft(sig, window=window, nperseg=256)

    # print("CFA PREPROCESSING took ", str(end-start))
    return spectrogram
