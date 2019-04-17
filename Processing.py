import glob
import time

import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
from numba import jit

import AudioConverter as ac
import util

@jit(cache=True)
def preprocessing_new(wav_path):
    (rate, signal) = scipy.io.wavfile.read(wav_path)
    sig = np.array(signal)

    # Convert signal to mono
    sig = util.stereo2mono(sig)

    # Apply noise gate
    noise_gate_level = 0.5
    sig[sig < noise_gate_level] = 0

    # Estimate the spectrogram using a Hanning window
    window = np.hanning(256)  # 1024 samples correspond to ~ 100ms

    # Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spectrogram = scipy.signal.stft(sig, fs=rate, window=window, nperseg=256)

    if rate < 11025:
        raise ValueError(
            "The sampling rate of the incoming signal is too low for Continuous Frequency Activation and GRAD processing.")

    # Cut the spectrogram to 11khz for cfa and grad processing
    # NOTE: Assuming that the frequencies are distributed linearly along the spectrogram
    upper_limit_idx = np.argmin(np.abs(frequencies - (11025 / 2)))
    spectrogram = spectrogram[:upper_limit_idx, :]

    return sig, rate, frequencies, times, spectrogram


@jit(cache=True)
def cfa_grad_filerate_preprocessing(file):
    """
    Filter rate preprocessing for CFA and GRAD
    :param file: The file to preprocess. The function checks if an 11kHz file exists and if not, converts the given file
    to an 11kHz MONO file.
    :return: the processed file or the original (11kHz file if it already exists)
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
def cfa_grad_preprocessing(file):
    start = time.time()
    file = cfa_grad_filerate_preprocessing(file)

    end = time.time()
    (rate, signal) = wav.read(file)
    sig = np.array(signal)

    # Convert signal to mono
    sig = util.stereo2mono(sig)

    # Apply noise gate
    noise_gate_level = 0.5
    sig[sig < noise_gate_level] = 0

    # Estimate the spectrogram using a Hanning window
    window = np.hanning(256)  # 1024 samples correspond to ~ 100ms

    # Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spectrogram = scipy.signal.stft(sig, fs=rate, window=window, nperseg=256)

    print("CFA_GRAD PREPROCESSING took ", str(end-start))
    return spectrogram

