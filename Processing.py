import glob

import numpy as np
import scipy.signal
import scipy.io.wavfile as wav

import AudioConverter as ac
import util


def cfa_abl_preprocessing(file):
    # Read the audio file and cut off all frequencies > 11kHz
    # Check if a converted wav is present before converting to 11kHz wav
    if ".mp3" in file:
        file = ac.mp3_to_11_khz_wav(file)

    converted_path = file[:-4] + "_11_kHz.wav"
    if "11_kHz" in file:
        pass
    elif "11_kHz" not in file and not glob.glob(converted_path):
        file = ac.wav_to_11_khz(file)
    else:
        file = converted_path
    (rate, signal) = wav.read(file)
    sig = np.array(signal)

    # Convert signal to mono
    sig = util.stereo2mono(sig)

    # Apply noise gate
    noise_gate_level = 0.5
    sig[sig < noise_gate_level] = 0

    # Estimate the spectrogram using a Hanning window
    window = np.hanning(1024)  # 1024 samples correspond to ~ 100ms

    # Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spectrogram = scipy.signal.stft(sig, fs=rate, window=window, nperseg=1024)

    return spectrogram

