import glob

import librosa
import numpy as np
import scipy
import scipy.io.wavfile as wav
from scipy.signal import stft

import AudioConverter as ac
import util

"""
An idea for a feature
"""

def calculate_abl(file):
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
    frequencies, times, spec = scipy.signal.stft(sig, fs=rate, window=window, nperseg=1024)

    # HPSS
    harmonic, percussive = librosa.decompose.hpss(spec)
    h_sum = np.sum(harmonic, axis=1)
    p_sum = np.sum(percussive, axis=1)

    # spec = np.multiply(spec[27:280, :], 0.001)
    # EQ the primary voice frequencies away
    spec = np.delete(spec, np.s_[27:280], axis=0)

    gradient = np.gradient(np.sum(spec, axis=1))
    gradient = np.sum(gradient) / spec.shape[1]
    gradient = gradient.real
    result = "Speech" if gradient > -1500 else "Music"
    print("Gradient: " + str(round(gradient, 2)) + " - " + result)

    a = 2

    # EQ the speech frequencies out (in the range 300Hz - 3000Hz)
    #spec = np.delete(spec, np.s_[27:280], axis=0)
    #np.multiply(spec[27:280, :], 0.001)

    # Binarize
    #spec = np.where(spec > 10, 1, 0)

    return gradient


def calculate_abls(path_speech, path_music, max_items):
    # -------------------------- Speech --------------------------
    speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
    sp_abls = []
    for file in range(max_items):
        print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
        result = calculate_abl(speech_files[file])
        sp_abls.append(result)

    print("Processed " + str(max_items) + " speech files.")

    sp_abls = np.array(sp_abls)
    sp_lbls = np.zeros(len(sp_abls))
    print("Average Gradient speech: " + str(np.average(sp_abls)))

    # -------------------------- Music --------------------------
    if path_music is None:
        return

    music_files = glob.glob(path_music + "/**/*.wav", recursive=True)  # list of files in given path
    mu_abls = []

    i = 0
    for file in range(max_items):
        print(str(file) + " of " + str(len(music_files)) + " - processing " + str(music_files[file]))
        result = calculate_abl(music_files[file])
        mu_abls.append(result)
        i += 1

    print("Processed " + str(max_items) + " music files.")

    # Convert CFA list into array for fitting
    mu_abls = np.array(mu_abls)
    # Create the labels for the music files
    mu_lbls = np.ones(len(mu_abls))

    # Concat arrays
    trn = np.concatenate((sp_abls, mu_abls))
    lbls = np.concatenate((sp_lbls, mu_lbls))

    print("Average Gradient speech: " + str(np.average(sp_abls)) + ", Average Gradient music: " + str(np.average(mu_abls)))

    return trn, lbls
