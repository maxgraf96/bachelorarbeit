import glob
import math

import numpy as np
import scipy
import scipy.io.wavfile as wav
import tensorflow as tf
from scipy.signal import stft

import AudioConverter as ac
import util

"""
Continuous frequency activation feature via http://www.cp.jku.at/research/papers/Seyerlehner_etal_DAFx_2007.pdf
"""

def calculate_cfa(file):
    # The CFA calculation is subidivided into several steps

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

    # N = 21
    # for j in range(spec.shape[0]):
    #     k = 0 if j < 10 else 10
    #     l = 10 if j + 10 < spec.shape[0] else spec.shape[0] - j
    #     current_sum = np.sum(spec[(j - k):(j + l), :], axis=0)
    #     spec[j, :] = spec[j, :] - ((1 / N) * current_sum)

    # EQ the speech frequencies out (in the range 300Hz - 3000Hz)
    #spec = np.delete(spec, np.s_[27:280], axis=0)
    np.multiply(spec[27:280, :], 0.001)

    # Binarize
    spec = np.where(spec > 10, 1, 0)

    # Create blocks consisting of 100 frames each with 50 blocks overlap
    no_blocks = math.ceil(spec.shape[1] / 25)
    blocks = []
    peakis = []  # in the end this list contains the peakiness values for all blocks
    for step in range(no_blocks):
        start = step * 25
        end = start + 50
        if end > spec.shape[1]:
            end = spec.shape[1]
        block = spec[:, start:end]
        blocks.append(block)

        # Compute the frequency activation function for each block
        act_func = np.sum(block, axis=1) / block.shape[1]

        # Detect strong peaks
        peaks = scipy.signal.argrelextrema(act_func, np.greater)[0]  # [0] because we only want the indices
        minima = scipy.signal.argrelextrema(act_func, np.less)[0]
        pvvalues = []
        for peakidx in peaks:
            # find nearest local minimum to the left
            smaller_peak_indices = minima[minima < peakidx]
            search = np.abs(smaller_peak_indices - peakidx)
            if len(search) == 0:
                nearest_idx_l = 0
                # print("Len = 0 L")
            else:
                nearest_idx_l = smaller_peak_indices[search.argmin()]
            # find nearest local minimum to the right
            greater_peak_indices = minima[minima > peakidx]
            search = np.abs(greater_peak_indices - peakidx)
            if len(search) == 0:
                nearest_idx_r = 0
                # print("Len = 0 R")
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
        finals = pvvalues[0:5]
        peakiness = np.sum(finals)
        peakis.append(peakiness)

    result = np.sum(peakis) / len(peakis)
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

def train_cfa_nn(path_speech, path_music, max_cfas):

    # Calculate CFA features
    trn, lbls = calculate_cfas(path_speech, path_music, max_cfas)

    # Classifier fitting
    # Tensorflow nn
    clf = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    clf.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    trn = trn.reshape((trn.shape[0], 1, 1))
    clf.fit(trn, lbls, epochs=3)

    return clf

def predict_nn(clf, cfa):
    cfa = cfa.reshape((1, 1, 1))
    prediction = clf.predict(cfa)
    return prediction

