import glob
import math

import scipy

import numpy as np
import sklearn
import tensorflow as tf
from sklearn.decomposition import PCA
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy.signal import stft

import util
from pyAudioAnalysis import audioBasicIO
import AudioConverter as ac

"""
Continuous frequency activation feature via http://www.cp.jku.at/research/papers/Seyerlehner_etal_DAFx_2007.pdf
"""


def train_cfa_nn(path_speech, path_music, max_duration):

    # Calculate CFA features
    trn, lbls = calculate_cfas(path_speech, path_music, max_duration)

    # Preprocessing
    trn = sklearn.preprocessing.scale(trn, axis=1)

    # Classifier fitting
    # Tensorflow nn
    clf = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 26)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    clf.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    trn = trn.reshape((trn.shape[0], 1, trn.shape[1]))
    clf.fit(trn, lbls, epochs=3)

    return clf

def predict_nn(clf, mfcc_in):
    mfcc_in = mfcc_in.reshape((mfcc_in.shape[0], 1, mfcc_in.shape[1]))
    prediction = clf.predict(mfcc_in)
    result = np.greater(prediction[:, 1], prediction[:, 0])
    return result.astype(int)

def calculate_cfa(file, threshold):
    # The CFA calculation is subidivided into several steps
    # 1: Read the audio file and cut off all frequencies > 11kHz
    # Check if a converted wav is present before converting to 11kHz wav
    if ".mp3" in file:
        file = ac.mp3_to_11_khz_wav(file)

    converted_path = file[:-4] + "_11_kHz.wav"
    if "11_kHz" in file:
        pass;
    elif not "11_kHz" in file and not glob.glob(converted_path):
        file = ac.wav_to_11_khz(file)
    else:
        file = converted_path
    (rate, signal) = wav.read(file)
    sig = np.array(signal)
    # Convert signal to mono
    sig = audioBasicIO.stereo2mono(sig)
    # 2: Apply noise gate
    noise_gate_level = 0.005
    sig[sig < noise_gate_level] = 0

    # 3: Estimate the spectrogram using a Hanning window
    window = np.hanning(1024)  # 1024 samples correspond to ~ 100ms

    # 4: Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spec = scipy.signal.stft(sig, window=window, nperseg=1024)
    # N = 21
    # frames = spec.shape[1]
    # for i in range(frames):
    #
    #     for j in range(spec.shape[0]):
    #         k = 0 if j < 10 else 10
    #         l = 10 if j + 10 < spec.shape[0] else spec.shape[0] - j
    #         current_sum = np.sum(spec[(j - k):(j + l), i])
    #         spec[j, i] = spec[j, i] - ((1 / N) * current_sum)

    # 5: Binarize
    spec = np.where(spec > 0.1, 1, 0)

    # 6: Create blocks consisting of 100 frames each with 50 blocks overlap
    no_blocks = math.ceil(spec.shape[1] / 50)
    blocks = []
    peakis = []  # in the end this list contains the peakiness values for all blocks
    for step in range(no_blocks):
        start = step * 50
        end = start + 100
        if end > spec.shape[1]:
            end = spec.shape[1]
        block = spec[:, start:end]
        blocks.append(block)

        # Compute the frequency activation function for each block
        act_func = np.sum(block, axis=1) / block.shape[1]

        # Detect strong peaks
        peaks = scipy.signal.find_peaks(act_func)[0]  # [0] because we only want the indices
        minima = scipy.signal.argrelextrema(act_func, np.less)[0]
        pvvalues = []
        for peakidx in peaks:
            # find nearest local minimum to the left
            search = np.abs(minima[minima < peakidx] - peakidx)
            if len(search) == 0:
                nearest_idx_l = 0
                # print("Len = 0 L")
            else:
                nearest_idx_l = search.argmin()
            # find nearest local minimum to the right
            search = np.abs(minima[minima > peakidx] - peakidx)
            if len(search) == 0:
                nearest_idx_r = 0
                # print("Len = 0 R")
            else:
                nearest_idx_r = search.argmin()

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
    if result < threshold:
        print("speech")
    else:
        print("music")
    return peakis

def calculate_cfas_music(path_music, max_duration, threshold):
    # -------------------------- Music --------------------------
    music_files = glob.glob(path_music + "/**/*.wav", recursive=True)  # list of files in given path
    sp_cfas = []
    acc_duration = 0  # The accumulated length of processed files in seconds
    for file in range(len(music_files)):
        duration = util.get_wav_duration(music_files[file])
        if acc_duration + duration > max_duration:
            break
        acc_duration += duration

        print(str(file) + " of " + str(len(music_files)) + " - processing " + str(music_files[file]))
        peakis = calculate_cfa(music_files[file], threshold)


def calculate_cfas(path_speech, path_music, max_duration):
    # -------------------------- Speech --------------------------
    speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
    sp_cfas = []
    acc_duration = 0  # The accumulated length of processed files in seconds
    for file in range(len(speech_files)):
        duration = util.get_wav_duration(speech_files[file])
        if acc_duration + duration > max_duration:
            break
        acc_duration += duration

        print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
        peakis = calculate_cfa(speech_files[file])
        a = 2


    #for j in range(len(current_mfccs)):
    #    sp_mfccs.append(current_mfccs[j])

    print("Processed " + str(round(acc_duration, 2) / 60) + " minutes of speech data.")
    #print("Got " + str(len(sp_mfccs)) + " speech MFCCs.")

    #len_sp_mfccs = len(sp_mfccs)
    #sp_mfccs = np.array(sp_mfccs)
    #sp_lbls = np.zeros(len(sp_mfccs))

    # -------------------------- Music --------------------------
    """
    mp3s = glob.glob(path_music + "/**/*.mp3", recursive=True)  # list of mp3s in given path

    # convert mp3s to wavs
    for i in range(len(mp3s)):
        # Only convert if the converted file doesn't exist yet
        if len(glob.glob(mp3s[i][:-4] + ".wav")) > 0:
            print("Found wav file, skipping...")
            continue

        print("Converting " + str(mp3s[i]) + " to wav.")
        ac.mp3_to_wav(mp3s[i])

    # process music wavs
    wavs = glob.glob(path_music + "/**/*.wav", recursive=True)
    wav_mfccs = []
    acc_duration = 0  # reset processed length
    retries = 0  # retries for the skipping of files that are too long
    for i in range(len(wavs)):
        duration = get_wav_duration(speech_files[i])
        if acc_duration + duration > max_duration:
            break
        acc_duration += duration

        print(str(i) + " of " + str(len(wavs)) + " - processing " + str(wavs[i]))
        current_mfccs = read_mfcc(wavs[i])

        # Only append the MFCCs if the number of current MFCCs + the number of MFCCs in the currently processed file
        # do not exceed the total number of MFCCs in the speech set.
        # This guarantees that the numbers of MFCCs in the speech and music sets are always ~ the same
        if len(wav_mfccs) + len(current_mfccs) > len_sp_mfccs:
            # Use 'continue' and not 'break' because that way it migth be possible that a shorter file is
            # found that can 'fill the remaining-MFCC-gap'
            # Break after 15 tries
            if retries > 15:
                break
            retries += 1
            continue
        for j in range(len(current_mfccs)):
            wav_mfccs.append(current_mfccs[j])

    print("Processed " + str(round(acc_duration, 2)) + " minutes of music data.")

    # Convert MFCC list into array for fitting
    mu_mfccs = np.array(wav_mfccs)
    # Create the labels for the music files
    mu_lbls = np.ones(len(wav_mfccs))

    print("Got " + str(len(sp_mfccs)) + " speech MFCCs and " + str(len(mu_mfccs)) + " music MFCCs.")

    # Concat arrays
    trn = np.concatenate((sp_mfccs, mu_mfccs))
    lbls = np.concatenate((sp_lbls, mu_lbls))
    """

    #return trn, lbls
    return 0, 1
