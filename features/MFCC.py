import glob

import joblib
import numpy as np
import scipy.io.wavfile as wav
import sklearn
import tensorflow as tf
from numba import jit
from python_speech_features import delta
from python_speech_features import mfcc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import AudioConverter as ac
import util


def train_mfcc_nn(path_speech, path_music, max_duration):

    # Calculate MFCCs
    if len(glob.glob(util.ext_hdd_path + "data/mfcc_trn.joblib")) < 1:
        trn, lbls = calculate_mfccs(path_speech, path_music, max_duration)
        joblib.dump(trn, util.ext_hdd_path + "data/mfcc_trn.joblib")
        joblib.dump(lbls, util.ext_hdd_path + "data/mfcc_lbls.joblib")
    else:
        trn = joblib.load(util.ext_hdd_path + "data/mfcc_trn.joblib")
        lbls = joblib.load(util.ext_hdd_path + "data/mfcc_lbls.joblib")

    # Preprocessing
    scaler = sklearn.preprocessing.StandardScaler()
    trn = scaler.fit_transform(trn)
    #trn = sklearn.preprocessing.scale(trn)
    #pca = PCA(n_components=13)
    #prcomp = pca.fit_transform(trn)

    # Classifier fitting
    # Tensorflow nn
    clf = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 26)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    clf.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])



    trn = trn.reshape((trn.shape[0], 1, trn.shape[1]))
    clf.fit(trn, lbls, epochs=5)

    return clf, scaler  #, pca

def predict_nn(clf, scaler, mfcc_in):
    mfcc_in = scaler.fit_transform(mfcc_in)
    mfcc_in = mfcc_in.reshape((mfcc_in.shape[0], 1, mfcc_in.shape[1]))
    prediction = clf.predict(mfcc_in)
    result = np.greater(prediction[:, 1], prediction[:, 0])
    return result.astype(int)

def train_mfcc_knn(path_speech, path_music, max_duration):
    """
    Trains a KNN with the given label (0 = speech, 1 = music) and all the files within the given path.
    If MP3 files are found, they are converted to 16kHz wav files before processing
    :param path_speech: The path to the speech files
    :param path_music: The path to the music files
    :param max_duration: How many files in the folder should be processed before termination (in seconds)
    :return: The trained classifier and PCA object
    """

    # Calculate MFCCs
    trn, lbls = calculate_mfccs(path_speech, path_music, max_duration)


    # Preprocessing
    trn = sklearn.preprocessing.scale(trn, axis=1)
    pca = PCA(n_components=7)
    principal_components = pca.fit_transform(trn)

    # Classifier fitting
    # knn
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(principal_components, lbls)
    return [clf, pca]

def calculate_mfccs(path_speech, path_music, max_duration):
    # -------------------------- Speech --------------------------
    speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
    # Remove 11 kHz files from list
    speech_files = [file for file in speech_files if "11_kHz.wav" not in file]
    processed_files = []
    sp_mfccs = []
    acc_duration = 0  # The accumulated length of processed files in seconds
    for i in range(len(speech_files)):
        if speech_files[i] in processed_files:
            continue
        # Convert speech file to 16 kHz if necessary
        if "16_kHz.wav" not in speech_files[i] and speech_files[i][:-4] + "_16_kHz.wav" not in speech_files:
            processed_files.append(speech_files[i])  # Also append unconverted file so the file isn't processed twice
            print("Converting ", speech_files[i], " to 16kHz wav...")
            speech_files[i] = ac.wav_to_16_khz(speech_files[i])

        # Continue if the file was already converted before
        elif "16_kHz.wav" not in speech_files[i] and speech_files[i][:-4] + "_16_kHz.wav" in speech_files:
            continue

        duration = util.get_wav_duration(speech_files[i])

        # Break if the duration of processed files exceeds the maximum specified duration
        if acc_duration + duration > max_duration:
            break
        acc_duration += duration

        print(str(i) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[i]))
        # Append the current file to the list of processed files in order to avoid double processing of the same file
        processed_files.append(speech_files[i])
        # Read the current MFCCs
        current_mfccs = read_mfcc(speech_files[i])
        for j in range(len(current_mfccs)):
            sp_mfccs.append(current_mfccs[j])

    print("Processed " + str(round(acc_duration, 2)) + " minutes of speech data.")
    print("Got " + str(len(sp_mfccs)) + " speech MFCCs.")
    len_sp_mfccs = len(sp_mfccs)

    sp_mfccs = np.array(sp_mfccs)
    sp_lbls = np.zeros(len(sp_mfccs))

    # -------------------------- Music --------------------------
    mp3s = glob.glob(path_music + "/**/*.mp3", recursive=True)  # list of mp3s in given path
    # convert mp3s to wavs
    for i in range(len(mp3s)):
        # Only convert if the converted file doesn't exist yet
        if len(glob.glob(mp3s[i][:-4] + ".wav")) > 0:
            print("Found wav file, skipping...")
            continue

        print("Converting " + str(mp3s[i]) + " to wav.")
        ac.mp3_to_16_khz_wav(mp3s[i])

    # process music wavs
    wavs = glob.glob(path_music + "/**/*.wav", recursive=True)
    # Remove 11 kHz files from list
    wavs = [file for file in wavs if "11_kHz.wav" not in file]
    processed_files = []
    wav_mfccs = []
    acc_duration = 0  # reset processed length
    retries = 0  # retries for the skipping of files that are too long
    for i in range(len(wavs)):
        if wavs[i] in processed_files:
            continue

        # Convert wav file to 16 kHz if necessary
        if "16_kHz.wav" not in wavs[i] and wavs[i][:-4] + "_16_kHz.wav" not in wavs:
            processed_files.append(wavs[i])  # Also append unconverted file so the file isn't processed twice
            print("Converting ", wavs[i], " to 16kHz wav...")
            wavs[i] = ac.wav_to_16_khz(wavs[i])

        # Continue if the file was already converted before
        elif "16_kHz.wav" not in wavs[i] and wavs[i][:-4] + "_16_kHz.wav" in wavs:
            continue

        duration = util.get_wav_duration(speech_files[i])
        if acc_duration + duration > max_duration:
            break
        acc_duration += duration

        print(str(i) + " of " + str(len(wavs)) + " - processing " + str(wavs[i]))
        processed_files.append(wavs[i])
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

    return trn, lbls

@jit(cache=True)
def read_mfcc(wav_file_path):
    (rate, sig) = wav.read(wav_file_path)
    # Convert signal to mono
    sig = util.stereo2mono(sig)
    mfcc_feat = mfcc(sig, rate, nfft=1386, winlen=0.025)
    d_mfcc_feat = delta(mfcc_feat, 2)

    return np.append(mfcc_feat, d_mfcc_feat, axis=1)

@jit(cache=True)
def read_mfcc_new(sig, rate):
    mfcc_feat = mfcc(sig, rate, nfft=1386, winlen=0.025)
    d_mfcc_feat = delta(mfcc_feat, 2)

    return np.append(mfcc_feat, d_mfcc_feat, axis=1)