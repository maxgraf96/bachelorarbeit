import sklearn
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import tensorflow as tf
import glob
import AudioConverter as ac
import soundfile as sf
from pyAudioAnalysis import audioBasicIO
import util

def train_mfcc_nn(path_speech, path_music, max_duration):

    # Calculate MFCCs
    trn, lbls = calculate_mfccs(path_speech, path_music, max_duration)

    # Preprocessing
    trn = sklearn.preprocessing.scale(trn, axis=1)
    pca = PCA(n_components=7)
    principal_components = pca.fit_transform(trn)

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

    return clf, pca

def predict_nn(clf, mfcc_in):
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
    sp_mfccs = []
    acc_duration = 0  # The accumulated length of processed files in seconds
    for i in range(len(speech_files)):
        duration = util.get_wav_duration(speech_files[i])
        if acc_duration + duration > max_duration:
            break
        acc_duration += duration

        print(str(i) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[i]))
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
        ac.mp3_to_wav(mp3s[i])

    # process music wavs
    wavs = glob.glob(path_music + "/**/*.wav", recursive=True)
    wav_mfccs = []
    acc_duration = 0  # reset processed length
    retries = 0  # retries for the skipping of files that are too long
    for i in range(len(wavs)):
        duration = util.get_wav_duration(speech_files[i])
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

    return trn, lbls


def read_mfcc(wav_file_path):
    (rate, sig) = wav.read(wav_file_path)
    # Convert signal to mono
    sig = audioBasicIO.stereo2mono(sig)
    mfcc_feat = mfcc(sig, rate, nfft=768, winlen=0.025)
    d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(sig, rate)

    return np.append(mfcc_feat, d_mfcc_feat, axis=1)
