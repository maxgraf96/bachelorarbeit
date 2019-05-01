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

import joblib
import numpy as np
import scipy.io.wavfile as wav
import sklearn
import tensorflow as tf
from numba import jit

import AudioConverter as ac
import util
from features.MFCC_base import mfcc, delta


def train_mfcc_nn(path_speech, path_music, max_duration):
    """
    Trains the MFCC classifier, a Keras neural network
    :param path_speech: The path to the speech data
    :param path_music: The path to the music data
    :param max_duration: The total duration of files that should be selected of each class. For example, 5000 would
    train the network with 5000 minutes of speech files and 5000 minutes of music files
    :return: The trained classifier and the scaler used to scale the training data
    """

    # Use existing training data (= extracted MFCCs). This is to skip the process of recalculating the MFCC each time.
    # WARNING: If the MFCC calculation method is changed or the training data changes the joblib files must be deleted manually.
    if len(glob.glob(util.data_path + "mfcc_trn.joblib")) < 1:
        trn, lbls = calculate_mfccs(path_speech, path_music, max_duration)
        joblib.dump(trn, util.data_path + "mfcc_trn.joblib")
        joblib.dump(lbls, util.data_path + "mfcc_lbls.joblib")
    else:
        trn = joblib.load(util.data_path + "mfcc_trn.joblib")
        lbls = joblib.load(util.data_path + "mfcc_lbls.joblib")

    # Scale the training data
    scaler = sklearn.preprocessing.StandardScaler()
    trn = scaler.fit_transform(trn)

    # Prepare the model
    clf = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 26)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # The reshaping of data is necessary for the use of this network
    trn = trn.reshape((trn.shape[0], 1, trn.shape[1]))
    # Fit the classifier
    clf.fit(trn, lbls, epochs=5)

    return clf, scaler

def predict_nn(clf, scaler, mfcc_in):
    """
    Uses the trained classifier to predict if one or more given MFCC are of the type speech or music.
    :param clf: The trained classifier. In this case a TensorFlow neural network
    :param scaler: The scaler used to scale the training data. Must also be used to scale new data.
    :param mfcc_in: The incoming MFCC.
    :return: An array that contains the predicted value (0 or 1) for each MFCC that was provided.
    """

    # Scale the incoming data using the same scaler used to scale the training data
    mfcc_in = scaler.transform(mfcc_in)
    # Reshape the data so the network can process it
    mfcc_in = mfcc_in.reshape((mfcc_in.shape[0], 1, mfcc_in.shape[1]))
    # Make a prediction for each element of the incoming MFCC array
    prediction = clf.predict(mfcc_in)
    # For each prediction, compare the predicted values for the two classes (speech and music)
    result = np.greater(prediction[:, 1], prediction[:, 0])
    # np.greater() returns an array of boolean values. The .astype(int) call converts those values to 0 and 1.
    return result.astype(int)

def calculate_mfccs(path_speech, path_music, max_duration):
    """
    Calculates MFCC feature vectors. There are two termination conditions for this process.
    1. The processed data reaches the max_duration.
    2. All files in the "path_speech" folder or "path_music" folder have been processed.
    :param path_speech: The (absolute or relative) path to the speech training data.
    :param path_music: The (absolute or relative) path to the music training data.
    :param max_duration: IN MINUTES. The duration of data of each class that should be processed. Example: 20000
    :return:
    """
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
        current_mfccs = read_mfcc_from_file(speech_files[i])
        for j in range(len(current_mfccs)):
            sp_mfccs.append(current_mfccs[j])

    print("Processed " + str(round(acc_duration, 2)) + " minutes of speech data.")
    print("Got " + str(len(sp_mfccs)) + " speech MFCCs.")
    len_sp_mfccs = len(sp_mfccs)

    sp_mfccs = np.array(sp_mfccs)
    sp_lbls = np.zeros(len(sp_mfccs))

    # -------------------------- Music --------------------------
    # Convert any mp3s in the music directory to WAV
    mp3s = glob.glob(path_music + "/**/*.mp3", recursive=True)  # list of mp3s in given path
    for i in range(len(mp3s)):
        # Only convert if the converted file doesn't exist yet
        if len(glob.glob(mp3s[i][:-4] + ".wav")) > 0:
            print("Found wav file, skipping...")
            continue

        print("Converting " + str(mp3s[i]) + " to wav.")
        ac.mp3_to_16_khz_wav(mp3s[i])

    # Process music wavs
    wavs = glob.glob(path_music + "/**/*.wav", recursive=True)
    # Remove 11 kHz files from list
    wavs = [file for file in wavs if "11_kHz.wav" not in file]
    processed_files = []
    wav_mfccs = []
    acc_duration = 0  # Reset processed length
    retries = 0  # Retries for the skipping of files that are too long
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
        current_mfccs = read_mfcc_from_file(wavs[i])

        # Only append the MFCCs if the number of current MFCCs + the number of MFCCs in the currently processed file
        # do not exceed the total number of MFCCs in the speech set.
        # This guarantees that the numbers of MFCCs in the speech and music sets are always close to the same
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
def read_mfcc(sig, rate):
    mfcc_feat = mfcc(sig, rate, appendEnergy=False)
    d_mfcc_feat = delta(mfcc_feat, 2)

    return np.append(mfcc_feat, d_mfcc_feat, axis=1)

def read_mfcc_from_file(filepath):
    (rate, sig) = wav.read(filepath)
    # Convert signal to mono
    sig = util.stereo2mono(sig)
    return read_mfcc(sig, rate)
