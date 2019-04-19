import glob

import math
import numpy as np
import sklearn
from numba import jit
from sklearn.decomposition import PCA
import tensorflow as tf

import Processing

"""
An idea for a feature
"""

def calculate_grad(file=None, spec=None):

    # Get the spectrogram
    if file is None and spec is None:
        raise ValueError("Either a file for conversion or a spectrogram must be passed to this function.")
    elif file is not None and spec is None:
        spec = Processing.cfa_grad_preprocessing(file)

    # Remove frequencies not primarily in the voice spectrum
    spec = np.delete(spec, np.s_[27:], axis=0)
    spec = np.delete(spec, np.s_[0:6], axis=0)

    return calculation(spec)

@jit(cache=True)
def calculation(spec):

    # Create blocks consisting of 3 frames each
    no_blocks = math.ceil(spec.shape[1] / 3)
    gradients = []
    for step in range(no_blocks):
        start = step * 3
        end = start + 3
        if end > spec.shape[1]:
            end = spec.shape[1]
        block = spec[:, start:end]

        # gradient = np.gradient(np.sum(spec, axis=1))
        if block.shape[1] == 1 or block.size == 0:
            continue

        gradient = np.real(np.gradient(block))

        if len(gradient) > 1:
            gradient = gradient[0]
        # gradient = np.sum(gradient, axis=1)
        # gradient = np.sum(gradient) / block.shape[0]
        # gradient = gradient.real

        # Take mean gradient per frequency band
        gradient = np.mean(gradient, axis=1)
        gradients.append(gradient)

    return gradients


def calculate_grads(path_speech, path_music, max_grads):

    # -------------------------- Speech --------------------------
    speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
    # Remove 16 kHz files (from MFCC processing) from list
    speech_files = [file for file in speech_files if "16_kHz.wav" not in file]
    sp_grads = []
    no_grads = 0
    file = 0
    processed_files = []
    while no_grads < max_grads:
        if file in processed_files:
            file += 1
            continue
        speech_files[file] = Processing.cfa_grad_filerate_preprocessing(speech_files[file])
        print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
        gradients = calculate_grad(speech_files[file])
        sp_grads.extend(gradients)
        no_grads += len(gradients)
        file += 1
        processed_files.append(file)


    print("Processed " + str(file) + " speech files and " + str(no_grads) + " Grads.")

    sp_grads = np.array(sp_grads)
    sp_lbls = np.zeros(len(sp_grads))
    print("Average Gradient speech: " + str(np.average(sp_grads)))

    # -------------------------- Music --------------------------
    if path_music is None:
        return

    music_files = glob.glob(path_music + "/**/*.wav", recursive=True)  # list of files in given path
    # Remove 16 kHz files (from MFCC processing) from list
    music_files = [file for file in music_files if "16_kHz.wav" not in file]
    mu_grads = []

    no_grads = 0
    file = 0
    processed_files = []
    while no_grads < max_grads:
        if file in processed_files:
            file += 1
            continue
        music_files[file] = Processing.cfa_grad_filerate_preprocessing(music_files[file])
        print(str(file) + " of " + str(len(music_files)) + " - processing " + str(music_files[file]))
        gradients = calculate_grad(music_files[file])
        mu_grads.extend(gradients)
        no_grads += len(gradients)
        file += 1
        processed_files.append(file)

    print("Processed " + str(file) + " music files and " + str(no_grads) + " Grads.")

    # Convert GRAD list into array for fitting
    mu_grads = np.array(mu_grads)
    # Create the labels for the music files
    mu_lbls = np.ones(len(mu_grads))

    # Concat arrays
    trn = np.concatenate((sp_grads, mu_grads))
    lbls = np.concatenate((sp_lbls, mu_lbls))

    print("Average Gradient speech: " + str(np.average(sp_grads)) + ", Average Gradient music: " + str(np.average(mu_grads)))

    return trn, lbls


def train_grad_nn(path_speech, path_music, max_grads):

    # Calculate GRADs
    trn, lbls = calculate_grads(path_speech, path_music, max_grads)

    # Preprocessing
    scaler = sklearn.preprocessing.StandardScaler()
    trn = scaler.fit_transform(trn)

    #pca = PCA(n_components=15)
    #prcomp = pca.fit_transform(trn)

    # Classifier fitting
    # Tensorflow nn
    clf = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 21)),
        tf.keras.layers.Dense(21, activation=tf.nn.relu),
        tf.keras.layers.Dense(11, activation=tf.nn.relu),
        tf.keras.layers.Dense(5, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    clf.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    trn = trn.reshape((trn.shape[0], 1, trn.shape[1]))
    clf.fit(trn, lbls, epochs=5)

    return clf, scaler


def predict_nn(clf, scaler_grad, grad_in):
    grad_in = scaler_grad.transform(grad_in.reshape(1, -1))
    grad_in = grad_in.reshape((1, 1, grad_in.shape[1]))
    prediction = clf.predict(grad_in)
    result = np.greater(prediction[:, 1], prediction[:, 0])
    return result.astype(int)
