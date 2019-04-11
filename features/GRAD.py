import glob

import math
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import tensorflow as tf

import Processing

"""
An idea for a feature
"""

def calculate_grad(file, spec=None):

    # Get the spectrogram
    if spec is None:
        spec = Processing.cfa_grad_preprocessing(file)

    # "EQ" the primary voice frequencies out
    #spec = np.delete(spec, np.s_[27:280], axis=0)

    # Create blocks consisting of 10 frames each
    no_blocks = math.ceil(spec.shape[1] / (spec.shape[1] / 5))
    gradients = []
    for step in range(no_blocks):
        start = step * 10
        end = start + 10
        if end > spec.shape[1]:
            end = spec.shape[1]
        block = spec[:, start:end]

        # gradient = np.gradient(np.sum(spec, axis=1))
        if block.shape[1] == 1 or block.size == 0:
            continue

        gradient = np.real(np.gradient(block))

        # try:
        # except:
        #     print(block)
        #     continue

        if len(gradient) > 1:
            gradient = gradient[1]
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
    sp_grads = []
    no_grads = 0
    file = 0
    while no_grads < max_grads:
        speech_files[file] = Processing.cfa_grad_filerate_preprocessing(speech_files[file])
        print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
        gradients = calculate_grad(speech_files[file])
        sp_grads.extend(gradients)
        no_grads += len(gradients)
        file += 1


    print("Processed " + str(file) + " speech files.")

    sp_grads = np.array(sp_grads)
    sp_lbls = np.zeros(len(sp_grads))
    print("Average Gradient speech: " + str(np.average(sp_grads)))

    # -------------------------- Music --------------------------
    if path_music is None:
        return

    music_files = glob.glob(path_music + "/**/*.wav", recursive=True)  # list of files in given path
    mu_grads = []

    no_grads = 0
    file = 0
    while no_grads < max_grads:
        music_files[file] = Processing.cfa_grad_filerate_preprocessing(music_files[file])
        print(str(file) + " of " + str(len(music_files)) + " - processing " + str(music_files[file]))
        gradients = calculate_grad(music_files[file])
        mu_grads.extend(gradients)
        no_grads += len(gradients)
        file += 1

    print("Processed " + str(file) + " music files.")

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
    #trn = sklearn.preprocessing.scale(trn, axis=1)
    #pca = PCA(n_components=9)
    #principal_components = pca.fit_transform(trn)

    # Classifier fitting
    # Tensorflow nn
    clf = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 513)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    clf.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    trn = trn.reshape((trn.shape[0], 1, trn.shape[1]))
    clf.fit(trn, lbls, epochs=5)

    return clf#, pca


def predict_nn(clf, grad_in):
    grad_in = grad_in.reshape((1, 1, grad_in.shape[0]))
    prediction = clf.predict(grad_in)
    return prediction
