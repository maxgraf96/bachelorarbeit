import glob

import math
import numpy as np

import Processing

"""
An idea for a feature
"""

def calculate_grad(file, spec=None, threshold=-1500):

    # Get the spectrogram
    if spec is None:
        spec = Processing.cfa_abl_preprocessing(file)

    # "EQ" the primary voice frequencies out
    #spec = np.delete(spec, np.s_[27:280], axis=0)

    # Create blocks consisting of 10 frames each with 5 blocks overlap
    no_blocks = math.ceil(spec.shape[1] / 5)
    gradients = []
    for step in range(no_blocks):
        start = step * 5
        end = start + 10
        if end > spec.shape[1]:
            end = spec.shape[1]
        block = spec[:, start:end]

        # gradient = np.gradient(np.sum(spec, axis=1))
        gradient = np.gradient(block)
        gradient = np.sum(gradient, axis=1)
        gradient = np.sum(gradient) / block.shape[0]
        gradient = gradient.real

        gradients.append(gradient)

    # TODO fix this
    result = np.mean(gradients)
    return result


def calculate_abls(path_speech, path_music, max_items):

    # -------------------------- Speech --------------------------
    speech_files = glob.glob(path_speech + "/**/*.wav", recursive=True)  # list of files in given path
    sp_abls = []
    for file in range(max_items):
        print(str(file) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[file]))
        result = calculate_grad(speech_files[file])
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
        result = calculate_grad(music_files[file])
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
