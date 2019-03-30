import glob

import numpy as np

import Processing

"""
An idea for a feature
"""

def calculate_abl(file, spec=None):

    # Get the spectrogram
    if spec is None:
        spec = Processing.cfa_abl_preprocessing(file)

    # HPSS
    #harmonic, percussive = librosa.decompose.hpss(spec)
    #h_sum = np.sum(harmonic, axis=1)
    #p_sum = np.sum(percussive, axis=1)

    # "EQ" the primary voice frequencies out
    spec = np.delete(spec, np.s_[27:280], axis=0)

    # Binarize
    #spec = np.where(spec > 10, 1, 0)

    gradient = np.gradient(np.sum(spec, axis=1))
    gradient = np.sum(gradient) / spec.shape[1]
    gradient = gradient.real
    result = "Speech" if gradient > -1500 else "Music"
    print("Gradient: " + str(round(gradient, 2)) + " - " + result)

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
