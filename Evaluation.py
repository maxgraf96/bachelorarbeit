import glob

import joblib
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

import Processing
import util
from features import CFA, MFCC
import sklearn.metrics
from tqdm import tqdm
import tensorflow as tf

import Processing
import util

labels = [0, 1]
target_names = ["Speech", "Music"]
ext_hdd_path = "/media/max/Elements/bachelorarbeit/"
speech_path = "data/test/speech"
music_path = "data/test/music"
plots_path = "plots/"

runs = 1
samples = 200

def main():
    print("Loading models...")
    # MFCC Tensorflow nn
    clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')
    scaler_mfcc = joblib.load("scaler_mfcc.joblib")

    for it in range(runs):
        data = []
        y_true = []

        print("Preparing random subset of training data...")
        speech_wavs = np.array(glob.glob(ext_hdd_path + speech_path + "/**/*.wav", recursive=True))
        music_wavs = np.array(glob.glob(ext_hdd_path + music_path + "/**/*.wav", recursive=True))

        # Shuffle and select n samples
        np.random.shuffle(speech_wavs)
        np.random.shuffle(music_wavs)
        random_speech = speech_wavs[:samples]
        random_music = music_wavs[:samples]

        # Prepare random subset of data for abl and cfa
        for i in range(samples):
            data.append(random_speech[i])
            y_true.append(0)

            data.append(random_music[i])
            y_true.append(1)

        # print("Evaluating MFCC Feature...")
        # evaluate_mfcc(data, y_true, clf_mfcc, scaler_mfcc, it)

        print("Evaluating CFA Feature...")
        # evaluate_cfa(data, y_true, thresholds=[1.2, 1.5, 1.8, 2.2, 2.4, 2.6, 3, 3.2, 3.4, 3.6, 3.8])
        evaluate_cfa(data, y_true, thresholds=[3, 3.1, 3.2, 3.3, 3.4, 3.5], iteration=it)

def pretty_print(confusion_matrix, classifier, iteration=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    plt.title('Confusion matrix for ' + classifier)
    fig.colorbar(cax)

    # Show values in boxes
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    lbls_str = ["Speech", "Music"]
    ax.set_xticklabels([''] + lbls_str)
    ax.set_yticklabels([''] + lbls_str)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()
    plt.savefig(plots_path + classifier + '/cm_' + classifier + '_' + str(iteration) + '.png')

def evaluate_mfcc(x_tst, y_true, clf, scaler, iteration):
    y_mfccs = []
    for file in tqdm(x_tst):
        # GRAD classification
        mfcc = MFCC.read_mfcc_from_file(file)
        result = MFCC.predict_nn(clf, scaler, mfcc)
        ones = np.count_nonzero(result)
        total = ones / len(result)
        if total < 0.5:
            y_mfccs.append(0)
        else:
            y_mfccs.append(1)

    print("Evaluation for MFCC feature:")

    y_pred = y_mfccs

    report = sklearn.metrics.classification_report(y_true, y_pred, labels, target_names)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels)

    print("Report")
    print(report)

    print("Confusion Matrix")
    print(confusion_matrix)
    pretty_print(confusion_matrix, "MFCC", iteration)
    print("----------------------------------------------------------------- \n\n")


def evaluate_cfa(x_tst, y_true, thresholds, iteration):
    y_cfas = []
    y_peakis = []
    for file in tqdm(x_tst):
        #print("Current file: " + file)

        # Calculate the spectrogram
        spectrogram = Processing.cfa_preprocessing(file)

        # CFA classification
        cfa, peakis = CFA.calculate_cfa(file, spectrogram)
        y_cfas.append(cfa)
        y_peakis.append(peakis)

    for threshold in thresholds:
        print("Evaluation for CFA with threshold " + str(threshold) + ":")

        y_pred = []
        for peakis in y_peakis:
            if np.mean(peakis) < threshold:
                y_pred.append(0)  # Speech
            else:
                y_pred.append(1)  # Music

        report = sklearn.metrics.classification_report(y_true, y_pred, labels, target_names)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels)

        print("Report")
        print(report)

        print("Confusion Matrix")
        print(confusion_matrix)
        pretty_print(confusion_matrix, "CFA", iteration)
        print("----------------------------------------------------------------- \n\n")


main()
