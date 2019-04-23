import joblib
import numpy as np
import sklearn.metrics

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

def main():
    data = []
    y_true = []

    print("Loading models...")
    # MFCC Tensorflow nn
    clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')
    scaler_mfcc = joblib.load("scaler_mfcc.joblib")

    print("Preparing random subset of training data...")
    # Prepare random subset of data for abl and cfa
    for i in tqdm(range(100)):
        s_file = util.get_random_file("wav", ext_hdd_path + "data/speech")
        # If file already in data pick new one
        while s_file in data or s_file[:-4] + "_11_kHz.wav" in data:
            s_file = util.get_random_file("wav", ext_hdd_path + "data/speech")
        data.append(s_file)
        y_true.append(0)

        m_file = util.get_random_file("wav", ext_hdd_path + "data/music")
        while m_file in data or m_file[:-4] + "_11_kHz.wav" in data:
            m_file = util.get_random_file("wav", ext_hdd_path + "data/music")
        data.append(m_file)
        y_true.append(1)

    print("Evaluating MFCC Feature...")
    evaluate_mfcc(data, y_true, clf_mfcc, scaler_mfcc)

    # print("Evaluating CFA Feature...")
    # evaluate_cfa(data, y_true, thresholds=[0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2])
    #evaluate_cfa(data, y_true, thresholds=[1.2, 1.5, 1.8, 2.2, 2.4, 2.6, 3, 3.2, 3.4, 3.6, 3.8])

def evaluate_mfcc(x_tst, y_true, clf, scaler):
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
    print("----------------------------------------------------------------- \n\n")


def evaluate_cfa(x_tst, y_true, thresholds):
    y_cfas = []
    for file in tqdm(x_tst):
        #print("Current file: " + file)

        # Calculate the spectrogram
        spectrogram = Processing.cfa_preprocessing(file)

        # CFA classification
        cfa = CFA.calculate_cfa(file, spectrogram)
        y_cfas.append(cfa)

    for threshold in thresholds:
        print("Evaluation for CFA with threshold " + str(threshold) + ":")

        y_pred = []
        for cfa_value in y_cfas:
            if cfa_value < threshold:
                y_pred.append(0)  # Speech
            else:
                y_pred.append(1)  # Music

        report = sklearn.metrics.classification_report(y_true, y_pred, labels, target_names)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels)

        print("Report")
        print(report)

        print("Confusion Matrix")
        print(confusion_matrix)
        print("----------------------------------------------------------------- \n\n")


main()
