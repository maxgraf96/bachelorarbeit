import sklearn.metrics

import Processing
import util
from features import ABL, CFA
import sklearn.metrics

import Processing
import util
from features import ABL

labels = [0, 1]
target_names = ["Speech", "Music"]

def main():
    # Prepare random subset of data for abl and cfa
    data = []
    y_true = []
    for i in range(20):
        data.append(util.get_random_file("wav", "data/speech"))
        y_true.append(0)

        data.append(util.get_random_file("wav", "data/music"))
        y_true.append(1)

    # evaluate_abl(data, y_true, -1500)
    evaluate_cfa(data, y_true, 1.1)

def evaluate_abl(x_tst, y_true, threshold):
    y_pred = []

    for file in x_tst:
        # Calculate the spectrogram
        spectrogram = Processing.cfa_abl_preprocessing(file)

        # ABL classification
        abl = ABL.calculate_abl(file, spectrogram)
        if abl > threshold:
            y_pred.append(0)  # Speech
        else:
            y_pred.append(1)  # Music

    report = sklearn.metrics.classification_report(y_true, y_pred, labels, target_names)
    print(report)

def evaluate_cfa(x_tst, y_true, threshold=1.24):
    y_pred = []

    for file in x_tst:
        # Calculate the spectrogram
        spectrogram = Processing.cfa_abl_preprocessing(file)

        # CFA classification
        cfa = CFA.calculate_cfa(file, spectrogram)
        if cfa < threshold:
            y_pred.append(0)  # Speech
        else:
            y_pred.append(1)  # Music

        print("Current file: " + file)

    report = sklearn.metrics.classification_report(y_true, y_pred, labels, target_names)
    print(report)


main()
