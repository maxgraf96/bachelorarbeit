import sklearn.metrics

import Processing
import util
from features import GRAD, CFA
import sklearn.metrics
from tqdm import tqdm

import Processing
import util
from features import GRAD

labels = [0, 1]
target_names = ["Speech", "Music"]
ext_hdd_path = "/media/max/Elements/bachelorarbeit/"

def main():
    data = []
    y_true = []

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

    print("Evaluating GRAD Feature...")
    # evaluate_grad(data, y_true, [-1700, -1600, -1500, -1400, -1300, -1200])
    evaluate_grad(data, y_true, [-1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, -50, -20, -10])

    # print("Evaluating CFA Feature...")
    # evaluate_cfa(data, y_true, thresholds=[0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2])

def evaluate_grad(x_tst, y_true, thresholds):
    y_grads = []
    for file in tqdm(x_tst):
        #print("Current file: " + file)

        # Calculate the spectrogram
        spectrogram = Processing.cfa_abl_preprocessing(file)

        # GRAD classification
        grad = GRAD.calculate_grad(file, spectrogram)
        y_grads.append(grad)

    for threshold in thresholds:
        print("Evaluation for GRAD with threshold " + str(threshold) + ":")

        y_pred = []
        for grad_value in y_grads:
            if grad_value > threshold:
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

def evaluate_cfa(x_tst, y_true, thresholds):
    y_cfas = []
    for file in tqdm(x_tst):
        #print("Current file: " + file)

        # Calculate the spectrogram
        spectrogram = Processing.cfa_abl_preprocessing(file)

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
