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
import scipy
import sklearn.model_selection
import matplotlib.pyplot as plt

import Main
import Processing
import util
from features import CFA, MFCC
import sklearn.metrics
from tqdm import tqdm
import tensorflow as tf

import Processing
import util

# The labels for speech and music
labels = [0, 1]
# The target names for speech and music
target_names = ["Speech", "Music"]

speech_path = "test/speech"
music_path = "test/music"
plots_path = "plots/"

def run():
    print("Loading models...")
    # MFCC Keras nn
    clf_mfcc = tf.keras.models.load_model('saved_classifiers/clf_mfcc.h5')
    scaler_mfcc = joblib.load("saved_classifiers/scaler_mfcc.joblib")

    is_mfcc = True
    is_cfa = True

    # How many runs should be used to train the features on the test data
    runs = 10
    # How many samples should be used for each class
    samples = 200
    # The different threshold values against which to compare the CFA values
    thresholds = [3, 3.1, 3.2, 3.25, 3.3, 3.4, 3.5, 3.6, 3.7]
    evaluate_features_on_test_data(runs, samples, is_mfcc, is_cfa, clf_mfcc, scaler_mfcc, thresholds)

    evaluate_muspeak_mirex_folder(clf_mfcc, scaler_mfcc)

    evaluate_mfcc_kfold(util.data_path + "speech/gtzan", util.data_path + "music/gtzan", 500, k=2)

def evaluate_mfcc_kfold(path_speech, path_music, max_duration, k):
    """
       Trains and evaluates a Keras neural network
       :param path_speech: The path to the speech data
       :param path_music: The path to the music data
       :param max_duration: The total duration of files that should be selected of each class. For example, 5000 would
       :param k: The number of splits on the training data
       train the network with 5000 minutes of speech files and 5000 minutes of music files
       :param test: If the data should be split into a training and a test set
       :return: The trained classifier and the scaler used to scale the training data
       """

    # Use existing training data (= extracted MFCCs). This is to skip the process of recalculating the MFCC each time.
    if len(glob.glob(util.data_path + "mfcc_trn_kfold.joblib")) < 1:
        trn, lbls = MFCC.calculate_mfccs(path_speech, path_music, max_duration)
        joblib.dump(trn, util.data_path + "mfcc_trn_kfold.joblib")
        joblib.dump(lbls, util.data_path + "mfcc_lbls_kfold.joblib")
    else:
        trn = joblib.load(util.data_path + "mfcc_trn_kfold.joblib")
        lbls = joblib.load(util.data_path + "mfcc_lbls_kfold.joblib")

    # Initialize
    kfold = sklearn.model_selection.KFold(n_splits=k, shuffle=True)

    # The accuracy of each iteration is stored in this list
    accuracies = []

    for train_index, test_index in kfold.split(trn):
        print("TRAIN:", train_index, "TEST:", test_index)

        # Take split from the total training data
        x_train, x_test = trn[train_index], trn[test_index]
        y_train, y_test = lbls[train_index], lbls[test_index]

        # Scale data
        scaler = sklearn.preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)

        # Classifier fitting
        # Keras nn
        clf = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(1, 26)),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Data reshaping required for the network
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        # Fit data
        clf.fit(x_train, y_train, epochs=5)

        # Run on test set
        correct = 0
        incorrect = 0

        y_pred = []
        for i in tqdm(range(len(x_test))):
            result = MFCC.predict_nn(clf, scaler, x_test[i].reshape(1, -1))
            ones = np.count_nonzero(result)
            result = ones / len(result)
            if result >= 0.5:
                y_pred.append(1)
                if y_test[i] == 1:
                    correct += 1
                else:
                    incorrect += 1
            else:
                y_pred.append(0)
                if y_test[i] == 0:
                    correct += 1
                else:
                    incorrect += 1

        accuracy = correct / (correct + incorrect)
        accuracies.append(accuracy)
        print("Results for test set: ", str(round(accuracy, 2)) + "%")

        y_pred = np.array(y_pred)

        report = sklearn.metrics.classification_report(y_test, y_pred, labels, target_names)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, labels)

        print("Report")
        print(report)

        print("Confusion Matrix")
        print(confusion_matrix)
        pretty_print_cm(confusion_matrix, "MFCC", str(train_index))
        print("----------------------------------------------------------------- \n\n")

    print("Mean accuracy:", np.mean(np.array(accuracies)))

def evaluate_features_on_test_data(runs, samples, is_mfcc, is_cfa, clf_mfcc, scaler_mfcc, thresholds):
    """
    Evaluates trained classifier on a given number of samples of unknown test data. On every iteration, all of the
    test data is shuffled to produce a new subset containing n samples
    :param runs: How many iterations should be performed
    :param samples: How many samples should be classified per iteration
    :param is_mfcc: If MFCC classifier should be evaluated
    :param is_cfa: If the CFA classifier should be evaluated
    :param clf_mfcc: The trained MFCC classifier
    :param scaler_mfcc: The MFCC scaler (from training)
    :param thresholds: A list of thresholds to compare for the CFA feature
    """
    cfa_cms = []  # CFA confusion matrices for every threshold on each run
    for it in range(runs):
        data = []
        y_true = []

        print("Preparing random subset of training data...")
        speech_wavs = np.array(glob.glob(util.data_path + speech_path + "/**/*.wav", recursive=True))
        music_wavs = np.array(glob.glob(util.data_path + music_path + "/**/*.wav", recursive=True))

        # Shuffle and select n samples
        np.random.shuffle(speech_wavs)
        np.random.shuffle(music_wavs)
        random_speech = speech_wavs[:samples]
        random_music = music_wavs[:samples]

        # Prepare random subset of data for CFA
        for i in range(samples):
            data.append(random_speech[i])
            y_true.append(0)

            data.append(random_music[i])
            y_true.append(1)

        if is_mfcc:
            evaluate_mfcc(data, y_true, clf_mfcc, scaler_mfcc, iteration=it)

        if is_cfa:
            cfa_cms.append(evaluate_best_cfa_threshold(data, y_true, thresholds))

    if is_cfa:
        cfa_cms = np.array(cfa_cms)
        print(cfa_cms)
        cfa_cms = np.sum(cfa_cms, axis=0)
        for i in range(len(cfa_cms)):
            print("Confusion Matrix for threshold", thresholds[i], ": \n", cfa_cms[i])
            pretty_print_cm(cfa_cms[i], "CFA", "threshold_" + str(thresholds[i]))

def evaluate_mfcc(x_tst, y_true, clf, scaler, iteration):
    """
    Evaluates the trained classifier with the given data. Prints and stores confusion matrices.
    :param x_tst: The test data
    :param y_true: The true test labels
    :param clf: The classifier
    :param scaler: The scaler
    :param iteration: The current iteration
    """
    print("Evaluating MFCC feature...")
    y_mfccs = []
    for file in tqdm(x_tst):
        # MFCC classification
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
    pretty_print_cm(confusion_matrix, "MFCC", str(iteration))
    print("----------------------------------------------------------------- \n\n")

def evaluate_best_cfa_threshold(x_tst, y_true, thresholds):
    """
    Takes training data and a list of thresholds and calculates confusion matrices for each threshold
    :param x_tst: The test data
    :param y_true: The test labels
    :param thresholds: List of thresholds
    :return:
    """
    print("Evaluating CFA Feature...")
    y_cfas = []
    y_peakis = []
    confusion_matrices = []
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
        confusion_matrices.append(confusion_matrix)
        #pretty_print(confusion_matrix, "CFA", iteration=str(iteration) + "_" + str(threshold))
        print("----------------------------------------------------------------- \n\n")

    return confusion_matrices  # Confusion matrices of all thresholds for 1 run

def evaluate_muspeak_mirex_folder(clf_mfcc, scaler_mfcc):
    """
    Special evaluation method for the Muspeak-mirex dataset. For every audiofile classification is performed and speech_music_maps are stored.
    :param clf_mfcc: The MFCC classifier
    :param scaler_mfcc: The MFCC scaler
    """

    names = [
        "ConscinciasParalelasN3-OsSentidosOSentirEAsNormasParte318-10-1994",
        "ConscinciasParalelasN7-OsSentidosOSentirEAsNormasParte715-1-1994",
        "ConscinciasParalelasN11-OEspelhoEOReflexoFantasiasEPerplexidadesParte413-12-1994",
        "eatmycountry1609",
        "theconcert2_v2",
        "theconcert16",
        "UTMA-26_v2"
    ]

    equal = 0
    length = 0
    for name in names:
        eq, l = evaluate_muspeak_mirex(util.data_path + "test/muspeak-mirex2015-detection-examples/" + name + ".csv", name, clf_mfcc, scaler_mfcc)
        equal += eq
        length += l

    # weigh accuracies by length of files
    print("Overall accuracy:", str(equal/length))

def evaluate_muspeak_mirex(csv_path, filename, clf, scaler):
    """
    Classifies and stores a speech_music_map for one muspea-mirex file
    :param csv_path: If the file was previously processed the calculation can be skipped.
    :param filename: The filename of the current muspeak-mirex file
    :param clf: The MFCC classifier
    :param scaler: The MFCC scaler
    :return: The number of matching samples and the length of the file
    """

    print("Evaluating", filename)
    if len(glob.glob("plots/speech_music_maps/" + filename + ".csv")) < 1:
        print("Calculating MFCCs for ", filename)
        Main.calc_from_file(csv_path[:-4] + ".mp3", filename, clf, scaler, is_mfcc=True, is_cfa=True)

    truth = util.load_and_convert_muspeak_mirex_csv(csv_path)[:, 1]
    estimation = util.load_speech_music_map_csv("plots/speech_music_maps/" + filename + ".csv")[:, 1]

    # Remove discepancies between different file lengths (caused by the format conversion)
    if truth.shape[0] > estimation.shape[0]:
        truth = truth[:estimation.shape[0]]
    else:
        estimation = estimation[:truth.shape[0]]

    # Plot and save
    x = np.arange(len(estimation)) / 2
    util.plot_speech_music_map(filename, x, estimation, save_csv=False)

    # Plot speech music map for truth data
    if len(glob.glob("plots/speech_music_maps/" + filename + "_true.csv")) < 1:
        util.plot_speech_music_map(filename + "_true", x, truth, save_csv=False)

    equal = np.sum(truth == estimation)
    print(equal, "/", len(truth), " values are equal.")
    print("That's a similarity of ", equal / len(truth), "%")
    print()

    return equal, len(truth)

def pretty_print_cm(confusion_matrix, classifier, iteration="0"):
    """
    Plots and shows/saves a confusion matrix
    :param confusion_matrix: The confusion matrix
    :param classifier: The name of the classifier which was evaluated
    :param iteration: The current iteration
    """

    plt.clf()
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
    plt.savefig(plots_path + classifier + '/cm_' + classifier + '_' + iteration + '.png')


run()
