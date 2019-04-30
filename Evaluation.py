import glob

import joblib
import numpy as np
import scipy
import sklearn.metrics
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

labels = [0, 1]
target_names = ["Speech", "Music"]
ext_hdd_path = "/media/max/Elements/bachelorarbeit/"
speech_path = "data/test/speech"
music_path = "data/test/music"
plots_path = "plots/"

def run():
    print("Loading models...")
    # MFCC Tensorflow nn
    clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')
    scaler_mfcc = joblib.load("scaler_mfcc.joblib")

    runs = 1
    samples = 200

    # evaluate_features(runs, samples, clf_mfcc, scaler_mfcc)

    # evaluate_muspeak_mirex_folder(clf_mfcc, scaler_mfcc)

    # train_mfcc_kfold(util.data_path + "data/speech/gtzan", util.data_path + "data/music/gtzan", 20000, k=10)

def train_mfcc_kfold(path_speech, path_music, max_duration, k):
    """
       Trains a TensorFlow neural network
       :param path_speech: The path to the speech data
       :param path_music: The path to the music data
       :param max_duration: The total duration of files that should be selected of each class. For example, 5000 would
       :param k: The number of splits on the training data
       train the network with 5000 minutes of speech files and 5000 minutes of music files
       :param test: If the data should be split into a training and a test set
       :return: The trained classifier and the scaler used to scale the training data
       """

    # Use existing training data (= extracted MFCCs). This is to skip the process of recalculating the MFCC each time.
    if len(glob.glob(util.data_path + "data/mfcc_trn_kfold.joblib")) < 1:
        trn, lbls = MFCC.calculate_mfccs(path_speech, path_music, max_duration)
        joblib.dump(trn, util.data_path + "data/mfcc_trn_kfold.joblib")
        joblib.dump(lbls, util.data_path + "data/mfcc_lbls_kfold.joblib")
    else:
        trn = joblib.load(util.data_path + "data/mfcc_trn_kfold.joblib")
        lbls = joblib.load(util.data_path + "data/mfcc_lbls_kfold.joblib")

    kfold = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    accuracies = []

    for train_index, test_index in kfold.split(trn):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = trn[train_index], trn[test_index]
        y_train, y_test = lbls[train_index], lbls[test_index]

        # Preprocessing
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Classifier fitting
        # Tensorflow nn
        clf = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(1, 26)),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        clf.fit(X_train, y_train, epochs=5)

        # Run on test set
        correct = 0
        incorrect = 0

        y_pred = []
        for i in tqdm(range(len(X_test))):
            result = MFCC.predict_nn(clf, scaler, X_test[i].reshape(1, -1))
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
        print("y_pred", y_pred)
        print("y_pred length", len(y_pred))

        print("y_true", y_test)
        print("y_true length", len(y_test))

        report = sklearn.metrics.classification_report(y_test, y_pred, labels, target_names)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, labels)

        print("Report")
        print(report)

        print("Confusion Matrix")
        print(confusion_matrix)
        pretty_print(confusion_matrix, "MFCC", train_index)
        print("----------------------------------------------------------------- \n\n")

    print("Mean accuracy:", np.mean(np.array(accuracies)))

# TODO docu, this is for thresholds
def evaluate_features(runs, samples, clf_mfcc, scaler_mfcc):
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

        # evaluate_mfcc(data, y_true, clf_mfcc, scaler_mfcc, it)

        evaluate_cfa(data, y_true, thresholds=[3.2, 3.3, 3.4, 3.5, 3.6, 3.7], iteration=it)

def evaluate_mfcc(x_tst, y_true, clf, scaler, iteration):
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
    pretty_print(confusion_matrix, "MFCC", iteration)
    print("----------------------------------------------------------------- \n\n")

def evaluate_cfa(x_tst, y_true, thresholds, iteration):
    print("Evaluating CFA Feature...")
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
            # print(np.mean(peakis))
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

def evaluate_muspeak_mirex_folder(clf_mfcc, scaler_mfcc):

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
        eq, l = evaluate_muspeak_mirex(ext_hdd_path + "data/test/muspeak-mirex2015-detection-examples/" + name + ".csv", name, clf_mfcc, scaler_mfcc)
        equal += eq
        length += l

    # weigh accuracies by length of files
    print("Overall accuracy:", str(equal/length))

def evaluate_muspeak_mirex(csv_path, filename, clf, scaler):
    print("Evaluating", filename)
    if len(glob.glob("plots/speech_music_maps/" + filename + ".csv")) < 1:
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


run()
