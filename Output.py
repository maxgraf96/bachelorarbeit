from time import sleep
import numpy as np
from features import MFCC, CFA


def print_mfcc(mfcc, clf, duration, splits=9):
    """
    Splits the n seconds (coming from duration) into 'splits' bundles each containing the information for n/splits seconds
    """

    split = np.array_split(mfcc, splits)
    sleep_duration = duration / splits

    total = 0
    for i in range(splits):
        result = MFCC.predict_nn(clf, split[i])
        ones = np.count_nonzero(result)
        total += ones / len(result)
        zeros = len(result) - ones
        #print("Ones: " + str(ones))
        #print("Zeros: " + str(zeros))
        # print("Music: " + str(round(ones / len(result), 4)))
        #print()
        # sleep(sleep_duration)
    print("MFCC Music: ", str(round(total / splits, 4)))


def print_cfa(cfa, threshold=1.24):
    if cfa < threshold:
        res = "Speech"
    else:
        res = "Music"
    print("CFA: " + str(round(cfa, 2)) + " - " + res)


def print_grad(grad, threshold=-1000):
    result = "Speech" if grad > threshold else "Music"
    print("Gradient: " + str(round(grad, 2)) + " - " + result)