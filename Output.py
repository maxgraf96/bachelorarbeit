from time import sleep
import numpy as np
from features import MFCC, CFA, GRAD


def print_mfcc(mfcc, clf, duration, result_mfcc, splits=9):
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
    result = round(total / splits, 4)
    result_mfcc[0] = result
    print("MFCC Music: ", str(result))


def print_cfa(cfa, result_cfa, threshold=1.24):
    if cfa < threshold:
        res = "Speech"
    else:
        res = "Music"

    result = round(cfa, 4)
    result_cfa[0] = result
    print("CFA: " + str(result) + " - " + res)


def print_grad(grads, clf, result_grad):
    ones = 0
    zeroes = 0
    for grad in grads:
        result = GRAD.predict_nn(clf, grad)
        if result[0][0] > result[0][1]:
            zeroes += 1
        else:
            ones += 1

    else:
        result = ones / (ones + zeroes)

    result_grad[0] = result
    print("GRAD Music: ", result)