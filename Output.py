import time
import numpy as np
from features import MFCC, CFA, GRAD


def print_mfcc(mfcc, clf, scaler_mfcc, duration, result_mfcc, splits=9):
    """
    Splits the n seconds (coming from duration) into 'splits' bundles each containing the information for n/splits seconds
    """

    split = np.array_split(mfcc, splits)

    total = 0
    for i in range(splits):
        result = MFCC.predict_nn(clf, scaler_mfcc, split[i])
        ones = np.count_nonzero(result)
        total += ones / len(result)
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
    total = 0
    start = time.time()
    for grad in grads:
        result = GRAD.predict_nn(clf, grad)
        ones = np.count_nonzero(result)
        total += ones / len(result)
    result = round(total / len(grads), 4)

    end = time.time()
    print("Time for GRAD NN prediction: ", str(end - start))

    result_grad[0] = result
    print("GRAD Music: ", result)