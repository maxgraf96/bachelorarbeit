import time
import numpy as np
from features import MFCC, CFA


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


def print_cfa(cfa, result_cfa):
    result = round(cfa, 4)
    result_cfa[0] = result
    print("CFA Music: " + str(result))