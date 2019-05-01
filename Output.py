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

import numpy as np

from features import MFCC


def print_mfcc(mfcc, clf, scaler_mfcc, result_mfcc, splits=9):
    """
    Split the incoming MFCC data into 9 blocks, make a prediction and print the result
    :param mfcc: The incoming MFCC feature vectors
    :param clf: The trained MFCC classifier
    :param scaler_mfcc: The scaler instance (from training)
    :param result_mfcc: List to which the result is written
    :param splits: How many splits should be performed on the feature vectors
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