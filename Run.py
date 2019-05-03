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
import sys

import joblib
import tensorflow as tf

import Main
import radiorec
import util
from features import MFCC

cl_arguments = sys.argv[1:]

# Command line arguments check
station = cl_arguments[0]
# Stream 0.5s chunks or 10 second chunks
if cl_arguments[1] == "live":
    live_stream = True
elif cl_arguments[1] == "10s":
    live_stream = False
else:
    raise ValueError("The mode parameter must be either 'live' or '10s'.")

try:
    listening_preference = cl_arguments[2]
except:
    listening_preference = None

try:
    replacement_path = cl_arguments[3]
except:
    replacement_path = None

# Check if replacement file is specified when using live option
if live_stream and listening_preference is not None and replacement_path is None:
    raise ValueError("Please specify a replacement file path when using the live option.")

# Use both classifiers. Altering these values would allow for classification using only one of the classification methods.
is_mfcc = True
is_cfa = True

def run():
    """
    Runs the classification program using input values from the command line.
    """

    # Clear previous streams
    Main.clear_streams()

    # Persist classifier if it does not exist
    if len(glob.glob("saved_classifiers/clf_mfcc.h5")) < 1:
        print("Saving model...")
        clf_mfcc, scaler_mfcc = MFCC.train_mfcc_nn(util.data_path + "speech", util.data_path + "music", 20000)
        clf_mfcc.save('saved_classifiers/clf_mfcc.h5')
        # Persist scaler
        joblib.dump(scaler_mfcc, "saved_classifiers/scaler_mfcc.joblib")

    else:
        print("Restoring models...")
        clf_mfcc = tf.keras.models.load_model('saved_classifiers/clf_mfcc.h5')
        scaler_mfcc = joblib.load("saved_classifiers/scaler_mfcc.joblib")

    if live_stream:
        Main.calc_from_stream(station, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa, listening_preference, replacement_path)
    else:
        filename = "stream_long"
        radiorec.record(station, 10, filename)
        file = "data/test/" + filename + ".mp3"
        Main.calc_from_file(file, filename, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa)


run()
