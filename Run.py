import glob
import sys

import joblib
import tensorflow as tf
from pygame import mixer

import Main
import radiorec
import util
from features import MFCC

station = "fm4"

# Livestream or from file
live_stream = False

cl_arguments = sys.argv[1:]

# Command line arguments check
is_mfcc = "mfcc" in cl_arguments
is_cfa = "cfa" in cl_arguments

# CFA threshold
cfa_threshold = 3.2

def run():
    # Clear previous streams
    Main.clear_streams()

    # Persist classifier if it does not exist
    if len(glob.glob("clf_mfcc.h5")) < 1:
        print("Saving model...")
        clf_mfcc, scaler_mfcc = MFCC.train_mfcc_nn(util.data_path + "data/speech", util.data_path + "data/music", 20000, test=False)
        clf_mfcc.save('clf_mfcc.h5')
        joblib.dump(scaler_mfcc, "scaler_mfcc.joblib")

    else:
        print("Restoring models...")
        clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')
        scaler_mfcc = joblib.load("scaler_mfcc.joblib")

    if live_stream:
        Main.calc_from_stream(station, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa, "music")
    else:
        filename = "stream_long"
        radiorec.my_record(station, 10, filename)
        file = "data/test/" + filename + ".mp3"
        #
        # filename = "klangcollage"
        # file = "data/replacements/" + filename + ".mp3"

        Main.calc_from_file(file, filename, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa)

run()