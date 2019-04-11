import glob
import sys
import time
from time import sleep

import numpy as np
import Output
from pathlib import Path
from pygame import mixer

import Processing
import radiorec
import util
from features import MFCC, CFA, GRAD
import AudioConverter as ac
import tensorflow as tf
import threading

cl_arguments = sys.argv[1:]
ext_hdd_path = "/media/max/Elements/bachelorarbeit/"

def main(station):

    # clear previous streams
    for p in Path("data/test").glob("stream*.mp3"):
        p.unlink()

    # init
    mixer.init()

    # persist classifiers
    # MFCC
    if len(glob.glob("clf.h5")) < 1:
        print("Saving model...")
        # Tensorflow nn, Note: Only saves the network currently (pca is discarded)
        clf_mfcc, pca = MFCC.train_mfcc_nn(ext_hdd_path + "data/speech", ext_hdd_path + "data/music", 1000)
        clf_mfcc.save('clf.h5')
    else:
        print("Restoring models...")
        # MFCC Tensorflow nn
        clf_mfcc = tf.keras.models.load_model('clf.h5')

    # GRAD
    if len(glob.glob("clf_grad.h5")) < 1:
        clf_grad = GRAD.train_grad_nn(ext_hdd_path + "data/speech", ext_hdd_path + "data/music", 1000)
        clf_grad.save('clf_grad.h5')
    else:
        # GRAD Tensorflow nn
        clf_grad = tf.keras.models.load_model('clf_grad.h5')


    i = 0
    while i < 30:
        current_file = "stream_" + str(i)
        radiorec.my_record(station, 3.0, current_file)
        path = "data/test/" + current_file + ".mp3"

        print("Current: " + current_file)

        # Convert streamed mp3 to wav
        wav_path = ac.mp3_to_wav(path)

        # Use features specified in command line arguments
        if "mfcc" in cl_arguments:
            # MFCC classification
            current_mfcc = MFCC.read_mfcc(wav_path)
            # Output for MFCC
            current_duration = util.get_wav_duration(wav_path)
            thread = threading.Thread(target=Output.print_mfcc(current_mfcc, clf_mfcc, current_duration, 9), args=(10,))
            thread.start()
            thread.join()

        if "cfa" or "grad" in cl_arguments:
            # Calculate the spectrogram once => cfa and abl use the same spectrogram
            spectrogram = Processing.cfa_grad_preprocessing(path)

        if "cfa" in cl_arguments:
            # CFA classification
            cfa = CFA.calculate_cfa(path, np.copy(spectrogram))  # np.copy() because numpy arrays are mutable

            # Output
            cfa_thread = threading.Thread(target=Output.print_cfa(cfa, threshold=1.24), args=(10,))
            cfa_thread.start()
            cfa_thread.join()

        if "grad" in cl_arguments:
            # GRAD classification and output
            grad = GRAD.calculate_grad(path, spectrogram)
            grad_thread = threading.Thread(target=Output.print_grad(grad, clf_grad))
            grad_thread.start()
            grad_thread.join()

        # play audio stream
        mixer.music.load(wav_path)
        mixer.music.play()

        # Fadeout after 3 seconds, this call also blocks. This makes sure that the current file is always played to the end
        #mixer.music.fadeout(1000)

        sleep(1)
        print()

        i += 1


station = "fm4"
main(station)
