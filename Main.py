import glob
import sys
import threading
import time
from pathlib import Path

import math
import numpy as np
import tensorflow as tf
from numba import jit
from pygame import mixer
from joblib import dump, load

import AudioConverter as ac
import Output
import Processing
import radiorec
import util
from features import MFCC, CFA, GRAD

station = "oe1"
cl_arguments = sys.argv[1:]

# Command line arguments check
is_mfcc = "mfcc" in cl_arguments
is_cfa = "cfa" in cl_arguments
is_grad = "grad" in cl_arguments

# CFA threshold
cfa_threshold = 3.4

# Livestream or from file
live_stream = False
from_file = not live_stream

def clear_streams():
    # clear previous streams
    for p in Path("data/test").glob("*"):
        p.unlink()

def calc_from_stream(clf_mfcc, scaler_mfcc, clf_grad, scaler_grad):
    succ_music = 0
    succ_speech = 0
    i = 0
    while True:
        results = {}
        current_file = "stream_" + str(i)
        radiorec.my_record(station, 0.5, current_file)
        path = "data/test/" + current_file + ".mp3"
        wav_path = ac.mp3_to_16_khz_wav(path)

        print("Current: " + current_file)

        # Preprocess audio
        sig, rate, frequencies, times, spectrogram = Processing.preprocessing_new(wav_path)

        # Take time
        start = time.time()

        # Use features specified in command line arguments
        if is_mfcc:
            # MFCC classification
            current_mfcc = MFCC.read_mfcc_new(sig, rate)
            current_duration = util.get_wav_duration(wav_path)
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(
                target=Output.print_mfcc(current_mfcc, clf_mfcc, scaler_mfcc, current_duration, result_mfcc, 9), args=(10,))

        if is_cfa or is_grad:
            if is_cfa:
                # CFA classification
                cfa = CFA.calculate_cfa(spec=np.copy(spectrogram))  # np.copy() because numpy arrays are mutable
                result_cfa = [-1]
                thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa, cfa_threshold))

            if is_grad:
                # GRAD classification
                result_grad = [-1]
                grad = GRAD.calculate_grad(spec=spectrogram)
                thread_grad = threading.Thread(target=Output.print_grad(grad, clf_grad, scaler_grad, result_grad))

        if is_mfcc:
            thread_mfcc.start()
        if is_cfa:
            thread_cfa.start()
        if is_grad:
            thread_grad.start()

        if is_mfcc:
            thread_mfcc.join()
            results["mfcc"] = result_mfcc
        if is_cfa:
            thread_cfa.join()
            results["cfa"] = result_cfa
        if is_grad:
            thread_grad.join()
            results["grad"] = result_grad
        # Make a decision and add to blocks
        # Right now we assume that all 3 features are in use
        final_result = 0  # If this value is > a threshold, we assume that music is played, and if it is < 0 we assume speech

        divisor = 0
        if is_mfcc:
            divisor += 1
        if is_grad:
            divisor += 1
        grad_weight = 0.5
        cfa_value = 0.2

        mfcc = results["mfcc"][0] if is_mfcc else 0
        cfa = results["cfa"][0] if is_cfa else 0
        grad = results["grad"][0] * grad_weight if is_grad else 0

        if mfcc > 0.9:
            cfa_value = 0.1
            divisor -= 1

        if cfa is not 0:
            if cfa > cfa_threshold:
                cfa = cfa_value
                if cfa > 4:
                    cfa += cfa_value
            else:
                cfa = -cfa_value

        if grad < 0.3:
            grad *= 0.5

        if grad is 0:
            divisor += 1

        final_result = (mfcc + grad) / divisor + cfa

        # Add to successive blocks
        if final_result > 0.5:
            succ_music += 1
            succ_speech = 0
        else:
            succ_speech += 1
            succ_music = 0

        result_str = "SPEECH" if final_result <= 0.5 else "MUSIC"
        print("FINAL RESULT: ", final_result, " => " + result_str)
        print("Successive music blocks: ", succ_music)
        print("Successive speech blocks: ", succ_speech)

        # Fadeout the track if the currently played type does not correspond to what we want to hear
        # if "music" in cl_arguments:
        #     if succ_speech > 2 and not is_replacement:
        #         mixer.music.fadeout(300)
        #         ac.mp3_to_22_khz_wav("data/replacements/klangcollage.mp3")
        #         mixer.music.load("data/replacements/klangcollage_22_kHz.wav")
        #         mixer.music.play()
        #         is_replacement = True
        #     if succ_music > 2 and is_replacement:
        #         is_replacement = False
        #         mixer.music.fadeout(300)
        #
        # if not is_replacement:
        #     # Play audio stream
        #     mixer.music.load(ac.mp3_to_22_khz_wav(path))
        #     mixer.music.play()

        mixer.music.load(wav_path)
        mixer.music.play()

        # Clear previous streams on every 10th iteration
        if i % 10 == 0:
            clear_streams()

        i += 1

        # Measure execution time
        end = time.time()
        print("Elapsed Time: ", str(end - start))
        print()

def calc_from_file(clf_mfcc, scaler_mfcc, clf_grad, scaler_grad):
    file = "stream_long"
    radiorec.my_record(station, 15, file)
    path = "data/test/" + file + ".mp3"
    wav_path = ac.mp3_to_16_khz_wav(path)

    # Preprocess audio
    sig, rate, frequencies, times, spectrogram = Processing.preprocessing_new(wav_path)

    half_seconds = math.ceil(util.get_wav_duration(wav_path) * 2)

    mixer.music.load(wav_path)
    mixer.music.play()

    succ_music = 0
    succ_speech = 0

    for i in range(half_seconds):
        # Take time
        start = time.time()

        results = {}
        print("Current: ")

        # Use features specified in command line arguments
        if is_mfcc:
            # MFCC classification
            startidx = math.floor(len(sig) * i / half_seconds)
            endidx = math.ceil(len(sig) * (i + 1) / half_seconds)
            current_mfcc = MFCC.read_mfcc_new(sig[startidx:endidx], rate)
            current_duration = 0.5
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(
                target=Output.print_mfcc(current_mfcc, clf_mfcc, scaler_mfcc, current_duration, result_mfcc, 9), args=(10,))

        if is_cfa or is_grad:
            startidx = math.floor(spectrogram.shape[1] * i / half_seconds)
            endidx = math.ceil(spectrogram.shape[1] * (i + 1) / half_seconds)
            print(endidx)
            if is_cfa:
                # CFA classification
                cfa = CFA.calculate_cfa(spec=np.copy(
                    spectrogram[:, startidx:endidx]))  # np.copy() because numpy arrays are mutable
                result_cfa = [-1]
                thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa, cfa_threshold))

            if is_grad:
                # GRAD classification
                result_grad = [-1]
                grad = GRAD.calculate_grad(spec=np.copy(spectrogram[:, startidx:endidx]))
                thread_grad = threading.Thread(target=Output.print_grad(grad, clf_grad, scaler_grad, result_grad))

        if is_mfcc:
            thread_mfcc.start()
        if is_cfa:
            thread_cfa.start()
        if is_grad:
            thread_grad.start()

        if is_mfcc:
            thread_mfcc.join()
            results["mfcc"] = result_mfcc
        if is_cfa:
            thread_cfa.join()
            results["cfa"] = result_cfa
        if is_grad:
            thread_grad.join()
            results["grad"] = result_grad

        # Make a decision and add to blocks
        # Right now we assume that all 3 features are in use
        final_result = 0  # If this value is > a threshold, we assume that music is played, and if it is < 0 we assume speech

        divisor = 0
        if is_mfcc:
            divisor += 1
        if is_grad:
            divisor += 1
        grad_weight = 0.3
        cfa_value = 0.2

        mfcc = results["mfcc"][0] if is_mfcc else 0
        cfa = results["cfa"][0] if is_cfa else 0
        grad = results["grad"][0] * grad_weight if is_grad else 0

        if mfcc > 0.9 and is_cfa:
            cfa_value = 0.1
            divisor -= 1

        if cfa is not 0:
            if cfa > cfa_threshold:
                cfa = cfa_value
                if cfa > 4:
                    cfa += cfa_value
            else:
                cfa = -cfa_value

        if grad < 0.3:
            grad *= 0.5

        if grad is 0:
            divisor += 1

        final_result = (mfcc + grad) / divisor + cfa

        # Add to successive blocks
        if final_result > 0.5:
            succ_music += 1
            succ_speech = 0
        else:
            succ_speech += 1
            succ_music = 0

        result_str = "SPEECH" if final_result <= 0.5 else "MUSIC"
        print("FINAL RESULT: ", final_result, " => " + result_str)
        print("Successive music blocks: ", succ_music)
        print("Successive speech blocks: ", succ_speech)

        # Fadeout the track if the currently played type does not correspond to what we want to hear
        # if "music" in cl_arguments:
        #     if succ_speech > 2 and not is_replacement:
        #         mixer.music.fadeout(300)
        #         ac.mp3_to_22_khz_wav("data/replacements/klangcollage.mp3")
        #         mixer.music.load("data/replacements/klangcollage_22_kHz.wav")
        #         mixer.music.play()
        #         is_replacement = True
        #     if succ_music > 2 and is_replacement:
        #         is_replacement = False
        #         mixer.music.fadeout(300)
        #
        # if not is_replacement:
        #     # Play audio stream
        #     mixer.music.load(ac.mp3_to_22_khz_wav(path))
        #     mixer.music.play()

        i += 1

        # Measure execution time
        end = time.time()
        print("Elapsed Time: ", str(end - start))
        print()

def main():
    # Clear previous streams
    clear_streams()

    # init
    mixer.init()

    # persist classifiers if they do not exist
    # MFCC
    if len(glob.glob("clf_mfcc.h5")) < 1:
        print("Saving model...")
        # Tensorflow nn, Note: Only saves the network currently (pca is discarded)
        clf_mfcc, scaler_mfcc = MFCC.train_mfcc_nn(util.ext_hdd_path + "data/speech", util.ext_hdd_path + "data/music", 150000)
        clf_mfcc.save('clf_mfcc.h5')
        dump(scaler_mfcc, "scaler_mfcc.joblib")
        # Store pca in joblib
        #dump(pca_mfcc, 'pca_mfcc.joblib')

    else:
        print("Restoring models...")
        # MFCC Tensorflow nn
        clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')
        scaler_mfcc = load("scaler_mfcc.joblib")
        #pca_mfcc = load('pca_mfcc.joblib')

    # GRAD
    if len(glob.glob("clf_grad.h5")) < 1:
        clf_grad, scaler_grad = GRAD.train_grad_nn(util.ext_hdd_path + "data/speech", util.ext_hdd_path + "data/music", 150000)
        clf_grad.save('clf_grad.h5')
        dump(scaler_grad, "scaler_grad.joblib")
        # Store pca in joblib
        #dump(pca_grad, 'pca_grad.joblib')
    else:
        # GRAD Tensorflow nn
        clf_grad = tf.keras.models.load_model('clf_grad.h5')
        scaler_grad = load("scaler_grad.joblib")

    # Flag for checking if currently playing replacement
    is_replacement = False

    if from_file:
        calc_from_file(clf_mfcc, scaler_mfcc, clf_grad, scaler_grad)

    if live_stream:
        calc_from_stream(clf_mfcc, scaler_mfcc, clf_grad, scaler_grad)


main()

