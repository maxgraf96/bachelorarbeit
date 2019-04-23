import glob
import sys
import threading
import time
from pathlib import Path

import math
import numpy as np
import tensorflow as tf
from pygame import mixer
from joblib import dump, load

import AudioConverter as ac
import Output
import Processing
import radiorec
import util
from features import MFCC, CFA

station = "fm4"
cl_arguments = sys.argv[1:]

# Command line arguments check
is_mfcc = "mfcc" in cl_arguments
is_cfa = "cfa" in cl_arguments

# CFA threshold
cfa_threshold = 3.4

# Livestream or from file
live_stream = False

def clear_streams():
    # clear previous streams
    for p in Path("data/test").glob("*"):
        p.unlink()

def decide(results):
    cfa_value = 0.2

    mfcc = results["mfcc"][0] if is_mfcc else 0
    cfa = results["cfa"][0] if is_cfa else 0

    # If MFCC is sure about music, reduce the CFA influence
    if mfcc > 0.9 and is_cfa:
        cfa_value = 0.1

    if is_cfa:
        if cfa > cfa_threshold:
            cfa = cfa_value
            if cfa > cfa_threshold + 0.5:
                cfa += cfa_value
        else:
            cfa = -cfa_value
            if cfa < cfa_threshold - 0.5:
                cfa -= cfa_value

    final_result = mfcc + cfa

    return final_result

def calc_from_stream(clf_mfcc, scaler_mfcc):
    succ_speech = 0
    succ_music = 0
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
            current_mfcc = MFCC.read_mfcc(sig, rate)
            current_duration = util.get_wav_duration(wav_path)
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(
                target=Output.print_mfcc(current_mfcc, clf_mfcc, scaler_mfcc, current_duration, result_mfcc, 9), args=(10,))

        if is_cfa:
            # CFA classification
            cfa = CFA.calculate_cfa(spec=np.copy(spectrogram))  # np.copy() because numpy arrays are mutable
            result_cfa = [-1]
            thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa, cfa_threshold))

        if is_mfcc:
            thread_mfcc.start()
        if is_cfa:
            thread_cfa.start()

        if is_mfcc:
            thread_mfcc.join()
            results["mfcc"] = result_mfcc
        if is_cfa:
            thread_cfa.join()
            results["cfa"] = result_cfa

        # Make a decision and add to blocks
        final_result = decide(results)

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

def calc_from_file(clf_mfcc, scaler_mfcc):
    succ_speech = 0
    succ_music = 0
    file = "stream_long"
    radiorec.my_record(station, 15, file)
    path = "data/test/" + file + ".mp3"
    wav_path = ac.mp3_to_16_khz_wav(path)

    # Preprocess audio
    sig, rate, frequencies, times, spectrogram = Processing.preprocessing_new(wav_path)

    half_seconds = math.ceil(util.get_wav_duration(wav_path) * 2)

    mixer.music.load(wav_path)
    mixer.music.play()

    time_per_iteration = 0
    i = 0
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
            current_mfcc = MFCC.read_mfcc(sig[startidx:endidx], rate)
            current_duration = 0.5
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(
                target=Output.print_mfcc(current_mfcc, clf_mfcc, scaler_mfcc, current_duration, result_mfcc, 9), args=(10,))

        if is_cfa:
            startidx = math.floor(spectrogram.shape[1] * i / half_seconds)
            endidx = math.ceil(spectrogram.shape[1] * (i + 1) / half_seconds)
            # CFA classification
            cfa = CFA.calculate_cfa(spec=np.copy(
                spectrogram[:, startidx:endidx]))  # np.copy() because numpy arrays are mutable
            result_cfa = [-1]
            thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa, cfa_threshold))

        if is_mfcc:
            thread_mfcc.start()
        if is_cfa:
            thread_cfa.start()

        if is_mfcc:
            thread_mfcc.join()
            results["mfcc"] = result_mfcc
        if is_cfa:
            thread_cfa.join()
            results["cfa"] = result_cfa

        # Make a decision and add to blocks

        final_result = decide(results)

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
        elapsed = end - start
        time_per_iteration += elapsed if i > 1 else 0  # First iteration takes longer as numba caches all the functions
        print("Elapsed Time: ", str(elapsed))
        print()

    print("Average time per iteration: ", str(time_per_iteration / i))

def main():
    # Clear previous streams
    clear_streams()

    # init
    mixer.init()

    # persist classifiers if they do not exist
    # MFCC
    if len(glob.glob("clf_mfcc.h5")) < 1:
        print("Saving model...")
        # Tensorflow nn
        clf_mfcc, scaler_mfcc = MFCC.train_mfcc_nn(util.ext_hdd_path + "data/speech", util.ext_hdd_path + "data/music", 15000, test=False)
        clf_mfcc.save('clf_mfcc.h5')
        dump(scaler_mfcc, "scaler_mfcc.joblib")

    else:
        print("Restoring models...")
        # MFCC Tensorflow nn
        clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')
        scaler_mfcc = load("scaler_mfcc.joblib")

    # Flag for checking if currently playing replacement
    is_replacement = False

    if live_stream:
        calc_from_stream(clf_mfcc, scaler_mfcc)
    else:
        calc_from_file(clf_mfcc, scaler_mfcc)


main()

