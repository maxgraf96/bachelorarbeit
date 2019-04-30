import glob
import sys
import threading
import time
from pathlib import Path

import math
import numpy as np
from pygame import mixer
from joblib import dump, load

import AudioConverter as ac
import Output
import Processing
import radiorec
import util
from features import MFCC, CFA

# CFA threshold
cfa_threshold = 3.2

def decide(results):
    mfcc = results["mfcc"][0] if "mfcc" in results else 0
    cfa_result = results["cfa"][0] if "cfa" in results else 0
    divisor = 2 if "mfcc" in results and "cfa" in results else 1
    bias = 0

    # Add bias
    if mfcc > 0.5:
        bias += 0.2

    final_result = (mfcc + cfa_result) / divisor + bias

    return final_result

def calc_from_stream(station, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa, listening_preference):
    mixer.init()
    succ_speech = 0
    succ_music = 0
    # Flag for checking if currently playing replacement
    is_replacement = False
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
            cfa, peakis = CFA.calculate_cfa(spec=spectrogram, threshold=cfa_threshold)
            result_cfa = [-1]
            thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa))

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
        if listening_preference == "music":
            if succ_speech > 4 and not is_replacement:
                mixer.music.fadeout(300)
                ac.mp3_to_22_khz_wav("data/replacements/klangcollage.mp3")
                mixer.music.load("data/replacements/klangcollage_22_kHz.wav")
                mixer.music.play()
                is_replacement = True
            if succ_music > 4 and is_replacement:
                is_replacement = False
                mixer.music.fadeout(300)

        if not is_replacement:
            # Play audio stream
            mixer.music.load(ac.mp3_to_22_khz_wav(path))
            mixer.music.play()

        i += 1

        # Measure execution time
        end = time.time()
        print("Elapsed Time: ", str(end - start))
        print()

def calc_from_file(file, filename, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa):
    speech_music_map = []
    succ_speech = 0
    succ_music = 0
    path = file
    wav_path = ac.mp3_to_16_khz_wav(path)

    # Preprocess audio
    sig, rate, frequencies, times, spectrogram = Processing.preprocessing_new(wav_path)

    half_seconds = math.ceil(util.get_wav_duration(wav_path) * 2)

    mixer.init()
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
            cfa, peakis = CFA.calculate_cfa(spec=spectrogram[:, startidx:endidx], threshold=cfa_threshold)
            result_cfa = [-1]
            thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa))

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
            speech_music_map.append(1)
        else:
            succ_speech += 1
            succ_music = 0
            speech_music_map.append(0)

        result_str = "SPEECH" if final_result <= 0.5 else "MUSIC"
        print("FINAL RESULT: ", final_result, " => " + result_str)
        print("Successive music blocks: ", succ_music)
        print("Successive speech blocks: ", succ_speech)

        i += 1

        # Measure execution time
        end = time.time()
        elapsed = end - start
        time_per_iteration += elapsed if i > 1 else 0  # First iteration takes longer as numba caches all the functions
        print("Elapsed Time: ", str(elapsed))
        print()

    x = np.arange(len(speech_music_map)) / 2  # Convert from samples (every 0.5s to seconds)
    util.plot_speech_music_map(filename, x, speech_music_map, save_csv=True)

    print("Average time per iteration: ", str(time_per_iteration / i))

    return x, speech_music_map

def clear_streams():
    # clear previous streams
    for p in Path("data/test").glob("*"):
        p.unlink()

