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

import threading
import time
from pathlib import Path

import math
import numpy as np
from pygame import mixer

import AudioConverter as ac
import Output
import Processing
import radiorec
import util
from features import MFCC, CFA

# CFA threshold
cfa_threshold = 3.25

def decide(results):
    """
    Takes the list of results as input and makes a final decision
    :param results: The list of results. Comprised of the MFCC and CFA values
    :return: A floating point number. If < 0.5 the currently observed chunks is classified as speech, else it is
    classified as music
    """
    mfcc = results["mfcc"][0] if "mfcc" in results else 0
    cfa_result = results["cfa"] if "cfa" in results else 0
    divisor = 2 if "mfcc" in results and "cfa" in results else 1
    bias = 0

    # Add bias
    if mfcc > 0.5:
        bias += 0.2

    final_result = (mfcc + cfa_result) / divisor + bias

    return final_result

def calc_from_stream(station, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa, listening_preference, replacement_path):
    """
    Streams 0.5 seconds of audio and classifies the data.
    :param station: The radio station from which the data should be streamed. See "radiorec_settings.ini" for available stations.
    :param clf_mfcc: The trained neural network classifier for MFCC classification
    :param scaler_mfcc: The scaler instance used to scaled the original MFCC training data
    :param is_mfcc: If the MFCC value should be calculated and taken into consideration for the final result
    :param is_cfa: If the CFA value should be calculated and taken into consideration for the final result
    :param listening_preference: The listening preference specifies whether spoken segments or music segments should be kept
    :param replacement_path: The path to the audio file that is played when the unwanted class is detected
    :return: void
    """
    mixer.init(frequency=16000, channels=1)
    succ_speech = 0
    succ_music = 0
    # Flag for checking if currently playing replacement
    is_replacement = False
    i = 0
    while True:
        results = {}
        current_file = "stream_" + str(i)
        radiorec.record(station, 0.5, current_file)
        path = "data/test/" + current_file + ".mp3"
        wav_path = ac.mp3_to_16_khz_wav(path)

        print("Current: " + current_file)

        # Preprocess audio
        sig, rate, frequencies, times, spectrogram = Processing.preprocessing(wav_path)

        # Take time
        start = time.time()

        # Use features specified in command line arguments
        if is_mfcc:
            # MFCC classification
            current_mfcc = MFCC.read_mfcc(sig, rate)
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(
                target=Output.print_mfcc(current_mfcc, clf_mfcc, scaler_mfcc, result_mfcc, 9), args=(10,))

        if is_cfa:
            # CFA classification
            cfa, peakis = CFA.calculate_cfa(spec=spectrogram, threshold=cfa_threshold)
            result = round(cfa, 4)
            results["cfa"] = result
            print("CFA Music: " + str(result))

        if is_mfcc:
            thread_mfcc.start()

        if is_mfcc:
            thread_mfcc.join()
            results["mfcc"] = result_mfcc


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

        # Fadeout the track if the currently played type does not correspond to what was specified via the command line
        # 4 blocks provide a good user experience because sometimes single or double blocks are classified wrong
        if listening_preference == "music":
            if succ_speech > 4 and not is_replacement:
                mixer.music.load(replacement_path)
                mixer.music.fadeout(300)
                mixer.music.play()
                is_replacement = True
            if succ_music > 4 and is_replacement:
                is_replacement = False
                mixer.music.fadeout(300)

        if listening_preference == "speech":
            if succ_music > 4 and not is_replacement:
                mixer.music.load(replacement_path)
                mixer.music.fadeout(300)
                mixer.music.play()
                is_replacement = True
            if succ_speech > 4 and is_replacement:
                is_replacement = False
                mixer.music.fadeout(300)

        if not is_replacement:
            # Play audio stream
            mixer.music.load(wav_path)
            mixer.music.play()

        i += 1

        # Measure execution time
        end = time.time()
        print("Elapsed Time: ", str(end - start))
        print()

def calc_from_file(file, filename, clf_mfcc, scaler_mfcc, is_mfcc, is_cfa):
    """
    Classifies an mp3 audio file and saves the results in a CSV and PNG file. See "plots" folder.
    :param file: The file to classify.
    :param filename: The filename. Used to save the CSV and PNG files.
    :param clf_mfcc: The trained neural network classifier for MFCC classification
    :param scaler_mfcc: The scaler instance used to scaled the original MFCC training data
    :param is_mfcc: If the MFCC value should be calculated and taken into consideration for the final result
    :param is_cfa: If the CFA value should be calculated and taken into consideration for the final result
    :return: An array containing the range of seconds of the duration of the file in steps of 0.5s. The generated speech_music_map
    """
    speech_music_map = []
    succ_speech = 0
    succ_music = 0
    path = file
    wav_path = ac.mp3_to_16_khz_wav(path)

    # Preprocess audio
    sig, rate, frequencies, times, spectrogram = Processing.preprocessing(wav_path)

    half_seconds = math.ceil(util.get_wav_duration(wav_path) * 2)

    mixer.init(frequency=16000, channels=1)
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
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(
                target=Output.print_mfcc(current_mfcc, clf_mfcc, scaler_mfcc, result_mfcc, 9), args=(10,))

        if is_cfa:
            startidx = math.floor(spectrogram.shape[1] * i / half_seconds)
            endidx = math.ceil(spectrogram.shape[1] * (i + 1) / half_seconds)
            # CFA classification
            cfa, peakis = CFA.calculate_cfa(spec=spectrogram[:, startidx:endidx], threshold=cfa_threshold)
            result = round(cfa, 4)
            results["cfa"] = result
            print("CFA Music: " + str(result))

        if is_mfcc:
            thread_mfcc.start()
            thread_mfcc.join()
            results["mfcc"] = result_mfcc

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

    # Save CSV and PNG of sequence of classified data (speech_music_map)
    x = np.arange(len(speech_music_map)) / 2  # Convert from samples (every 0.5s to seconds)
    util.plot_speech_music_map(filename, x, speech_music_map, save_csv=True)

    print("Average time per iteration: ", str(time_per_iteration / i))

    return x, speech_music_map

def clear_streams():
    """
    Clear all files from the "data/test" folder (contains previous streams)
    :return:
    """
    for p in Path("data/test").glob("*"):
        p.unlink()

