import glob
import sys
import time
from time import sleep

import math
import numpy as np
import scipy

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

station = "oe1"
cl_arguments = sys.argv[1:]
ext_hdd_path = "/media/max/Elements/bachelorarbeit/"

def clear_streams():
    # clear previous streams
    for p in Path("data/test").glob("*"):
        p.unlink()

def main():

    livestream = False

    # Clear previous streams
    clear_streams()

    # init
    mixer.init()

    # persist classifiers if they do not exist
    # MFCC
    if len(glob.glob("clf_mfcc.h5")) < 1:
        print("Saving model...")
        # Tensorflow nn, Note: Only saves the network currently (pca is discarded)
        clf_mfcc, pca = MFCC.train_mfcc_nn(ext_hdd_path + "data/speech", ext_hdd_path + "data/music", 6000)
        clf_mfcc.save('clf_mfcc.h5')
    else:
        print("Restoring models...")
        # MFCC Tensorflow nn
        clf_mfcc = tf.keras.models.load_model('clf_mfcc.h5')

    # GRAD
    if len(glob.glob("clf_grad.h5")) < 1:
        clf_grad = GRAD.train_grad_nn(ext_hdd_path + "data/speech", ext_hdd_path + "data/music", 60000)
        clf_grad.save('clf_grad.h5')
    else:
        # GRAD Tensorflow nn
        clf_grad = tf.keras.models.load_model('clf_grad.h5')

    # Command line arguments check
    is_mfcc = "mfcc" in cl_arguments
    is_cfa = "cfa" in cl_arguments
    is_grad = "grad" in cl_arguments

    # Memory for successive block of speech and music
    succ_speech = 0
    succ_music = 0

    # CFA threshold
    cfa_threshold = 3.4

    # Flag for checking if currently playing replacement
    is_replacement = False



    file = "stream_long"
    radiorec.my_record(station, 15, file)
    path = "data/test/" + file + ".mp3"
    wav_path = ac.mp3_to_22_khz_wav(path)

    (rate, signal) = scipy.io.wavfile.read(wav_path)
    sig = np.array(signal)

    # Convert signal to mono
    sig = util.stereo2mono(sig)

    # Apply noise gate
    noise_gate_level = 0.5
    sig[sig < noise_gate_level] = 0

    # Estimate the spectrogram using a Hanning window
    window = np.hanning(256)  # 1024 samples correspond to ~ 100ms

    # Calculate the spectrogram using stft and emphasize local maxima
    frequencies, times, spectrogram = scipy.signal.stft(sig, fs=rate, window=window, nperseg=256)

    if rate < 11025:
        raise ValueError("The sampling rate of the incoming signal is too low for Continuous Frequency Activation and GRAD processing.")

    # Cut the spectrogram to 11khz for cfa and grad processing
    # NOTE: Assuming that the frequencies are distributed linearly along the spectrogram
    upper_limit_idx = np.argmin(np.abs(frequencies - (11025 / 2)))
    spectrogram = spectrogram[:upper_limit_idx, :]

    half_seconds = math.ceil(util.get_wav_duration(wav_path) * 2)

    mixer.music.load(wav_path)
    mixer.music.play()

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
                target=Output.print_mfcc(current_mfcc, clf_mfcc, current_duration, result_mfcc, 9), args=(10,))

        if is_cfa or is_grad:
            startidx = math.floor(spectrogram.shape[1] * i / half_seconds)
            endidx = math.ceil(spectrogram.shape[1] * (i + 1) / half_seconds)
            print(endidx)
            if is_cfa:
                # CFA classification
                cfa = CFA.calculate_cfa(path, np.copy(spectrogram[:, startidx:endidx]))  # np.copy() because numpy arrays are mutable
                result_cfa = [-1]
                thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa, cfa_threshold))


            if is_grad:
                # GRAD classification
                result_grad = [-1]
                grad = GRAD.calculate_grad(path, np.copy(spectrogram[:, startidx:endidx]))
                thread_grad = threading.Thread(target=Output.print_grad(grad, clf_grad, result_grad))


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

        # mfcc_unsure = 0.4 < mfcc < 0.6
        # cfa_unsure = cfa_threshold - 1 < cfa < cfa_threshold + 1 if cfa else False
        # # Check MFCC value -> it works best for identifying music, but not so well for speech
        # if mfcc > 0.5:
        #     final_result += 1
        #     if mfcc > 0.8:
        #         final_result += 2
        # else:
        #     final_result -= 1
        #
        # cfa_value = 2 if mfcc_unsure else 1
        # if cfa_threshold < cfa < cfa_threshold + 1:
        #     final_result += cfa_value
        #     if cfa > cfa_threshold + 1:
        #         final_result += cfa_value
        # else:
        #     final_result -= cfa_value
        #
        # grad_value = 0
        # if mfcc_unsure and cfa_unsure:
        #     # Higher weight for grad classifier if the others are unsure. This is because the grad classifier works
        #     # especially well for identifying speech, whereas the others are better for music classification
        #     grad_value = 2
        # elif mfcc_unsure ^ cfa_unsure:
        #     grad_value = 1
        # if grad > 0.5:
        #     final_result += grad_value
        # else:
        #     final_result -= grad_value

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







    i = 0
    if livestream:
        while True:
            results = {}
            current_file = "stream_" + str(i)
            radiorec.my_record(station, 0.5, current_file)
            path = "data/test/" + current_file + ".mp3"

            print("Current: " + current_file)

            # Take time
            start = time.time()

            # Use features specified in command line arguments
            if is_mfcc:
                # MFCC classification
                # Convert streamed mp3 to wav
                wav_path = ac.mp3_to_16_khz_wav(path)
                current_mfcc = MFCC.read_mfcc(wav_path)
                current_duration = util.get_wav_duration(wav_path)
                result_mfcc = [-1]
                thread_mfcc = threading.Thread(target=Output.print_mfcc(current_mfcc, clf_mfcc, current_duration, result_mfcc, 9), args=(10,))
                thread_mfcc.start()
                thread_mfcc.join()
                results["mfcc"] = result_mfcc

            if is_cfa or is_grad:
                # Calculate the spectrogram once => cfa and abl use the same spectrogram
                spectrogram = Processing.cfa_grad_preprocessing(path)
                if is_cfa:
                    # CFA classification
                    cfa = CFA.calculate_cfa(path, np.copy(spectrogram))  # np.copy() because numpy arrays are mutable
                    result_cfa = [-1]
                    thread_cfa = threading.Thread(target=Output.print_cfa, args=(cfa, result_cfa, cfa_threshold))
                    thread_cfa.start()
                    thread_cfa.join()
                    results["cfa"] = result_cfa


                if is_grad:
                    # GRAD classification
                    result_grad = [-1]
                    grad = GRAD.calculate_grad(path, spectrogram)
                    thread_grad = threading.Thread(target=Output.print_grad(grad, clf_grad, result_grad))
                    thread_grad.start()
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

            # mfcc_unsure = 0.4 < mfcc < 0.6
            # cfa_unsure = cfa_threshold - 1 < cfa < cfa_threshold + 1 if cfa else False
            # # Check MFCC value -> it works best for identifying music, but not so well for speech
            # if mfcc > 0.5:
            #     final_result += 1
            #     if mfcc > 0.8:
            #         final_result += 2
            # else:
            #     final_result -= 1
            #
            # cfa_value = 2 if mfcc_unsure else 1
            # if cfa_threshold < cfa < cfa_threshold + 1:
            #     final_result += cfa_value
            #     if cfa > cfa_threshold + 1:
            #         final_result += cfa_value
            # else:
            #     final_result -= cfa_value
            #
            # grad_value = 0
            # if mfcc_unsure and cfa_unsure:
            #     # Higher weight for grad classifier if the others are unsure. This is because the grad classifier works
            #     # especially well for identifying speech, whereas the others are better for music classification
            #     grad_value = 2
            # elif mfcc_unsure ^ cfa_unsure:
            #     grad_value = 1
            # if grad > 0.5:
            #     final_result += grad_value
            # else:
            #     final_result -= grad_value

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

            mixer.music.load(ac.mp3_to_22_khz_wav(path))
            mixer.music.play()

            # Clear previous streams on every 10th iteration
            if i % 10 == 0:
                clear_streams()

            i += 1

            # Measure execution time
            end = time.time()
            print("Elapsed Time: ", str(end - start))
            print()

main()
