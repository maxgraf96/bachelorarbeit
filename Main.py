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

station = "oe3"
cl_arguments = sys.argv[1:]
ext_hdd_path = "/media/max/Elements/bachelorarbeit/"

def main():

    # clear previous streams
    for p in Path("data/test").glob("stream*.mp3"):
        p.unlink()

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
        clf_grad = GRAD.train_grad_nn(ext_hdd_path + "data/speech", ext_hdd_path + "data/music", 6000)
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
    cfa_threshold = 1.24

    i = 0
    while True:
        results = []
        current_file = "stream_" + str(i)
        radiorec.my_record(station, 3.0, current_file)
        path = "data/test/" + current_file + ".mp3"

        print("Current: " + current_file)

        # Convert streamed mp3 to wav
        wav_path = ac.mp3_to_16_khz_wav(path)

        # Use features specified in command line arguments
        if is_mfcc:
            # MFCC classification
            current_mfcc = MFCC.read_mfcc(wav_path)
            current_duration = util.get_wav_duration(wav_path)
            result_mfcc = [-1]
            thread_mfcc = threading.Thread(target=Output.print_mfcc(current_mfcc, clf_mfcc, current_duration, result_mfcc, 9), args=(10,))
            thread_mfcc.start()
            thread_mfcc.join()
            results.extend(result_mfcc)

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
                results.extend(result_cfa)


            if is_grad:
                # GRAD classification
                result_grad = [-1]
                grad = GRAD.calculate_grad(path, spectrogram)
                thread_grad = threading.Thread(target=Output.print_grad(grad, clf_grad, result_grad))
                thread_grad.start()
                thread_grad.join()
                results.extend(result_grad)

        # Make a decision and add to blocks
        # Right now we assume that all 3 features are in use
        final_result = 0  # If this value is > 0, we assume that music is played, and if it is < 0 we assume speech
        mfcc = results[0]
        cfa = results[1]
        grad = results[2]

        mfcc_unsure = 0.4 < mfcc < 0.6
        cfa_unsure = cfa_threshold - 1 < cfa < cfa_threshold + 1
        # Check MFCC value -> it works best for identifying music, but not so well for speech
        if mfcc > 0.5:
            final_result += 1
            if mfcc > 0.8:
                final_result += 2
        else:
            final_result -= 1

        cfa_value = 2 if mfcc_unsure else 1
        if cfa_threshold < cfa < cfa_threshold + 1:
            final_result += cfa_value
            if cfa > cfa_threshold + 1:
                final_result += cfa_value
        else:
            final_result -= cfa_value

        grad_value = 0
        if mfcc_unsure and cfa_unsure:
            # Higher weight for grad classifier if the others are unsure. This is because the grad classifier works
            # especially well for identifying speech, whereas the others are better for music classification
            grad_value = 2
        elif mfcc_unsure ^ cfa_unsure:
            grad_value = 1
        if grad > 0.5:
            final_result += grad_value
        else:
            final_result -= grad_value

        # Add to successive blocks
        if final_result > 0:
            succ_music += 1
            succ_speech = 0
        else:
            succ_speech += 1
            succ_music = 0

        result_str = "SPEECH" if final_result <= 0 else "MUSIC"
        print("FINAL RESULT: ", final_result, " => " + result_str)
        print("Successive music blocks: ", succ_music)
        print("Successive speech blocks: ", succ_speech)

        # Play audio stream
        mixer.music.load(ac.mp3_to_22_khz_wav(path))
        mixer.music.play()

        # Block for 1s
        sleep(1.5)
        # mixer.music.fadeout(1500)
        print()
        i += 1

main()
