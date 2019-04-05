import glob
from time import sleep

import numpy as np
import Output
from pathlib import Path
from pygame import mixer

import Processing
import radiorec
import util
from features import MFCC, CFA, ABL
import AudioConverter as ac
import tensorflow as tf
import threading
import Controls

def main(station):

    # clear previous streams
    for p in Path("data/test").glob("stream*.mp3"):
        p.unlink()

    # init
    mixer.init()

    # persist classifier
    # if len(glob.glob("clf.joblib")) < 1:
    if len(glob.glob("clf.h5")) < 1:
        print("Saving model...")
        # kNN
        # clf, pca = MFCC.train_mfcc_knn("data/speech", "data/music", 3000)
        # dump([clf, pca], 'clf.joblib')

        # Tensorflow nn, Note: Only saves the network currently (pca is discarded)
        clf_mfcc, pca = MFCC.train_mfcc_nn("data/speech", "data/music", 1000)
        clf_mfcc.save('clf.h5')

        # clf_cfa = CFA.train_cfa_nn("data/speech", "data/music", 30)
        # clf_cfa.save('clf_cfa.h5')


    else:
        print("Restoring model...")
        # kNN
        # [clf, pca] = load('clf.joblib')

        # Tensorflow nn
        # clf_mfcc = tf.keras.models.load_model('clf.h5')

        # clf_cfa = tf.keras.models.load_model('clf_cfa.h5')

    # cfa = CFA.calculate_cfas("data/speech", "data/music", 10)
    # trn, lbls = ABL.calculate_abls("data/speech", "data/music", 20)

    i = 0
    while i < 30:
        currentFile = "stream_" + str(i)
        radiorec.my_record(station, 3.0, currentFile)
        path = "data/test/" + currentFile + ".mp3"

        # Convert streamed mp3 to wav
        wav_path = ac.mp3_to_wav(path)

        # MFCC classification
        # current_mfcc = MFCC.read_mfcc(wav_path)

        # Calculate the spectrogram once => cfa and abl use the same spectrogram
        spectrogram = Processing.cfa_abl_preprocessing(path)

        # CFA classification
        cfa = CFA.calculate_cfa(path, np.copy(spectrogram))  # np.copy() because numpy arrays are mutable

        # ABL classification and output
        abl = ABL.calculate_abl(path, spectrogram)

        # Output for MFCC
        # current_duration = util.get_wav_duration(wav_path)
        # thread = threading.Thread(target=Output.print_mfcc(current_mfcc, clf_mfcc, current_duration, 9), args=(10,))
        # thread.start()
        # thread.join()

        # Output for CFA
        cfa_thread = threading.Thread(target=Output.print_cfa(cfa), args=(10,))
        cfa_thread.start()
        cfa_thread.join()

        # play audio stream
        mixer.music.load(wav_path[:-4] + "_11_kHz.wav")
        mixer.music.play()

        # Fadeout after 3 seconds, this call also blocks. This makes sure that the current file is always played to the end
        mixer.music.fadeout(3000)

        i += 1

    '''
    result = aT.fileClassification(path, "svmSMtemp", "svm")

    music_speech_tuple = result[1]
    if music_speech_tuple[0] > music_speech_tuple[1]:
        renamed = "data/test/" + currentFile + "_mu.mp3"
        #os.rename(path, renamed)
    else:
        renamed = "data/test/" + currentFile + "_sp.mp3"
        #os.rename(path, renamed)

    '''

station = "fm4"

#key_thread = threading.Thread(target=Controls.start_listener(), args=(10,))
main_thread = threading.Thread(target=main(station), args=(10,))

main_thread.start()
#key_thread.start()

main_thread.join()
#key_thread.join()

