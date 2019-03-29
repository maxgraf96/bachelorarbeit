import glob
import sys
from time import sleep

import Output
from pathlib import Path
from pygame import mixer
import radiorec
import util
from features import MFCC, CFA
import AudioConverter as ac
import tensorflow as tf
import threading

#aT.featureAndTrain(["data/music/verylittle","data/speech/verylittle"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

# measure time
#start = time.time()

# clear previous streams
for p in Path("data/test").glob("stream*.mp3"):
    p.unlink()

# init
mixer.init()
i = 0

station = "radiovorarlberg"


# persist classifier
#if len(glob.glob("clf.joblib")) < 1:
if len(glob.glob("clf.h5")) < 1:
    print("Saving model...")
    # kNN
    # clf, pca = MFCC.train_mfcc_knn("data/speech", "data/music", 3000)
    # dump([clf, pca], 'clf.joblib')

    # Tensorflow nn, Note: Only saves the network currently (pca is discarded)
    clf_mfcc, pca = MFCC.train_mfcc_nn("data/speech", "data/music", 10000)
    clf_mfcc.save('clf.h5')

    # clf_cfa = CFA.train_cfa_nn("data/speech", "data/music", 30)
    # clf_cfa.save('clf_cfa.h5')


else:
    print("Restoring model...")
    # kNN
    #[clf, pca] = load('clf.joblib')

    # Tensorflow nn
    clf_mfcc = tf.keras.models.load_model('clf.h5')

    #clf_cfa = tf.keras.models.load_model('clf_cfa.h5')

#cfa = CFA.calculate_cfas("data/speech", "data/music", 10)

while i < 10:
    currentFile = "stream_" + str(i)
    radiorec.my_record(station, 3.0, currentFile)
    path = "data/test/" + currentFile + ".mp3"

    # Convert streamed mp3 to wav
    wav_path = ac.mp3_to_wav(path)

    # MFCC classification
    current_mfcc = MFCC.read_mfcc(wav_path)

    # CFA classification
    cfa = CFA.calculate_cfa(path)

    # play audio stream
    mixer.music.load(wav_path[:-4] + "_11_kHz.wav")
    mixer.music.play()

    # Output results
    current_duration = util.get_wav_duration(wav_path)
    thread = threading.Thread(target=Output.print_mfcc(current_mfcc, clf_mfcc, current_duration, 9), args=(10,))
    thread.start()
    thread.join()

    # Output for CFA
    cfa_thread = threading.Thread(target=Output.print_cfa(cfa), args=(10,))
    cfa_thread.start()
    cfa_thread.join()
    #print("Output thread finished...")

    sleep(.9)
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