import glob

import Output
from pathlib import Path
from pygame import mixer
import radiorec
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
url = 'https://fm4shoutcast.sf.apa.at'
#url = 'http://c14000-l.i.core.cdn.streamfarm.net/14000cina/live/3212erf_96/live_de_96.mp3'
mixer.init()
i = 0


# persist classifier
#if len(glob.glob("clf.joblib")) < 1:
if len(glob.glob("clf.h5")) < 1:
    print("Saving model...")
    # kNN
    # clf, pca = MFCC.train_mfcc_knn("data/speech", "data/music", 3000)
    # dump([clf, pca], 'clf.joblib')

    # Tensorflow nn, Note: Only saves the network currently (pca is discarded)
    clf, pca = MFCC.train_mfcc_nn("data/speech", "data/music", 10000)
    clf.save('clf.h5')


else:
    print("Restoring model...")
    # kNN
    #[clf, pca] = load('clf.joblib')

    # Tensorflow nn
    #clf = tf.keras.models.load_model('clf.h5')

    # Train CFA model
    clf, pca = CFA.train_cfa_nn("data/speech", "data/music", 100)



while i < 3:
    currentFile = "stream_" + str(i)
    radiorec.my_record(url, 3.0, currentFile)
    path = "data/test/" + currentFile + ".mp3"

    # for mfcc classification
    wav_path = ac.mp3_to_wav(path)

    current_mfcc = MFCC.read_mfcc(wav_path)

    # result = clf.predict(
    #     pca.transform(
    #         sklearn.preprocessing.scale(current_mfcc, axis=1)
    #     )
    # )

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

    # play audio stream
    mixer.music.load(wav_path)
    mixer.music.play()

    # Output results
    current_duration = MFCC.get_wav_duration(wav_path)
    thread = threading.Thread(target=Output.print_mfcc(current_mfcc, clf, current_duration, 9), args=(10,))
    thread.start()
    thread.join()
    print("Output thread finished...")

    i = i + 1
