import glob
import os

import sklearn

from pyAudioAnalysis import audioTrainTest as aT
from pathlib import Path
from pygame import mixer
import time
import radiorec
import pyAudioAnalysis.audioBasicIO as abio
import MFCC
import AudioConverter as ac
import scipy.io.wavfile as wav
import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA

#aT.featureAndTrain(["data/music/verylittle","data/speech/verylittle"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

# measure time
#start = time.time()

# clear previous streams
for p in Path("data/test").glob("stream*.mp3"):
    p.unlink()

# init
url = 'https://fm4shoutcast.sf.apa.at'
url = 'http://c14000-l.i.core.cdn.streamfarm.net/14000cina/live/3212erf_96/live_de_96.mp3'
mixer.init()
i = 0


# persist svm
if len(glob.glob("mfcc_svm.joblib")) < 1:
    mfcc_svm, pca = MFCC.train_mfcc_svm("data/speech", "data/music", 3000)
    dump([mfcc_svm, pca], 'mfcc_svm.joblib')
else:
    [mfcc_svm, pca] = load('mfcc_svm.joblib')

while i < 10:
    currentFile = "stream_" + str(i)
    radiorec.my_record(url, 3.0, currentFile)
    path = "data/test/" + currentFile + ".mp3"

    # for mfcc classification
    wav_path = ac.mp3_to_wav(path)

    current_mfcc = MFCC.read_mfcc(wav_path)

    result_mfcc = mfcc_svm.predict(
        pca.transform(
            sklearn.preprocessing.scale(current_mfcc, axis=1)
        )
    )
    ones = np.count_nonzero(result_mfcc)
    print("Ones: " + str(ones))
    print("Zeros: " + str(len(result_mfcc) - ones))
    print("Music: " + str(ones / len(result_mfcc)))
    #print(result_mfcc)

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
    #mixer.music.load(renamed)
    mixer.music.load(path)
    mixer.music.play()
    i = i + 1

    #print(result)


#end = time.time()
#print(end - start)

