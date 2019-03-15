import os

from pyAudioAnalysis import audioTrainTest as aT
from pathlib import Path
from pygame import mixer
import time
import radiorec
import librosa
import pyAudioAnalysis.audioBasicIO as abio

#aT.featureAndTrain(["data/music","data/speech/little"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

# measure time
start = time.time()

# clear previous streams
for p in Path("data/test").glob("stream*.mp3"):
    p.unlink()

# init
url = 'https://fm4shoutcast.sf.apa.at'
mixer.init()
i = 0

while i < 10:
    currentFile = "stream_" + str(i)
    radiorec.my_record(url, 3.0, currentFile)
    path = "data/test/" + currentFile + ".mp3"
    result = aT.fileClassification(path, "svmSMtemp", "svm")


    # MFCC
    np_dat = abio.stereo2mono(abio.readAudioFile(path)[1]).astype("float32")
    mfcc = librosa.feature.mfcc(np_dat)
    #print(mfcc)

    music_speech_tuple = result[1]
    if music_speech_tuple[0] > music_speech_tuple[1]:
        renamed = "data/test/" + currentFile + "_mu.mp3"
        #os.rename(path, renamed)
    else:
        renamed = "data/test/" + currentFile + "_sp.mp3"
        #os.rename(path, renamed)

    # play audio stream
    #mixer.music.load(renamed)
    mixer.music.load(path)
    mixer.music.play()
    i = i + 1

    print(result)


end = time.time()
print(end - start)

