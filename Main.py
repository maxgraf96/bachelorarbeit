import os

from pyAudioAnalysis import audioTrainTest as aT
from pathlib import Path
import time
import radiorec
from pygame import mixer

#aT.featureAndTrain(["data/music","data/speech/little"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

start = time.time()

#clear previous streams
for p in Path("data/test").glob("stream*.mp3"):
    p.unlink()

url = 'https://fm4shoutcast.sf.apa.at'
mixer.init()
i = 0
while i < 10:
    currentFile = "stream_" + str(i)
    radiorec.my_record(url, 3.0, currentFile)
    result = aT.fileClassification("data/test/" + currentFile + ".mp3", "svmSMtemp", "svm")
    music_speech_tuple = result[1]
    if music_speech_tuple[0] > music_speech_tuple[1]:
        renamed = "data/test/" + currentFile + "_mu.mp3"
        os.rename("data/test/" + currentFile + ".mp3", renamed)
    else:
        renamed = "data/test/" + currentFile + "_sp.mp3"
        os.rename("data/test/" + currentFile + ".mp3", renamed)

    # play audio stream
    mixer.music.load(renamed)
    mixer.music.play()
    i = i + 1

    print(result)


end = time.time()
print(end - start)

