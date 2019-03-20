import glob
import os

from pyAudioAnalysis import audioTrainTest as aT
from pathlib import Path
from pygame import mixer
import time
import radiorec
import pyAudioAnalysis.audioBasicIO as abio
import MFCC
import AudioConverter as ac
import scipy.io.wavfile as wav
from joblib import dump, load

#aT.featureAndTrain(["data/music/verylittle","data/speech/verylittle"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

# measure time
#start = time.time()

# clear previous streams
for p in Path("data/test").glob("stream*.mp3"):
    p.unlink()

# init
url = 'https://fm4shoutcast.sf.apa.at'
mixer.init()
i = 0


# persist svm
if len(glob.glob("mfcc_svm.joblib")) < 1:
    mfcc_svm = MFCC.train_mfcc_svm(0, "data/speech/little", "data/music")
    dump(mfcc_svm, 'mfcc_svm.joblib')
else:
    mfcc_svm = load('mfcc_svm.joblib')

while i < 10:
    currentFile = "stream_" + str(i)
    radiorec.my_record(url, 3.0, currentFile)
    path = "data/test/" + currentFile + ".mp3"

    # for mfcc classification
    wav_path = ac.mp3_to_wav(path)
    current_mfcc = MFCC.get_mfcc_average(MFCC.read_mfcc(wav_path))
    result_mfcc = mfcc_svm.predict(current_mfcc)
    print(result_mfcc)

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

