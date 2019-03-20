from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn import svm
import glob
import AudioConverter as ac

'''
Trains a SVM with the given label (0 = speech, 1 = music) and all the files within the given path
'''
def train_mfcc_svm(label, path_speech, path_music):

    # init classifier from sklearn
    clf = svm.SVC(gamma=0.001, C=100)

    # speech
    speech_files = glob.glob(path_speech + "/*.wav") # list of files in given path
    sp_trn = np.zeros((len(speech_files), 13))
    for i in range(len(speech_files)):
        print(str(i) + " of " + str(len(speech_files)) + " - processing " + str(speech_files[i]))

        # TODO verify if taking the average of all MFCCs over the time duration of the file is okay
        average = get_mfcc_average(read_mfcc(speech_files[i]))
        sp_trn[i] = average
    sp_lbls = np.zeros(len(speech_files))

    # music
    music_files = glob.glob(path_music + "/*.mp3")  # list of files in given path

    mu_trn = np.zeros((len(music_files), 13))
    for i in range(len(music_files)):
        print(str(i) + " of " + str(len(music_files)) + " - processing " + str(music_files[i]))
        music_files[i] = ac.mp3_to_wav(music_files[i])

        average = get_mfcc_average(read_mfcc(music_files[i]))
        mu_trn[i] = average
    mu_lbls = np.ones(len(music_files))

    # concat arrays
    trn = np.concatenate((sp_trn, mu_trn))
    lbls = np.concatenate((sp_lbls, mu_lbls))

    # fit classifier
    clf.fit(trn, lbls)

    return clf


def read_mfcc(wav_file_path):
    (rate, sig) = wav.read(wav_file_path)
    mfcc_feat = mfcc(sig, rate, nfft=2048)
    #d_mfcc_feat = delta(mfcc_feat, 2)
    #fbank_feat = logfbank(sig, rate)

    #print(fbank_feat[1:3, :])
    return mfcc_feat


def get_mfcc_average(mfcc):
    return np.average(mfcc, 0)