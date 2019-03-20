from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def read_mfcc(wav_file_path):
    (rate, sig) = wav.read(wav_file_path)
    mfcc_feat = mfcc(sig, rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig, rate)

    print(fbank_feat[1:3, :])
