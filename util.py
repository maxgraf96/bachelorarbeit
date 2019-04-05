import os
import random
from pathlib import Path

import soundfile as sf

def get_wav_duration(file):
    f = sf.SoundFile(file)
    # Duration = number of samples divided by the sample rate
    return len(f) / f.samplerate

def stereo2mono(x):
    '''
        This function converts the input signal
        (stored in a numpy array) to MONO (if it is STEREO)
    '''
    if isinstance(x, int):
        return -1
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x.flatten()
        else:
            if x.shape[1] == 2:
                return ((x[:, 1] / 2) + (x[:, 0] / 2))
            else:
                return -1

def get_random_file(ext, top=os.getcwd()):
    file_list = list(Path(top).glob(f"**/*.{ext}"))
    if not len(file_list):
        return f"No files matched that extension: {ext}"
    rand = random.randint(0, len(file_list) - 1)
    return str(file_list[rand])
