import os
import random
from pathlib import Path
import numpy as np
import time

import soundfile as sf
from numba import jit


def get_wav_duration(file):
    f = sf.SoundFile(file)
    # Duration = number of samples divided by the sample rate
    return len(f) / f.samplerate

@jit(cache=True)
def stereo2mono(x):
    """
        This function converts the input signal
        (stored in a numpy array) to MONO (if it is STEREO)
    """
    # if isinstance(x, int):
    #     return -1
    if x.ndim == 1:
        return x

    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x.flatten()
        else:
            if x.shape[1] == 2:
                return (x[:, 1] / 2) + (x[:, 0] / 2)
            else:
                return -1

def get_random_file(ext, top=os.getcwd()):
    file_list = list(Path(top).glob("**/*.{}".format(ext)))
    if not len(file_list):
        return "No files matched that extension: {}".format(ext)
    rand = random.randint(0, len(file_list) - 1)
    return str(file_list[rand])

def spec_to_khz_spec(spec, khz):
    return spec[spec < khz, :]
