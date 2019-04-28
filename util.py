import csv
import os
import random
from pathlib import Path
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

import soundfile as sf
from numba import jit

ext_hdd_path = "/media/max/Elements/bachelorarbeit/"

def get_wav_duration(file):
    f = sf.SoundFile(file)
    # Duration = number of samples divided by the sample rate
    return f.frames / f.samplerate

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

def plot_speech_music_map(filename, x, speech_music_map):
    speech_music_map = np.array(speech_music_map)
    colors = np.where(speech_music_map > 0, "green", "blue")
    dot_size = 240 / len(speech_music_map)
    plt.scatter(x, speech_music_map, c=colors, s=dot_size)
    plt.xlabel("Seconds")
    plt.ylabel("Class")
    plt.title("Speech/music distribution for " + filename)
    plt.yticks(ticks=[0, 1], labels=["Speech", "Music"])
    plt.rcParams["figure.figsize"] = [16, 2]
    plt.grid()
    # plt.show()
    plt.savefig("plots/speech_music_maps/" + filename + ".png")

    print("Saving CSV '" + filename + ".csv'")
    save_speech_music_map_csv([x, speech_music_map], "plots/speech_music_maps/", filename)

def save_speech_music_map_csv(columns, path, filename):
    with open(path + filename + '.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(columns[0])):
            row = []
            for column in columns:
                row.append(column[i])
            writer.writerow(row)

def load_speech_music_map_csv(path):
    df = pd.read_csv(path, sep=',', header=None)
    return df.values

def load_and_convert_muspeak_mirex_csv(path):
    df = pd.read_csv(path, sep=',', header=None)
    values = df.values
    starts = np.array(values[:, 0], dtype=float)
    durations = np.array(values[:, 1], dtype=float)
    labels = np.where(values[:, 2] == 'm', 1, 0)

    total_length = np.ceil(starts[len(starts) - 1] + durations[len(durations) - 1])

    result = []
    starts_idx = 0
    for i in np.arange(0.0, total_length, 0.5):
        if i > starts[starts_idx] + durations[starts_idx] and starts_idx != len(starts) - 1:
            starts_idx += 1
        result.append([i, labels[starts_idx]])

    return np.array(result)