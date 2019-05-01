# Copyright (c) 2019 Max Graf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

# The path to the root directory of the training data
data_path = "/media/max/Elements/bachelorarbeit/data/"

def get_wav_duration(file):
    """
    Gets the duration of a wav file in seconds
    :param file: The file
    :return: The duration
    """
    f = sf.SoundFile(file)
    # Duration = number of samples divided by the sample rate
    return f.frames / f.samplerate

@jit(cache=True)
def stereo2mono(x):
    """
    Converts the input signal (stored in a numpy array) to MONO (if it is STEREO)
    """
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

def plot_speech_music_map(filename, x, speech_music_map, save_csv=False):
    """
    Plots and save a speech_music_map (sequence of zeros (speech) and ones (music))
    :param filename: The filename under which the map should be saved
    :param x: The duration in steps of 0.5s
    :param speech_music_map: The speech_music_map (sequence of zeros (speech) and ones (music))
    :param save_csv: If true, a CSV file will be saved additionally to the PNG file.
    """
    if isinstance(speech_music_map, list):
        speech_music_map = np.array(speech_music_map)
    speech = np.argwhere(speech_music_map == 0)
    music = np.argwhere(speech_music_map == 1)

    # Set plot dimensions
    plt.rcParams["figure.figsize"] = [16, 4]

    for idx in speech:
        plt.axvline(x=idx, color='red', linestyle='-', linewidth=0.1)
    for idx in music:
        plt.axvline(x=idx, color='green', linestyle='-', linewidth=0.1)

    plt.figure(1)
    plt.subplot(111)

    plt.title("Speech/music distribution for " + filename)
    plt.xlabel("Seconds / class")
    plt.ylabel("")
    plt.yticks(ticks=[0, 1], labels=["", ""])
    plt.grid()

    # plt.show()
    plt.savefig("plots/speech_music_maps/" + filename + ".png")
    # Clear plot
    plt.clf()

    if save_csv:
        print("Saving CSV '" + filename + ".csv'")
        save_speech_music_map_csv([x, speech_music_map], "plots/speech_music_maps/", filename)

def save_speech_music_map_csv(columns, path, filename):
    """
    Saves the a speech music map as CSV file
    :param columns: The columns
    :param path: Where the file should be saved
    :param filename: How the file should be named
    :return:
    """
    with open(path + filename + '.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(columns[0])):
            row = []
            for column in columns:
                row.append(column[i])
            writer.writerow(row)

def load_speech_music_map_csv(path):
    """
    Loads a speech_music_map from a CSV file
    :param path: The path to the CSV file
    :return: A numpy array containing the values
    """
    df = pd.read_csv(path, sep=',', header=None)
    return df.values

def load_and_convert_muspeak_mirex_csv(path):
    """
    Loads a speech_music_map from a CSV file in the muspeak_mirex format
    :param path: The path to the CSV file
    :return: An array containing the speech_music_map converted to the default format
    """
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