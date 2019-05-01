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

from numba import jit, njit
from pydub import AudioSegment

def mp3_to_16_khz_wav(src):
    """
    Takes a path to an mp3 file and converts it to a wav file with the same name and sampling rate of 16000 kHz.
    :param src: The source file
    :return: The converted wav file path
    """

    sound = AudioSegment.from_mp3(src)
    sound = sound.set_channels(1).set_frame_rate(16000)
    converted = src[:-3] + "_16_kHz.wav"
    sound.export(converted, format="wav")
    return converted

def wav_to_16_khz(src):
    """
    Takes a path to a wav file and converts it to a wav file with the same name and sampling rate of 16000 kHz.
    :param src: The source file
    :return: The converted wav file path
    """
    sound = AudioSegment.from_wav(src)
    sound = sound.set_channels(1).set_frame_rate(16000)
    converted = src[:-4] + "_16_kHz.wav"
    sound.export(converted, format="wav")
    return converted

def wav_to_11_khz(src):
    """
    Takes a path to a wav file and converts it to a 11kHz MONO wav file. The new filename is filename_11_kHz.wav
    :param src: The source file
    :return: The converted wav file
    """

    sound = AudioSegment.from_wav(src)
    sound = sound.set_channels(1).set_frame_rate(11025)
    converted = src[:-4] + "_11_kHz.wav"
    sound.export(converted, format="wav")
    return converted

def mp3_to_11_khz_wav(src):
    """
    Takes a path to an mp3 file and converts it to a 11kHz MONO wav file. The new filename is filename_11_kHz.wav
    :param src: The source file
    :return: The converted wav file
    """

    sound = AudioSegment.from_mp3(src)
    sound = sound.set_channels(1).set_frame_rate(11025)
    converted = src[:-4] + "_11_kHz.wav"
    sound.export(converted, format="wav")
    return converted

def mp3_to_22_khz_wav(src):
    """
    Takes a path to an mp3 file and converts it to a 22kHz STEREO wav file. The new filename is filename_22_kHz.wav
    :param src: The source file
    :return: The converted wav file
    """

    sound = AudioSegment.from_mp3(src)
    sound = sound.set_frame_rate(22100)
    converted = src[:-4] + "_22_kHz.wav"
    sound.export(converted, format="wav")
    return converted

def wav_to_22_khz(src):
    """
       Takes a path to a wav file and converts it to a wav file with the same name and sampling rate of 22050 kHz.
       :param src: The source file
       :return: The converted wav file path
       """
    sound = AudioSegment.from_wav(src)
    sound = sound.set_frame_rate(22050)
    converted = src[:-4] + "_22_kHz.wav"
    sound.export(converted, format="wav")
    return converted
