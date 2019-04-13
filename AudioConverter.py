from pydub import AudioSegment

def mp3_to_16_khz_wav(src):
    """
    Takes a path to an mp3 file and converts it to a wav file with the same name and sampling rate of 16000 kHz.
    :param src: The source file
    :return: The converted wav file path
    """

    sound = AudioSegment.from_mp3(src)
    sound = sound.set_channels(1).set_frame_rate(16000)
    converted = src[:-3] + "wav"
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

