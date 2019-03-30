from pydub import AudioSegment

def mp3_to_wav(src):
    """
    Takes a path to an mp3 file and converts it to a wav file with the same name.
    :param src: The source file
    :return: The converted wav file
    """

    sound = AudioSegment.from_mp3(src)
    converted = src[:-3] + "wav"
    sound.export(converted, format="wav", parameters=["-ar", "16000"])
    return converted

def wav_to_11_khz(src):
    """
    Takes a path to a wav file and converts it to a 11kHz wav file. The new filename is filename_11_kHz.wav
    :param src: The source file
    :return: The converted wav file
    """

    sound = AudioSegment.from_wav(src)
    converted = src[:-4] + "_11_kHz.wav"
    sound.export(converted, format="wav", parameters=["-ar", "11025"])
    return converted

def mp3_to_11_khz_wav(src):
    """
    Takes a path to an mp3 file and converts it to a 11kHz wav file. The new filename is filename_11_kHz.wav
    :param src: The source file
    :return: The converted wav file
    """

    sound = AudioSegment.from_mp3(src)
    converted = src[:-4] + "_11_kHz.wav"
    sound.export(converted, format="wav", parameters=["-ar", "11025"])
    return converted

