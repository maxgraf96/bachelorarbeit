from pydub import AudioSegment

'''
Takes a path to an mp3 file and converts it to a wav file with the same name.
'''
def mp3_to_wav(src):
    sound = AudioSegment.from_mp3(src)
    converted = src[:-3] + "wav"
    sound.export(converted, format="wav", parameters=["-ar", "16000"])
    return converted