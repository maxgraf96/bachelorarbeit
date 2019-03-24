import soundfile as sf

def get_wav_duration(file):
    f = sf.SoundFile(file)
    # Duration = number of samples divided by the sample rate
    return len(f) / f.samplerate
