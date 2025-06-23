import librosa

def sampleAudio(audio, window_size: int, n: int):
    assert len(audio) >= window_size, "Audio length should be greater than or equal to window size."
    if len(audio) < n * window_size:
        step = len(audio) // n
        start_indices = [i * step for i in range(n)]
    else:
        step = (len(audio) - window_size) // (n - 1)
        start_indices = [i * step for i in range(n)]
    slices = []
    slice_index = []
    for idx in start_indices:
        end_idx = idx + window_size
        slice_index.append((idx, end_idx))
        slices.append(audio[idx:end_idx])
    return slices, slice_index

def sampleAudioBySecond(audio, sample_rate:int, window_size: int=10, n: int=10):
    return sampleAudio(audio=audio, window_size=window_size*sample_rate, n=n)

if __name__=="__main__":
    audio_data, sample_rate = librosa.load('/NAS/datasets-mp3/ali/16/1691673_src.mp3', sr=48000)
    slices, indices = sampleAudioBySecond(audio_data, sample_rate=sample_rate)
    print(slices, indices)