import parselmouth
import pyworld as pw
import matplotlib.pyplot as plt

def load_audio(file_path):
    sound = parselmouth.Sound(file_path)
    return sound

def extract_f0_harvest(sound):
    x = sound.values.T.flatten()
    fs = sound.sampling_frequency
    f0, t = pw.harvest(x, fs)
    return f0, t

def plot_f0(time, f0, title='F0 contour'):
    plt.plot(time, f0, label='F0 (Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    audio_path = 'your_audio_file.wav'
    sound = load_audio(audio_path)
    f0, t = extract_f0_harvest(sound)
    plot_f0(t, f0)

if __name__ == "__main__":
    main()