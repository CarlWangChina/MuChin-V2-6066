from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence(audio_path, silence_thresh=-50, min_silence_len=500):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
        processed_audio = AudioSegment.silent(duration=0)
        for chunk in chunks:
            processed_audio += chunk
        output_path = audio_path.replace('.mp3', '_no_silence.mp3')
        processed_audio.export(output_path, format='mp3')
        return output_path
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None