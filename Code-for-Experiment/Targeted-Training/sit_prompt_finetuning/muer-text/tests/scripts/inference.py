import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pathlib import Path
from tqdm import tqdm
from Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.logging import get_logger
from Code_for_Experiment.Targeted_Training.text_to_phoneme_converter.ama-prof-divi_text.phoneme.phoneme import Lyrics2Phoneme

logger = get_logger(__name__)

class Lyrics2PhonemeProcessor():
    def __init__(self):
        self.special_tokens = {'(verse)', '(chorus)', '(intro)', '(ending)', '(prechorus)', '(bridge)', '(interlude)'}
        self.lyrics2phoneme = Lyrics2Phoneme(self.special_tokens)
        logger.info("Lyrics2Phoneme instance is created.")

    def load_data(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    data.append(line)
        return data

    def process(self, path):
        try:
            data = self.load_data(path)
            phoneme = self.lyrics2phoneme.translate(data)
            return phoneme
        except Exception as e:
            logger.error(f"Failed to process data from {path}. Error: {e}")
            return None

if __name__ == '__main__':
    processor = Lyrics2PhonemeProcessor()
    input_base_dir = '/nfs/datasets-mp3/zihao/funasr-txt/'
    output_base_dir = '/nfs/carl/phoneme/'
    for root, dirs, files in os.walk(input_base_dir):
        for file in tqdm(files):
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_base_dir)
                output_file_path = os.path.join(output_base_dir, relative_path)
                output_dir = os.path.dirname(output_file_path)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                processed_content = processor.process(input_file_path)
                if processed_content is not None:
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write('\n'.join(processed_content))