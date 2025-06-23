import os
import datasets

class MuChinDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=48000),
                    "text": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": "train.txt"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": "test.txt"},
            ),
        ]

    def _generate_examples(self, data_dir):
        with open(data_dir, "r") as f:
            for line in f:
                line = line.strip()
                audio_name, audio_path, desc = line.split()[0], line.split()[1], " ".join(line.split()[2:])
                yield audio_name, {"audio": audio_path, "text": desc}