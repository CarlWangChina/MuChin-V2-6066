import unittest
from tqdm.auto import tqdm
from pathlib import Path
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from ama_prof_divi_hifigan.trainer import ChunkedWavDataset
from torch.utils.data import DataLoader, RandomSampler

logger = get_logger(__name__)
DATA_DIR = Path("/mnt/d/Temp/mmap-ds")
META_FILE = DATA_DIR / "wav_list_0.csv"
DATA_FILE = DATA_DIR / "wav_list_0.dat"

class TestChunkedWavDataset(unittest.TestCase):
    def setUp(self):
        self.training_ds, self.validation_ds = ChunkedWavDataset.make_datasets(
            meta_file=str(META_FILE),
            data_file=str(DATA_FILE)
        )

    def tearDown(self):
        self.training_ds.close()
        self.validation_ds.close()

    def test_get_chunks(self):
        datasets = {
            "training": self.training_ds,
            "validation": self.validation_ds
        }

        for name, ds in datasets.items():
            logger.info("%s dataset has %d samples.", name, len(ds))
            sampler = RandomSampler(ds)
            dataloader = DataLoader(ds,
                                    batch_size=3,
                                    sampler=sampler,
                                    collate_fn=ds.collate_fn)
            num_batches = 20
            with tqdm(total=num_batches, desc=f"Getting chunks for {name}") as pbar:
                for i, batch in enumerate(dataloader):
                    self.assertIsNotNone(batch)
                    self.assertEqual(batch["wav"]["data"].size(), (3, ds.chunk_size))
                    self.assertEqual(batch["wav"]["padding_mask"].size(), (3, ds.chunk_size))
                    self.assertEqual(batch["mel"]["data"].size(), (3, 80, ds.chunk_size // 512))
                    self.assertEqual(batch["mel"]["padding_mask"].size(), (3, ds.chunk_size // 512))
                    pbar.update(1)
                    if i >= num_batches:
                        break