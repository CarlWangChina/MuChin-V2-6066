import unittest
from tqdm.auto import tqdm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
from music_dit2.trainers.dataset import DiTDataset
from torch.utils.data import DataLoader, RandomSampler

logger = get_logger(__name__)

class TestDiTDataset(unittest.TestCase):
    def setUp(self):
        self.training_ds, self.validation_ds = DiTDataset.make_datasets()

    def tearDown(self):
        self.training_ds.close()
        self.validation_ds.close()

    def test_load_data(self):
        logger.info("test_load_data.")
        logger.info("Training dataset size: %d", len(self.training_ds))
        logger.info("Validation dataset size: %d", len(self.validation_ds))
        sampler = RandomSampler(self.validation_ds)
        dataloader = DataLoader(self.validation_ds,
                                sampler=sampler,
                                batch_size=2,
                                collate_fn=DiTDataset.collate_fn)
        logger.info("Length of the dataloader: %d", len(dataloader))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            self.assertTrue("song_ids" in batch)
            self.assertTrue("vae" in batch)
            self.assertTrue("clap" in batch)
            self.assertEqual(batch["vae"]["data"].shape[0:2], (2, 512))
            self.assertEqual(len(batch["vae"]["padding_mask"]), 2)
            self.assertEqual(batch["clap"]["data"].shape[0:2], (2, 512))
            self.assertEqual(len(batch["clap"]["padding_mask"]), 2)
            if "lyrics" in batch:
                self.assertEqual(batch["lyrics"]["data"].shape[0:2], (2, 1))
                self.assertEqual(len(batch["lyrics"]["padding_mask"]), 2)