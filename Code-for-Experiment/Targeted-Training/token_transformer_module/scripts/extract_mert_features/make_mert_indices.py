import torch
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format="%(asctime)s:%(name)s:[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
MERT_FILE_DIR = "/data/mert-v1-330m-75hz"
MERT_FILE_PATTERN = "**/*_mert.pt"
OUTPUT_FILE = "/data/1/mert-v1-330m-75hz/mert-index.pt"

if __name__ == "__main__":
    assert Path(MERT_FILE_DIR).exists(), f"Directory {MERT_FILE_DIR} does not exist."
    mert_files = list(Path(MERT_FILE_DIR).glob(MERT_FILE_PATTERN))
    mert_indices = {
        "model": None,
        "feature_dim": None,
        "feature_rate": None,
        "mert_layer": None,
        "normalized_loudness": None,
        "mert_files": []
    }

    for mert_file in tqdm(mert_files):
        mert = torch.load(mert_file, map_location="cpu")
        model_name = mert["model"]

        if mert_indices["model"] is not None:
            assert mert_indices["model"] == model_name, \
                f"Model name mismatch: {mert_indices['model']} vs {model_name}"
        else:
            mert_indices["model"] = model_name
        feature_dim = mert["feature_dim"]

        if mert_indices["feature_dim"] is not None:
            assert mert_indices["feature_dim"] == feature_dim, \
                f"Feature dimension mismatch: {mert_indices['feature_dim']} vs {feature_dim}"
        else:
            mert_indices["feature_dim"] = feature_dim
        feature_rate = mert["feature_rate"]

        if mert_indices["feature_rate"] is not None:
            assert mert_indices["feature_rate"] == feature_rate, \
                f"Feature rate mismatch: {mert_indices['feature_rate']} vs {feature_rate}"
        else:
            mert_indices["feature_rate"] = feature_rate
        mert_layer = mert["mert_layer"]

        if mert_indices["mert_layer"] is not None:
            assert mert_indices["mert_layer"] == mert_layer, \
                f"MERT layer mismatch: {mert_indices['mert_layer']} vs {mert_layer}"
        else:
            mert_indices["mert_layer"] = mert_layer
        normalized_loudness = mert["normalized_loudness"]

        if mert_indices["normalized_loudness"] is not None:
            assert mert_indices["normalized_loudness"] == normalized_loudness, \
                f"Normalized loudness mismatch: {mert_indices['normalized_loudness']} vs {normalized_loudness}"
        else:
            mert_indices["normalized_loudness"] = normalized_loudness
        song_id = mert["song_id"]
        file_name = mert_file.name
        file_group = int(mert_file.parent.name)
        data_length = mert["data"].shape[0]
        audio_length = mert["audio_length"]
        mert_indices["mert_files"].append({
            "song_id": song_id,
            "file_name": file_name,
            "file_group": file_group,
            "data_length": data_length,
            "audio_length": audio_length
        })
    mert_indices["mert_files"] = sorted(mert_indices["mert_files"], key=lambda x: (x["file_group"], x["song_id"]))
    torch.save(mert_indices, OUTPUT_FILE)
    logger.info(f"Saved MERT indices to {OUTPUT_FILE}")