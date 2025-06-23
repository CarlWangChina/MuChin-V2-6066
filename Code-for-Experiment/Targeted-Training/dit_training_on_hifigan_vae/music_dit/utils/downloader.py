import os
import hashlib
import urllib.request
from tqdm.auto import tqdm
from Code_for_Experiment.Targeted_Training.token_transformer_module.ama-prof-divi.utils.logging import get_logger
import Code_for_Experiment.Targeted_Training.dit_training_on_hifigan_vae.music_dit.utils.dist_wrapper as dist

logger = get_logger(__name__)

def _get_sha246_checksum(file_path: str) -> str:
    with open(file_path, "rb") as fp:
        return hashlib.sha256(fp.read()).hexdigest()

def download_file(url: str, download_target: str, expected_sha256: str or None = None) -> str:
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        sha256_checksum = _get_sha246_checksum(download_target)
        if expected_sha256 is not None:
            if sha256_checksum == expected_sha256:
                return download_target
            else:
                if dist.is_primary():
                    logger.warning(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            if dist.is_primary():
                logger.info(f"{download_target} exists. SHA256 checksum ({sha256_checksum}) is not checked.")
                return download_target
    dist.barrier()
    if dist.is_primary():
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    loop.update(len(buffer))
    dist.barrier()
    sha256_checksum = _get_sha246_checksum(download_target)
    if expected_sha256:
        if sha256_checksum != expected_sha256:
            raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not match.")
    else:
        if dist.is_primary():
            logger.info(f"Downloaded {download_target}. SHA256 checksum is '{sha256_checksum}'.")
    return download_target