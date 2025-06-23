import requests
import oss2
import os
import sys
import sunoapi.config_loader
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
config = sunoapi.config_loader.config
END_POINT = sunoapi.config_loader.END_POINT
ACCESS_KEY_ID = sunoapi.config_loader.ACCESS_KEY_ID
ACCESS_KEY_SECRET = sunoapi.config_loader.ACCESS_KEY_SECRET
BUCKET_NAME = sunoapi.config_loader.BUCKET_NAME

def download_file(oss_address: str, local_path: str) -> bool:
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    bucket_name = "source"
    bucket = oss2.Bucket(auth, END_POINT, bucket_name)
    oss_sub_address = oss_address.split("source/")[-1]
    logger.info(" [x]-%d Downloading file %s to %s", os.getpid(), oss_sub_address, local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        result = bucket.get_object_to_file(oss_sub_address, local_path)
        if result.status == 200:
            logger.info("Download file %s Success!", oss_sub_address)
            return True
    except oss2.exceptions.NoSuchKey as e:
        logger.error("Download file %s failed! %s", oss_sub_address, e)
        return False

def download_file_from_url(url, save_path):
    response = requests.get(url, stream=True, timeout=10)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        logger.info("文件已下载到：%s", save_path)
    else:
        logger.error("下载失败:HTTP 状态码 %s", response.status_code)

def upload_file(local_path: str, oss_address: str) -> bool:
    def percentage(consumed_bytes, total_bytes):
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, END_POINT, BUCKET_NAME)
    try:
        result = bucket.put_object_from_file(
            oss_address, local_path, progress_callback=percentage
        )
        if result.status == 200:
            logger.info("Upload file %s Success!", oss_address)
            return True
    except oss2.exceptions.NoSuchKey as e:
        logger.error("Upload file %s failed! %s", oss_address, e)
        return False