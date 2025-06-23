import logging
logging.basicConfig(format="%(asctime)s:%(name)s:[%(levelname)s] %(message)s", level=logging.INFO)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)