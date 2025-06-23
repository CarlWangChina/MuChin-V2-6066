import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sunoapi.api import *

if __name__ == "__main__":
    music_result = generate_music("越人语天姥，云霞明灭或可睹", "instrument, piano, harp, male vocal", mv="chirp-v3-0")
    print("生成音乐结果:", music_result)
    download_result(music_result, config.PROJECT_ROOT+"/tests/outputs/test_gen/")