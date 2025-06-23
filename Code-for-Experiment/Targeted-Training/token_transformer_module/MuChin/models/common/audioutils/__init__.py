from .audioutils import AudioUtil
import os
import sys
from . import get_audio_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio import AudioUtils, get_audio_utils