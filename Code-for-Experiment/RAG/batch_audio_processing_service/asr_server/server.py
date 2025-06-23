import asr_dec_server
import clap_server
from ray import serve

asr_app = asr_dec_server.ASRDec.options(route_prefix="/ASRDec").bind()
clap_app = clap_server.CLAPProcessor.options(route_prefix="/CLAP").bind()