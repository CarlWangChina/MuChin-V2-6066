import Code_for_Experiment.Metrics.music_understanding_model.tensorboardX.tests.test_numpy
import librosa
import torch
import laion_clap
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
import torchaudio
import numpy as np

def compute_cosine_similarity(audio_embed, text_embed):
    audio_embed_norm = F.normalize(audio_embed, p=2, dim=1)
    text_embed_norm = F.normalize(text_embed, p=2, dim=1)
    similarity = torch.mm(audio_embed_norm, text_embed_norm.transpose(0, 1))
    return similarity

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def clap_test(audio_file, text_data, use_default=True):
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
    text_embed = model.get_text_embedding(text_data, use_tensor=True)
    similarity = compute_cosine_similarity(audio_embed, text_embed)
    print(similarity)

def mert_test(generated, original, use_default=True, data_length=80000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    res = []
    for i in range(len(generated)):
        generated_audio, _ = torchaudio.load(generated[i])
        original_audio, _ = torchaudio.load(original[i])
        generated_audio = generated_audio[:1,:data_length].to(device)
        original_audio = original_audio[:1,:data_length].to(device)
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
        with torch.no_grad():
            output_generated = model(generated_audio, output_hidden_states=True)
            output_original = model(original_audio, output_hidden_states=True)
        g_all_layer_hidden_states = torch.stack(output_generated.hidden_states).squeeze().mean(-2)
        o_ll_layer_hidden_states = torch.stack(output_original.hidden_states).squeeze().mean(-2)
        similarity = compute_cosine_similarity(g_all_layer_hidden_states, o_ll_layer_hidden_states)
        average_similarity = torch.mean(similarity)
        res.append(average_similarity)
    return res

if __name__ == '__main__':
    audio_file_list = ['/content/drive/MyDrive/Research/Audio_Evaluator/test_audio/s_0001/s_0001.mp3']
    gt_file_list = ['/content/drive/MyDrive/Research/Audio_Evaluator/test_audio/s_0001/消愁.mp3']
    text_list = ["流行歌曲, 抒情, 百年孤独, 寂寞"]
    res = mert_test(gt_file_list, audio_file_list)