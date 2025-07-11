import numpy as np
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
import json
import csv
from tqdm import trange


def compute_cosine_similarity(audio_embed, text_embed):
    # 计算余弦相似度
    audio_embed_norm = F.normalize(audio_embed, p=2, dim=1)
    text_embed_norm = F.normalize(text_embed, p=2, dim=1)
    similarity = torch.mm(audio_embed_norm, text_embed_norm.transpose(0, 1))
    return similarity


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def clap_test(audio_file, text_data, use_default=True):
    '''
    使用CLAP模型计算音频文件与文本数据之间的余弦相似度。
    
    参数:
    audio_file -- 音频文件路径或文件列表
    text_data -- 文本数据，需为可嵌入的格式
    use_default -- 是否直接使用默认的CLAP模型（不进行微调）
    
    返回:
    无返回值，直接打印余弦相似度矩阵。
    修改为返回余弦相似度矩阵
    '''
    # model = laion_clap.CLAP_Module(enable_fusion=False)
    # model.load_ckpt("D:/wangqi/630k-audioset-best.pt") # 加载默认的预训练检查点。
    # model.load_ckpt()
    audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
    text_embed = model.get_text_embedding(text_data, use_tensor=True)

    similarity = compute_cosine_similarity(audio_embed, text_embed)
    # print(similarity)  # 打印余弦相似度矩阵
    return similarity



if __name__ == '__main__':
    data = None
    with open("/home/zhehui/stable-audio-tools/test_input.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # prompt = data[str(i)]["short"]
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt("/home/zhehui/wangqi/630k-audioset-best.pt")  # 加载默认的预训练检查点。
    res_clap = []
    for i in range(1, 2):
        path = '/home/zhehui/FAD/result_short/' + str(i).zfill(3) + '.wav'
        prompt1 = data[str(i)]["short"]
        prompt2 = data[str(i)]["medium"]
        repeat_num = 100
        for _ in range(repeat_num):
            res2 = clap_test([path], [prompt1, prompt2])
            res_clap.append((path, res2[0, 0].item()))
    with open("/home/zhehui/wangqi/variance.csv","w",encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(100):
            writer.writerow(res_clap[i])

