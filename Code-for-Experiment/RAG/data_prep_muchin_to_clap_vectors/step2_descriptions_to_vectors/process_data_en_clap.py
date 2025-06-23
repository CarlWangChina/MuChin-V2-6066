import json
import os
from glob import glob
from typing import List, Dict
import torch
from transformers import ClapModel, AutoTokenizer

# 检查设备（CUDA、MPS 或 CPU）
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 加载 CLAP 模型和 Tokenizer
model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

def generate_clap_vector(texts: List[str], batch_size: int = 2) -> List[torch.Tensor]:
    """
    使用 CLAP 模型生成文本的定长向量。

    参数:
        texts (List[str]): 输入文本列表。
        batch_size (int): 批处理大小。

    返回:
        List[torch.Tensor]: 生成的定长向量列表。
    """
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入移动到正确的设备
        with torch.no_grad():
            audio_embed = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        vectors.extend(audio_embed.cpu().detach())
    return vectors

def split_text(text_list: List[str], length: str) -> str:
    """
    根据文本长度将文本分为不同类别，并用 '，' 连接它们。

    参数:
        text_list (List[str]): 待分类的文本列表。
        length (str): 长度类别（'short'、'medium'、'long'）。

    返回:
        str: 根据长度类别连接的文本。
    """
    text_list = [t for t in text_list if t is not None]
    short_texts = [t for t in text_list if len(t) < 40]
    medium_texts = [t for t in text_list if 40 <= len(t) < 120]
    long_texts = [t for t in text_list if len(t) >= 120]
    
    if length == 'short':
        return '，'.join(short_texts)
    elif length == 'medium':
        return '，'.join(medium_texts)
    elif length == 'long':
        return '，'.join(long_texts)
    return '，'.join(text_list)

def extract_labels(text_list: List[str]) -> List[str]:
    """
    从文本列表中提取标签（长度小于120字符）。

    参数:
        text_list (List[str]): 待提取标签的文本列表。

    返回:
        List[str]: 提取出的标签列表。
    """
    text_list = [t for t in text_list if t is not None]
    return [t for t in text_list if len(t) < 120]

def process_json(file_path: str) -> Dict[str, List[float]]:
    """
    处理单个JSON文件，提取分类后的文本，并生成CLAP向量。

    参数:
        file_path (str): JSON文件路径。

    返回:
        Dict[str, List[float]]: 包含10个CLAP向量的字典。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"文件 {file_path} 不是有效的JSON格式。")
            return {}
    
    amat = data.get('amat', [])
    prof = data.get('prof', [])

    amat = amat if isinstance(amat, list) else []
    prof = prof if isinstance(prof, list) else []

    amat_labels = extract_labels(amat)
    prof_labels = extract_labels(prof)

    amat_texts = {
        'short': split_text(amat_labels, 'short'),
        'medium': split_text(amat_labels, 'medium'),
        'long': split_text(amat, 'long'),
        'all_labels': '，'.join(amat_labels),
        'all_sentences': ' '.join(amat)
    }

    prof_texts = {
        'short': split_text(prof_labels, 'short'),
        'medium': split_text(prof_labels, 'medium'),
        'long': split_text(prof, 'long'),
        'all_labels': '，'.join(prof_labels),
        'all_sentences': ' '.join(prof)
    }

    # 将 10 种文本表示通过 CLAP 模型生成向量
    text_representations = [
        amat_texts['short'], amat_texts['medium'], amat_texts['long'], amat_texts['all_labels'], amat_texts['all_sentences'],
        prof_texts['short'], prof_texts['medium'], prof_texts['long'], prof_texts['all_labels'], prof_texts['all_sentences']
    ]

    clap_vectors = generate_clap_vector(text_representations)
    clap_dict = {f'clap{i}': vector.tolist() for i, vector in enumerate(clap_vectors)}

    return clap_dict

def main(input_dir: str, output_file: str) -> None:
    """
    处理输入目录中的所有JSON文件，并将结果写入输出文件。

    参数:
        input_dir (str): 包含JSON文件的目录。
        output_file (str): 输出JSON文件的路径。
    """
    json_files = glob(os.path.join(input_dir, '**/*.json'), recursive=True)
    all_results = {}
    
    for json_file in json_files:
        try:
            file_id = os.path.splitext(os.path.basename(json_file))[0]
            all_results[file_id] = process_json(json_file)
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(all_results, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_dir = '/Users/ricky/Desktop/processed_datasets/muchin_en'
    output_file = '/Users/ricky/Desktop/processed_datasets/muchin_en/processed_songs_en.json'
    main(input_dir, output_file)