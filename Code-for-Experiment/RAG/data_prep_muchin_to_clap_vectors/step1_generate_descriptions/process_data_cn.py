import json
import os
from glob import glob
from typing import List, Dict

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
    short_texts = [t for t in text_list if len(t) < 20]
    medium_texts = [t for t in text_list if 20 <= len(t) < 50]
    long_texts = [t for t in text_list if len(t) >= 50]

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
    return [t for t in text_list if len(t) < 50]

def process_json(file_path: str) -> Dict[str, Dict[str, str]]:
    """
    处理单个JSON文件，提取分类后的文本，并保留原始id。

    参数:
        file_path (str): JSON文件路径。

    返回:
        Dict[str, Dict[str, str]]: 包含'amat'和'prof'的分类文本字典，以及原始id。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"文件 {file_path} 不是有效的JSON格式。")
            return {}

    result = {
        'id': data.get('id', ''),
        'amat': {},
        'prof': {}
    }

    amat = data.get('amat', [])
    prof = data.get('prof', [])

    amat = amat if isinstance(amat, list) else []
    prof = prof if isinstance(prof, list) else []

    amat_labels = extract_labels(amat)
    prof_labels = extract_labels(prof)

    result['amat'] = {
        'short': split_text(amat_labels, 'short'),
        'medium': split_text(amat_labels, 'medium'),
        'long': split_text(amat, 'long'),
        'all_labels': '，'.join(amat_labels),
        'all_sentences': ' '.join(amat)
    }

    result['prof'] = {
        'short': split_text(prof_labels, 'short'),
        'medium': split_text(prof_labels, 'medium'),
        'long': split_text(prof, 'long'),
        'all_labels': '，'.join(prof_labels),
        'all_sentences': ' '.join(prof)
    }

    return result

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
    input_dir = '/Users/ricky/Desktop/processed_datasets/muchin'
    output_file = '/Users/ricky/Desktop/processed_datasets/muchin/processed_songs_cn.json'
    main(input_dir, output_file)