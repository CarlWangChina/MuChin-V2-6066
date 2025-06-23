import json
import os
from glob import glob
from typing import List, Dict

def split_text(text_list: List[str], length: str) -> str:
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
    text_list = [t for t in text_list if t is not None]
    return [t for t in text_list if len(t) < 120]

def process_json(file_path: str) -> Dict[str, Dict[str, str]]:
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