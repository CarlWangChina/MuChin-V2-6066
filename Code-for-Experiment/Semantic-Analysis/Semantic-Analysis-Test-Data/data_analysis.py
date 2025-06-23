from FlagEmbedding import FlagModel
import os
from tqdm import tqdm
import pandas as pd
import json
from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import KMeans
import pickle
from config import gt
import numpy as np

class MuChindata_Analyzer:
    def __init__(self, filename, model_dir=None):
        if model_dir:
            self.sentence_model = FlagModel(model_dir, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=True)
        self.data = filename

    def data_warping(self, directory, group="专业组", return_labels=True):
        def merge(filename, return_labels=False):
            res = {"歌名": filename.split('/')[-1].replace('.txt', '')}
            index = 0
            questions = ["" for _ in range(20)]
            answers = ["" for _ in range(20)]
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                passage_line = ''
                if return_labels:
                    for line in lines:
                        if line.startswith('Q') or line.startswith('\n') or line.startswith('@'):
                            continue
                        else:
                            line = line.replace('\t', '').replace('\n', '')
                            if line.startswith('-q'):
                                questions[index] = line[5:]
                                index += 1
                            elif line.startswith('la') or line.startswith('oa'):
                                answers[index - 1] = line[5:]
                            passage_line += line
                    for i in range(index):
                        res[questions[i]] = answers[i].split(',')
                else:
                    for line in lines:
                        if line.startswith('Q') or line.startswith('\n') or line.startswith('@'):
                            continue
                        else:
                            line = line.replace('\t', '').replace('\n', '')
                            if line.startswith('TA'):
                                questions[index] = '这首歌带给你的感受'
                                answers[index] = line[4:]
                                index += 1
                            elif line.startswith('-q'):
                                questions[index] = line[5:]
                                index += 1
                            elif line.startswith('la') or line.startswith('oa'):
                                answers[index - 1] = line[5:]
                    for i in range(index):
                        res[questions[i]] = answers[i]
            return res

        files = [directory + '/' + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        data = []
        for i in tqdm(range(len(files)), desc=f"Processing messages from {group}"):
            data.append(merge(files[i], return_labels=return_labels))
        df = pd.DataFrame(data)
        excel_path = f'./data/{group}_labels_{return_labels}.xlsx'
        df.to_excel(excel_path, index=False)

    def solve_labels(self, cmp_kinds, self_kinds_amateur):
        for cmp_kind in cmp_kinds:
            with open(f'data/divided_data_professional/{cmp_kind}.json') as f:
                cluster = json.load(f)
            for i in range(len(self_kinds_amateur)):
                self.compare_discrepancy(cluster, cmp_kind, self_kinds_amateur[i], True)

    def compare_discrepancy(self, clusters, cmp_type, self_type, has_labels=False):
        print(f"Beginning to compare cognitive differences in {self_type} labelers when {cmp_type} is the same")
        res = {}
        data = pd.read_excel(self.data, sheet_name=None)
        data_b = data['Sheet1']
        data_a = data['Sheet2']
        data_a.set_index('歌名', inplace=True)
        data_b.set_index('歌名', inplace=True)
        for center_name, points in clusters.items():
            print(f"Counting {center_name}...")
            tmp = []
            for i in tqdm(range(len(points))):
                for song_name in data_a.index:
                    try:
                        words = data_a.loc[song_name, cmp_type]
                        point = points[i]
                        if point in words and song_name in data_b.index:
                            amateur_feel = [data_b.loc[song_name, self_type]]
                            profess_feel = [data_a.loc[song_name, self_type]]
                            embeddings_1 = self.sentence_model.encode(amateur_feel)
                            embeddings_2 = self.sentence_model.encode(profess_feel)
                            similarity = embeddings_1 @ embeddings_2.T
                            tmp.append(str(similarity[0][0]))
                    except:
                        continue
            res[center_name] = tmp
        with open(f'exp/edu/{cmp_type}_cmp_{self_type}.json', "w", encoding="utf-8") as ff:
            json.dump(res, ff, ensure_ascii=False, indent=4)

    def compmodel_eval(self):
        def get_data_from_txt(filename):
            with open(filename, "r") as file:
                content = file.read()
            dict_strs = content.strip().split('\n')
            data = []
            for dict_str in dict_strs:
                try:
                    parsed_dict = eval(dict_str)
                    data.append(parsed_dict)
                except Exception as e:
                    print(f"Error parsing dictionary: {e}")
            return data

        def compare_labels(output, ori):
            similarity_list = []
            for out, ori_ in zip(output, ori):
                similarity = self.cal_word_similarity(out, ori_)
                similarity_list.append(similarity)
            return similarity_list

        for file in os.listdir('data/cpmodel_eval'):
            if 'output' in file:
                fn = 'data/cpmodel_eval' + '/' + file
                fn_split = file.split('.')[-2]
                if os.path.exists(f'exp/cpeval_bf/{fn_split}.npy'):
                    continue
                print(f'Processing {fn}....')
                data = get_data_from_txt(fn)
                output_labels = [d['output_label'] for d in data]
                ori_labels = [d['ori_label'] for d in data]
                similarity_results = np.array(compare_labels(output_labels, ori_labels))
                np.save(f'exp/cpeval_bf/{fn_split}.npy', similarity_results)

    def cal_word_similarity(self, output, ori):
        res = []
        for i in tqdm(range(len(output))):
            tmp = []
            for out_d in output[i]:
                for ori_d in ori[i]:
                    embeddings_1 = self.sentence_model.encode([out_d])
                    embeddings_2 = self.sentence_model.encode([ori_d])
                    similarity = embeddings_1 @ embeddings_2.T
                    tmp.append(similarity[0][0])
            tmp_array = np.array(tmp)
            mean = np.mean(tmp_array)
            std_dev = np.std(tmp_array)
            res.append([mean, std_dev])
        return res

if __name__ == '__main__':
    ama_prof_divi = MuChindata_Analyzer('data/labels-free.xlsx')
    ama_prof_divi.compmodel_eval()