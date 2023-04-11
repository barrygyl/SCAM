# 人工智能
# 项目：NLP进阶
# 开发人：高云龙
# 开发时间：2023-02-27  18:47
# 开发工具：PyCharm
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import time
from datetime import timedelta
from models.Radical_config import Find_Radical
import pickle
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

UNK, PAD, CLS = '<UNK>', '<PAD>', '[CLS]'  # 未知字，padding符号,Bert首字符
UNK_ra, PAD_ra = '虌', '卥'  # 部首向量的未知字，padding符号


def build_dataset(config):
    # 加载保存的字典
    with open(config.index_dict, 'rb') as f:
        index_dict = pickle.load(f)
    with open(config.vocab_path_1, 'rb') as f:
        vocab_same_radical = pickle.load(f)
    vocab_same_radical = {word: idx for idx, word in enumerate(vocab_same_radical)}

    def load_dataset(path, pad_size=config.pad_size):
        contents = []
        # f = pd.read_csv(path)  # CNT数据集
        # f = pd.read_csv(path, delimiter='\t')  # TNT数据集
        # a = 0
        # for lin in trange(len(f)):
        #     '''for pre'''
        #     # if a >= 1000:
        #     #     break
        #     # else:
        #     #     a += 1
        #     content = f.iloc[lin].content
        #     label = int(f.iloc[lin].labels)
        #     # words_line = []
        #     words_r_line = []
        #     words_s_line = []
        #     token = config.tokenizer.tokenize(content)
        #     # seq_len_r = len(token)
        #     token = [CLS] + token
        #     seq_len = len(token)
        #     mask = []
        #     token_ids = config.tokenizer.convert_tokens_to_ids(token)

        #     if pad_size:
        #         if len(token) < pad_size:
        #             mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
        #             token_ids += ([0] * (pad_size - len(token)))
        #             token.extend([PAD] * (pad_size - len(token)))
        #         else:
        #             mask = [1] * pad_size
        #             token_ids = token_ids[:pad_size]
        #             seq_len = pad_size
        #             token = token[:pad_size]
        #     words_r_line = Find_Radical(token, 1, config.radical_dict, pad_size)
        #     for word in words_r_line:
        #         rid = vocab_same_radical.get(word, vocab_same_radical.get(UNK_ra))
        #         words_s_line.append([index_dict.get(i, 100) for i in range(rid-1, rid-config.same_size-1, -1)])
        #     contents.append((token_ids, words_s_line, int(label), seq_len, mask))
        # return contents
        """
        使用TXT的数据集———THU数据集
        """
        with open(path, 'r', encoding='UTF-8') as f:
            a = 0
            for line in tqdm(f):
                # if a >= 1000:
                #     break
                # else:
                #     a += 1
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                # words_line = []
                words_r_line = []
                words_s_line = []
                token = config.tokenizer.tokenize(content)
                # seq_len_r = len(token)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                        token = token[:pad_size]
                words_r_line = Find_Radical(token, config.radical_dict, pad_size)
                for word in words_r_line:
                     rid = vocab_same_radical.get(word, vocab_same_radical.get(UNK_ra))
                     words_s_line.append([index_dict.get(i, 100) for i in range(rid-1, rid-config.same_size-1, -1)])
                contents.append((token_ids, words_s_line, int(label), seq_len, mask))
        return contents

    test = load_dataset(config.test_path, config.pad_size)
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        z = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        # token_type_ids = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        # return (x, seq_len, mask, token_type_ids), y, z
        return (x, seq_len, mask), y, z

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


