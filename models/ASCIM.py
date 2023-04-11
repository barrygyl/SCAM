# 人工智能
# 项目：Combining same-radical global information model
# 开发人：Barry
# 开发时间：2023-03-14  16:33
# 开发工具：PyCharm
# coding: UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from models.Radical_config import create_dict
import gensim
# from transformers import AutoTokenizer, AutoModelForPreTraining, AutoConfig
from pytorch_pretrained import BertModel, BertTokenizer
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'ASCIM'
        self.train_path = dataset + '/data/train.tsv'                               # 训练集
        self.dev_path = dataset + '/data/val.tsv'                                   # 验证集
        self.test_path = dataset + '/data/test.tsv'                                 # 测试集
        # self.train_path = dataset + '/data/train.csv'                               # 训练集
        # self.dev_path = dataset + '/data/dev.csv'                                   # 验证集
        # self.test_path = dataset + '/data/test.csv'                                 # 测试集
        # self.train_path = dataset + '/data/train.txt'                               # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                   # 验证集
        # self.test_path = dataset + '/data/test.txt'                                 # 测试集
        self.vocab_path_1 = './radical_vector/vocab_same_radical.pkl'               # 词表
        self.vocab_path_2 = './radical_vector/vocab_bert.pkl'                       # 词表
        self.index_dict = './radical_vector/sr_to_bert.pkl'                         # 指标字典
        self.radical_path = './radical_vector/char2radical.txt'                     # 部首的语料库位置
        self.radical_dict = create_dict(self.radical_path)                          # 构建部首查询字典
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]             # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'       # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.bert_path = './bert_pretrain'                                          # bert的选择
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)              # bert 分词器模型的选择
        self.batch_size = 64                                                        # mini-batch大小
        self.pad_size = 32                                                          # 每句话处理成的长度(短填长切)
        self.same_size = 28                                                         # 同部首字数
        self.dropout = 0.5                                                          # 随机失活
        self.require_improvement = 2000                                             # 若超过xx效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                                     # 类别数
        self.num_epochs = 30                                                        # epoch数
        self.learning_rate = 1e-5                                                   # 学习率 
        self.embed = 768                                                            # 部首向量维度
        self.vocab_size_2 = 21130                                                   # Bert词表
        self.l_hidden_size = 384                                                    # lstm隐藏层
        self.l_num_layers = 2                                                       # lstm层数
        self.hidden_size = 768
        self.hidden_size2 = 64                                                      # 全连接层


"""same-radical-embedding"""


class SameRadicalEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, vocab_size):
        super(SameRadicalEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape input tensor to (batch_size * dim1, dim2)
        x = x.view(-1, self.input_dim[-1])
        
        # Apply embedding to get (batch_size * dim1, dim2, embedding_dim)
        embedded_x = self.embedding(x)
        
        # Reshape to (batch_size, dim1, dim2, embedding_dim)
        embedded_x = embedded_x.view(batch_size, *self.input_dim, self.embedding_dim)
        
        return embedded_x


"""CBAM"""
'''channel'''


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


'''Spatial'''


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


'''last'''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x.permute(0, 3, 1, 2).contiguous()  # Change shape to (batch_size, dim3, dim1, dim2)
        out = self.channel_attention(out) * out
        out = out.permute(0, 2, 3, 1).contiguous()  # Change shape back to (batch_size, dim1, dim2, dim3)
        out = self.spatial_attention(out) * out
        return out


"""ASCIM"""


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dict = config.radical_dict
        self.SameRadicalEmbedding = SameRadicalEmbedding(input_dim=(config.pad_size, config.same_size),
                                                         embedding_dim=config.embed, vocab_size=config.vocab_size_2)
        self.device = config.device
        self.tokenizer = config.tokenizer
        self.radical_pre = config.radical_pre
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.l_hidden_size,
                            num_layers=config.l_num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=config.dropout
                            )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.cbam = CBAM(config.embed)  # CBAM层
        '''定义全连接层'''
        self.fc = nn.Linear(config.embed * config.pad_size * config.same_size, config.num_classes)
        self.fc1 = nn.Linear(config.hidden_size, config.num_classes)
        self.fc2 = nn.Linear(3*config.num_classes, config.num_classes)  # 原
        # self.fc2 = nn.Linear(2*config.num_classes, config.num_classes)  # 消融
        self.fc3 = nn.Linear(config.l_num_layers * config.l_hidden_size * (config.pad_size - 1), config.num_classes)
        self.fc4 = nn.Linear(config.l_hidden_size * 2, config.num_classes)
        self.bc = config.batch_size
        self.pad_size = config.pad_size
        '''attention层'''
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key_layer = nn.Linear(config.embed, config.embed, bias=False)
        self.value_layer = nn.Linear(config.embed, config.embed, bias=False)
        self.embed = config.embed

    def forward(self, x, x_r):
        context = x[0]  # sentence
        mask = x[2]     # mask

        """同部首字信息向量"""
        l = len(x_r)
        x_r = self.SameRadicalEmbedding(x_r)
        x_r = self.dropout(x_r)
        x_r = torch.tanh(x_r)
        '''CBAM'''
        out_a = self.cbam(x_r)
        '''full connected--ySC'''
        out_r = out_a.view(l, -1)  # 原
        # out_r = x_r.view(l, -1)  # 消融
        out_r = self.fc(out_r)     # 全连接
        out_r = self.dropout(out_r)
        out_r = torch.tanh(out_r)
        
        """字信息向量"""
        out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # Bert
        out, _ = self.lstm(out)  # Bil-stm层
        '''Attention'''
        out = torch.cat([out_a[:, 1:, :, :], out[:, 1:, :].unsqueeze(-2)], dim=-2)
        Q = self.query_layer(out)   # 查询张量
        K = self.key_layer(out)     # 键张量
        V = self.value_layer(out)   # 值张量
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed ** 0.5)
        attention_weights = nn.Softmax(dim=-1)(scores)
        out = torch.matmul(attention_weights, V)
        out = torch.mean(out, dim=-2)
        '''full connected--THR'''
        out = torch.sum(out, dim=1)  
        out = F.relu(out)
        out = self.fc4(out)
        out = self.dropout(out)
        out = torch.tanh(out)
        '''full connected--LCLS'''
        pooled = self.fc1(pooled)
        out = self.dropout(out)
        pooled = torch.tanh(pooled)
        # 三方结合
        out = torch.cat((out, out_r, pooled), dim=1)
        # out = torch.cat((out, pooled), dim=1)  # 消融
        '''full connected--all'''
        out = self.fc2(out)
        return out
