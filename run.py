# coding: UTF-8
import time
import torch
import numpy as np
import torch.nn as nn
from train_eval import train, init_network
from torch.nn.parallel import DistributedDataParallel
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: ASCIM')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 这里输入你的GPU_id
torch.cuda.empty_cache()


if __name__ == '__main__':
    # dataset = '/data/CNT'  # CNT数据集
    dataset = '/data/THUCNews'  # THUCNEWS数据集
    # dataset = '/data/FCT'  # FCT数据集
    # dataset = '/data/TNT'  # TNT数据集
    
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # 加载数据集
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # 加载模型
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)