# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
# from models.Focal_loss import FocalLoss


# def focal_loss(logits,label,a,r):
#     '''
#     :param logits: [batch size,num_classes] score value
#     :param label: [batch size,num_classes] gt value
#     :param a: generally be 0.5
#     :param r: generally be 0.9
#     :return: scalar loss value of a batch
#     '''
#     p_1 = - a*np.power(1-logits,r)*np.log2(logits)*label
#     p_0 = - (1-a)*np.power(logits,r)*np.log2(1-logits)*(1-label)
#     return (p_1 + p_0).sum() 


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels, radicals) in enumerate(train_iter):
            outputs = model(trains, radicals)
            model.zero_grad()
            # focal_loss = FocalLoss(20)
            loss = F.cross_entropy(outputs, labels)
            # loss = focal_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    # print("Confusion Matrix...")
    # print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return test_acc, test_loss


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, radicals in data_iter:
            outputs = model(texts, radicals)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


# if __name__ == '__main__':
#     from importlib import import_module
#     dataset = '/data/CNT'  # CNT数据集
#     # dataset = '/data/THUCNews'  # THUCNEWS数据集
#     # dataset = '/data/FCT'  # FCT数据集
#     # dataset = '/data/TNT'  # TNT数据集
  
#     model_name = 'CRGIM'  # bert
#     x = import_module('models.' + model_name)
#     config = x.Config(dataset)
#     np.random.seed(1)
#     torch.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     torch.backends.cudnn.deterministic = True  # 保证每次结果一样
#     from utils import build_dataset, build_iterator, get_time_dif
#     start_time = time.time()
#     print("Loading data...")
#     train_data, dev_data, test_data = build_dataset(config)
#     test_iter = build_iterator(test_data, config)
#     # train_iter = build_iterator(train_data, config)
#     # dev_iter = build_iterator(dev_data, config)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)
#     model = x.Model(config).to(config.device)
#     test(config, model, test_iter)