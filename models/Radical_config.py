# 人工智能
# 项目：radical model
# 开发人：高云龙
# 开发时间：2023-03-14  16:43
# 开发工具：PyCharm
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


def create_dict(f):
    radical_dict = {}  # 构建查找部首的字典
    a = 0
    with open(f, "r", encoding="utf-8") as f:
        text = f.readlines()
        for i in text:
            a += 1
            radical_dict[i[0]] = i[2]
    return radical_dict


# 查找字的偏旁部首
def Find_Radical(inpt, dict_r, seq_len):
    for j in range(seq_len):
        try:
            inpt[j] = dict_r[inpt[j]]
        except:
            if inpt[j] == '<PAD>':
                inpt[j] = '卥'
            else:
                inpt[j] = '虌'
    return inpt

