import torch
import torch.nn as nn
import os
import re
import jieba
from tqdm import trange
from gensim.models import Word2Vec
import numpy as np


class Net(nn.Module):
    def __init__(self, onehot_num):
        super(Net, self).__init__()
        onehot_size = onehot_num
        embedding_size = 256
        n_layer = 2
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)# 编码
        self.encode = torch.nn.Sequential(nn.Linear(onehot_size, embedding_size),nn.Dropout(0.5),nn.ReLU())# 解码
        self.decode = torch.nn.Sequential(nn.Linear(embedding_size, onehot_size),nn.Dropout(0.5),nn.Sigmoid())

    def forward(self, x):# 入
        em = self.encode(x).unsqueeze(dim=1)# 出
        out, (h, c) = self.lstm(em)
        res = 2*(self.decode(out[:,0,:])-0.5)
        return res

def read_data():
    with open('./白马啸西风.txt', "r", encoding='utf-8') as f:
        file_read = f.readlines()
        all_text = ""
        for line in file_read:
            line = re.sub('\s','', line)
            line = re.sub('！','。', line)
            line = re.sub('？','。', line)# 保留句号
            line = re.sub('[\u0000-\u3001]', '', line)
            line = re.sub('[\u3003-\u4DFF]', '', line)
            line = re.sub('[\u9FA6-\uFFFF]', '', line)
            all_text += line
        f.close()
    return all_text

def train():
    embed_size = 1024
    epochs = 50
    end_num = 10

    print("读取数据开始")
    all_text = read_data()
    text_terms = list()
    for text_line in all_text.split('。'):
        seg_list = list(jieba.cut(text_line, cut_all=False))  # 使用精确模式
        if len(seg_list) < 5:
            continue
        seg_list.append("END")
        text_terms.append(seg_list)
    print("读取数据结束")
    # 获得word2vec模型
    print("开始计算向量")
    if not os.path.exists('model.model'):
        print("开始构建模型")
        model = Word2Vec(sentences=text_terms,sg=0, vector_size=embed_size, min_count=1, window=10, epochs=10)
        print("模型构建完成")
        model.save('model.model')
    print("模型已保存")
    print("开始训练")
    sequences = text_terms
    vec_model = Word2Vec.load('model.model')
    model = Net(embed_size).cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    for epoch_id in range(epochs):
        for idx in trange(0, len(sequences) // end_num - 1):
            seq = []
            for k in range(end_num):
                seq += sequences[idx + k]
            target = []
            for k in range(end_num):
                target += sequences[idx + end_num + k]
            input_seq = torch.zeros(len(seq), embed_size)
            for k in range(len(seq)):
                input_seq[k] = torch.tensor(vec_model.wv[seq[k]])
                target_seq = torch.zeros(len(target), embed_size)
            for k in range(len(target)):
                target_seq[k] = torch.tensor(vec_model.wv[target[k]])
            all_seq = torch.cat((input_seq, target_seq), dim=0)
            optimizer.zero_grad()
            out_res = model(all_seq[:-1].cuda())
            f1 = ((out_res[-target_seq.shape[0]:] ** 2).sum(dim=1)) ** 0.5
            f2 = ((target_seq.cuda() ** 2).sum(dim=1)) ** 0.5
            loss = (1 - (out_res[-target_seq.shape[0]:] * target_seq.cuda()).sum(dim=1) / f1 / f2).mean()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                print("loss: ", loss.item(), " in epoch ", epoch_id, " res: ",out_res[-target_seq.shape[0]:].max(dim=1).indices, target_seq.max(dim=1).indices)
        state = {"models": model.state_dict()}
        torch.save(state, "model" + str(epoch_id) + ".pth")

def read_test_data():
    name = 'test.txt'
    with open(name, "r", encoding='utf-8') as f:
        file_read = f.readlines()
        all_text = ""
        for line in file_read:
            line = re.sub('\s', '', line)
            line = re.sub('！', '。', line)
            line = re.sub('？', '。', line)
            line = re.sub('，', '。', line)# 保留句号
            line = re.sub('[\u0000-\u3001]', '', line)
            line = re.sub('[\u3003-\u4DFF]', '', line)
            line = re.sub('[\u9FA6-\uFFFF]', '', line)
            all_text += line
    return all_text

def test():
    embed_size = 1024
    print("start read test data")
    text = read_test_data()
    text_terms = list()
    for text_line in text.split('。'):
        seg_list = list(jieba.cut(text_line,cut_all=False))  # 使用精确模式
        if len(seg_list) < 5:
            continue
        seg_list.append("END")
        text_terms.append(seg_list)
    print("end read data")
    checkpoint = torch.load("model" + str(49) + ".pth")

    model = Net(embed_size).eval().cuda()
    model.load_state_dict(checkpoint["models"])
    vec_model = Word2Vec.load('model.model')

    seqs = []
    for sequence in text_terms:
        seqs += sequence

    input_seq = torch.zeros(len(seqs), embed_size).cuda()
    result = ""
    with torch.no_grad():
        for k in range(len(seqs)):
            input_seq[k] = torch.tensor(vec_model.wv[seqs[k]])
        end_num = 0
        length = 0
        while end_num < 10 and length < 2000:
            print("length: ", length)
            out_res = model(input_seq.cuda())[-2:-1]
            key_value = vec_model.wv.most_similar(positive=np.array(out_res.cpu()), topn=20)
            key = key_value[np.random.randint(20)][0]
            if key == "END":
                result += "。"
                end_num += 1
            else:
                result += key
            length += 1
            input_seq = torch.cat((input_seq, out_res), dim=0)
    print(result)

if __name__ == "__main__":
#    train()
    test()

