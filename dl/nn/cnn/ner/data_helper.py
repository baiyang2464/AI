#coding:utf-8
import numpy as np
SEQ_LEN = 100

class DataUtil:
    def __init__(self):
        self._word_emb = []
        self._word2id = {}
        self._id2word = {}
        self._label2id = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-ORG": 3, "I-ORG": 4, "B-PER": 5, "I-PER": 6}
        self._id2label = {0: "O", 1: "B-LOC", 2: "I-LOC", 3: "B-ORG", 4: "I-ORG", 5: "B-PER", 6: "I-PER"}
        self._data = []
        self.add("PAD")
        self.add("UNK")

    def size(self):
        return len(self._data)

    def id2label(self, id):
        return self._id2label[id]

    def word2id(self, token):
        if token in self._word2id:
            return self._word2id[token]
        return self._word2id["UNK"]

    def add(self, token):
        if token not in self._word2id:
            id = len(self._word2id)
            self._word2id[token] = id
            self._id2word[id] = token
            return id
        return self._word2id[token]

    def load_emb(self, filename):
        for line in open(filename, 'r'):
            line = line.strip() #去掉一行后面的空格
            v = line.split() #按空格将一行元素分割
            if len(v) == 2: #文件首行，(字典大小， 每个汉字向量的维度)
                size, dim = int(v[0]), int(v[1])
                self._word_emb = np.zeros(shape=[size, dim], dtype=float)
                continue
            id = self.add(v[0])
            self._word_emb[id] = np.array(v[1:], dtype=float)

    def load_data(self, filename):
        data = []
        word_ids = [] #存放句子的字对应的id序列
        label_ids = [] #存放句子的标签对应的id序列
        word_seq = [] #存放句子原始字序列
        for line in open(filename, 'r'):
            line = line.strip()
            if len(line)==0:
                continue
            word, label = line.split()
            if word == "<S>":
                continue
            elif word == "<E>":
                data.append((word_ids, label_ids, word_seq))
                word_seq = []
                word_ids = []
                label_ids = []
            else:
                word_seq.append(word)
                word_ids.append(self.word2id(word))
                label_ids.append(self._label2id[label])
        self._data = data
        return data

    def gen_mini_batch(self, batch_size):
        data_size = len(self._data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            x_batch = []
            label_batch = []
            for i in range(batch_size):
                id = batch_indices[i]

                pad_lst = (SEQ_LEN - len(self._data[id][0])) * [0] #将数据进行补齐，所有batch数据都是相同的句长
                x_pad = self._data[id][0][0:SEQ_LEN] + pad_lst
                label_pad = self._data[id][1][0:SEQ_LEN] + pad_lst

                x_batch.append(x_pad)
                label_batch.append(label_pad)

            yield (x_batch, label_batch)