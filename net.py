import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from models import resnet
from dataloader import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = resnet.resnet152(pretrained=True)

    def forward(self, images):
        imgs = self.resnet(images)
        return F.normalize(imgs)


class LSTMqn(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=1024, drop=0.5):
        super(LSTMqn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        __, (_, c) = self.lstm(packed)
        return c.squeeze(0)


class Concat(nn.Sequential):
    def __init__(self, concat_dim):
        super(Concat, self).__init__()
        self.add_module('fc1', nn.Linear(concat_dim, 1024))
        self.add_module('relu', nn.ReLU())
        self.add_module('fc2', nn.Linear(1024, 3000))


class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
        self.cnn = CNN()
        self.lstm = LSTMqn(vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, imgs, qns, qn_lengths):
        img_features = self.cnn(imgs)
        qns = self.lstm(qns, qn_lengths)
        combined = torch.cat([img_features, qns], 1)

        concat = Concat(combined.shape[1])
        answer = concat(combined)
        return answer
