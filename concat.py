import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
from dataloader import *

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
	
	def forward(self, images):
		with torch.no_grad():
			ft_output = self.resnet(images)
		self.ft_output = F.normalize(ft_output)
		return self.ft_output

torch.manual_seed(1)

class LSTMquestion(nn.Module):
	def __init__(self, vocab_size, embedding_dim=300, hidden_dim=1024, drop=0.5):
		super(LSTMquestion, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.drop = nn.Dropout(drop)
		self.tanh = nn.Tanh()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
		
		
		self._init_lstm(self.lstm.weight_ih_l0)
		self._init_lstm(self.lstm.weight_hh_l0)
		self.lstm.bias_ih_l0.data.zero_()
		self.lstm.bias_hh_l0.data.zero_()
		
		init.xavier_uniform_(self.embedding.weight)
	
	def _init_lstm(self, weight):
		for w in weight.chunk(4, 0):
			init.xavier_uniform_(w)

	def forward(self, q, q_len):
		embedded = self.embedding(q)
		tanhed = self.tanh(self.drop(embedded))
		packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
		_, (_, c) = self.lstm(packed)
		self.lstm_output = c.squeeze(0)
		return self.lstm_output

class Concat(nn.Module):
	def __init__(self, concat_dim):
		super(Concat, self).__init__()
		self.fc1 = nn.Linear(concat_dim, 1024)
		self.fc2 = nn.Linear(1024,3000)

	def forward(self, concat_ft):
		fc1 = self.fc1(concat_ft)
		relu = F.relu(fc1)
		fc2 = self.fc2(relu)
		output = F.log_softmax(fc2)
		return output
