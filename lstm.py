import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMquestion(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=1024, max_seq_ln=15):
        super(LSTMquestion, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = (torch.zeroes(1, 1, hidden_dim),
                        torch.zeroes(1, 1, hidden_dim))
        self.max_seq_ln = max_seq_ln

    def forward(self, question):
        embeds = self.embeddings(question)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(question), 1, -1), self.hidden)
        )
        return lstm_out
        # classes = do linear mapping(lstm_out.view(len(word), -1))
        # scores = do F.log_softmax(classes, dim=1)
                        

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# print(hidden)
