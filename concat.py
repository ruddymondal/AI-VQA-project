import torch
import torch.nn as nn
import torchvision.models as models

class Concat(nn.Module):
    def __init__(self, concat_size):
        super(Concat, self).__init__()
        self.fc1 = nn.Linear(concat_size, 1024)
        self.fc2 = nn.Linear(1024,3000)

    def forward(self, ft_output,lstm_output):
        concat = torch.concat((ft_output, lstm_output),0)
        fc1 = self.fc1(concat)
        relu = F.relu(fc1)
        fc2 = self.fc2(relu)
        output = F.log_softmax(fc2)
        return output

        
