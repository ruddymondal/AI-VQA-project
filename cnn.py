import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        resnet = models.resnet152(pretrained=True)

    def forward(self, images):
        with torch.no_grad():
            ft_output = self.resnet(images)
        ft_output = F.normalize(ft_output)
        return ft_output
