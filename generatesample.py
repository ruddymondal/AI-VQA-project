import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from net import CNN, LSTMqn, Concat
from PIL import Image
import config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator():
    def __init__(self):
        args = config.settings

        # Image preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
        
        coco = CocoDataset(root='data/train2014',
                        anns_json=args["anns_json"],
                        qns_json=args["qns_json"],
                        vocab_file=args["vocab_path"],
                        index_file=args["index_file"],
                        transform=transform)

        # Data loader for COCO dataset
        self.data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], collate_fn=collate_fn)

        # Load vocabulary wrapper
        with open(args["vocab_path"], 'rb') as f:
            vocab = pickle.load(f)

        # Build models
        self.cnn_model = CNN().eval()
        self.lstmqn = LSTM(len(coco.qn_vocab).eval()

        self.lstmqn.load_state_dict(torch.load(args["lstmqn_path"]))
        self.lstmqn.eval()

    def get_image(i):
        (images, qns), targets = self.data_loader[i]
        self.images = images.to(device)
        self.qns = qns.to(device)
        ft_output = self.cnn_model(self.images)
        lstm_output = self.lstmqn(self.qns, qn_lengths)
        concat_ft = torch.cat((ft_output,lstm_output), 1)
        concat_dim = concat_ft.shape[1]
        concat = Concat(concat_dim)
        concat.load_state_dict(torch.load(args.concat_path))
        concat.eval()
        outputs = concat(concat_ft)
        answer = outputs.data.topk(top_k, dim=1)
        answer = vocab.idx2word[answer]
        return outputs, answer

    def get_question():
        print(self.images)
        print(self.qns)
            
            
            
            
                  
    
     
