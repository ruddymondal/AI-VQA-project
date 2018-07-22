import argparse
import torch
import numpy as np 
import os
import pickle 
from torchvision import transforms 
from dataloader import *
from net import *
import config

use_cuda = torch.cuda.is_available()

class Generator():
    def __init__(self):
        args = config.settings

        # Image preprocessing
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        coco = CocoDataset(root='data/val2014',
                           anns_json='data/v2_mscoco_val2014_annotations.json',
                           qns_json='data/v2_OpenEnded_mscoco_val2014_questions.json',
                           vocab_file='',
                           index_file='',
                           transform=transform)

        # Data loader for COCO dataset
        self.data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args["batch_size"], num_workers=args["num_workers"], collate_fn=collate_fn)

        self.ans_vocab = coco.ans_vocab

        # Build models
        net = nn.DataParallel(Net(len(coco.qn_vocab)))
        if use_cuda:
            net = net.cuda()
        net.eval()

    def get_image(i):
        (images, qns), targets, qn_lengths = self.data_loader[i]
        if use_cuda:
            self.images = images.cuda()
            self.qns = qns.cuda()
        else:
            self.images = images
            self.qns = qns
        ft_output = self.cnn_model(self.images)
        lstm_output = self.lstmqn(self.qns, qn_lengths)
        concat_ft = torch.cat((ft_output,lstm_output), 1)
        concat_dim = concat_ft.shape[1]
        concat = Concat(concat_dim)
        concat.load_state_dict(torch.load(args.concat_path))
        concat.eval()
        outputs = concat(concat_ft)
        answer = outputs.data.topk(top_k, dim=1)
        answer = self.ans_vocab.idx2word[answer]
        return outputs, answer

    def get_question():
        print(self.images)
        print(self.qns)
